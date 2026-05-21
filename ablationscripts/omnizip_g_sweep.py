"""
omnizip_g_sweep.py
==================
ABLATION: Sweep g ∈ {1, 2, 3, 5, 8}

`g` is the maximum number of non-anchor, non-dominant audio tokens that get
merged into each contextual anchor.

Inside omnizip_audio_attn(), after selecting dominant + contextual anchor tokens:
    pool_global = remaining non-anchor tokens  (to be merged or dropped)
    For each anchor c:
        scores_c[j] = max_k( cosine_sim(a_j, v_k) )   for all video tokens v_k
                    [or cosine_sim(a_j, a_c) if no video]
        topg = min(g, cand_c.numel())
        chosen = top-topg tokens by score
        w      = softmax(scores_c[chosen])
        new_anchor_c = (anchor_c + sum_j(w_j * a_j)) / (1 + sum(w))

Effect:
  g=1: minimal merging — each anchor absorbs at most 1 neighbor; least smoothing
  g=8: aggressive merging — up to 8 neighbors compressed into each anchor;
       more information preserved per anchor but more blending/averaging

Note: g only affects AUDIO merging, not video token selection. The video
compression pipeline is identical across g values. The key question is whether
more aggressive audio token merging improves or hurts the audio-guidance signal
fed to video token selection.

Fixed params: rho_video=0.6, rho_audio=0.3, contextual_ratio=0.05

Usage:
    python omnizip_g_sweep.py --value 3 \\
        --metadata /data/armaan/purs/metadata.json \\
        --videos   /data/armaan/purs/videos

    python omnizip_g_sweep.py --all_values ...
"""

import argparse
import gc
import glob
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)
OMNIZIP_DIR         = os.path.join(_REPO_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
for _p in (OMNIZIP_DIR, QWEN_OMNI_UTILS_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from mcq_answer_parse import parse_answer

# ── Constants ──────────────────────────────────────────────────────────────────
ABLATION_NAME = "omnizip_g_sweep"

SWEEP_VALUES = [1, 2, 3, 5, 8]

DEFAULT_MODEL            = "/data/armaan/models/Qwen2.5-Omni-7B"
DEFAULT_FPS              = 2.0
DEFAULT_MAX_PIXELS       = 100352
DEFAULT_MAX_FRAMES_VMME  = 768
DEFAULT_MAX_FRAMES_OTHER = 128
DEFAULT_MAX_NEW_TOKENS   = 256
DEFAULT_TEMPERATURE      = 0.1

FIXED_RHO_VIDEO        = 0.6
FIXED_RHO_AUDIO        = 0.3
FIXED_CONTEXTUAL_RATIO = 0.05

# ── Prompt builders ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
SYSTEM_MCQ_SUFFIX = (
    "For multiple-choice questions, reply with only one letter: A, B, C, or D. "
    "Do not explain, do not ask follow-up questions, and do not add text after the letter."
)
_VIDEO_MME_OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)
_VIDEO_MME_POST_PROMPT = "The best answer is:"
_WORLD_SENSE_SYS = (
    "Carefully watch this video and pay attention to every detail. "
    "Based on your observations, select the best option that accurately addresses the question."
)
_WORLD_SENSE_FRAMES_AUDIO = (
    "\nThese are the frames of a video and the corresponding audio. "
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
)


def _fmt_choices(choices):
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))


def build_prompt(dataset, question, choices):
    ds = (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")
    opts = _fmt_choices(choices)
    if ds in {"video-mme", "videomme"}:
        return _VIDEO_MME_OPTION_PROMPT + "\n" + question + "\n" + opts + "\n" + _VIDEO_MME_POST_PROMPT
    if ds == "worldsense":
        return _WORLD_SENSE_SYS + _WORLD_SENSE_FRAMES_AUDIO + question + "\n" + "\n".join(choices) + "\n"
    if ds in {"daily-omni", "dailyomni"}:
        return (
            "Listen and watch the video carefully. Select the best answer to the following multiple-choice question. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + question + "\n" + opts + "\n" + _VIDEO_MME_POST_PROMPT
        )
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        + question + "\n" + opts + "\n" + _VIDEO_MME_POST_PROMPT
    )


# ── Video utilities ────────────────────────────────────────────────────────────
def resolve_video_path(file_field, videos_dir):
    if os.path.exists(file_field):
        return file_field
    normalized = file_field.replace("\\", "/")
    filename = normalized.split("/")[-1]
    stem = filename.rsplit(".", 1)[0]
    cand = os.path.join(videos_dir, filename)
    if os.path.exists(cand):
        return cand
    for ext in ("mp4", "mkv", "webm", "avi"):
        m = glob.glob(os.path.join(videos_dir, "**", f"{stem}.{ext}"), recursive=True)
        if m:
            return m[0]
    return None


def check_video_has_audio(path):
    try:
        import av
        c = av.open(path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False


# ── CUDA timing ───────────────────────────────────────────────────────────────
def cuda_time_ms(fn):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out


# ── Tee logger ────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path, label=""):
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")
        self.log.write(f"\n{'='*60}\n{label} {datetime.now()}\n{'='*60}\n")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.log.close()


class StderrTee:
    def __init__(self, path):
        self.terminal = sys.__stderr__
        self.log = open(path, "a", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(model_path, g):
    print(f"[{ABLATION_NAME}] Loading model: {model_path}")
    print(f"  g={g}  rho_video={FIXED_RHO_VIDEO}  rho_audio={FIXED_RHO_AUDIO}  "
          f"contextual_ratio={FIXED_CONTEXTUAL_RATIO}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.thinker.omnizip_config = {
        "rho_audio":        FIXED_RHO_AUDIO,
        "rho_video":        FIXED_RHO_VIDEO,
        "g":                g,
        "contextual_ratio": FIXED_CONTEXTUAL_RATIO,
    }
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    alloc_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"  Model loaded. VRAM: {alloc_gb:.2f} GB")
    return model, processor, alloc_gb


def unload_model(model, processor):
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Inference ──────────────────────────────────────────────────────────────────
def run_one(model, processor, video_path, dataset, question, choices,
            use_audio, fps, max_pixels, max_frames_vmme, max_frames_other,
            max_new_tokens, temperature):
    ds = (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")
    max_frames = max_frames_vmme if ds in {"video-mme", "videomme"} else max_frames_other

    system_text = SYSTEM_PROMPT + " " + SYSTEM_MCQ_SUFFIX
    prompt = build_prompt(dataset, question, choices)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps,
             "max_pixels": max_pixels, "max_frames": max_frames},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio)

    if not videos or videos[0] is None or videos[0].shape[0] == 0:
        raise ValueError("0 video frames decoded")

    num_input_frames = int(videos[0].shape[0])
    model.thinker.nframes = num_input_frames

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio,
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)

    gen_in = {k: v for k, v in inputs.items() if k not in {"images", "return_tensors", "text"}}
    do_sample = temperature > 0
    gen_kw = dict(
        use_audio_in_video=use_audio,
        return_audio=False,
        thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=do_sample,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kw["thinker_temperature"] = temperature

    with torch.no_grad():
        prefill_ms, _ = cuda_time_ms(
            lambda: model.generate(**gen_in, **dict(gen_kw, thinker_max_new_tokens=1))
        )
    with torch.no_grad():
        e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))

    seq = raw_out.sequences if hasattr(raw_out, "sequences") else raw_out
    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, seq)]
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, num_input_frames, round(prefill_ms, 2), round(e2e_ms, 2)


# ── Per-value runner ───────────────────────────────────────────────────────────
def run_value(g, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    console_path = os.path.join(out_dir, "console.log")
    stderr_path  = os.path.join(out_dir, "stderr.log")
    results_path = os.path.join(out_dir, "results.jsonl")
    vram_path    = os.path.join(out_dir, "vram_log.jsonl")
    summary_path = os.path.join(out_dir, "run_summary.json")

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    tee  = Tee(console_path, label=f"g={g}")
    stee = StderrTee(stderr_path)
    sys.stdout, sys.stderr = tee, stee

    try:
        model, processor, model_alloc_gb = load_model(args.model, g)
        meta = json.loads(Path(args.metadata).read_text())
        runnable = [e for e in meta if e.get("questions")]
        print(f"  {len(runnable)} entries to run")

        correct = total = skipped = 0
        q_idx = 0

        with open(results_path, "w") as rf, open(vram_path, "w") as vf:
            for entry in runnable:
                vp = resolve_video_path(entry["file"], args.videos)
                if vp is None:
                    print(f"  SKIP: video not found — {entry['file']}")
                    skipped += 1
                    continue

                use_audio = check_video_has_audio(vp) and not args.no_audio

                for q in entry["questions"]:
                    question  = q["question"]
                    choices   = q["choices"]
                    answer    = q["answer"].strip().upper()
                    dataset   = entry.get("dataset", "")
                    task_type = q.get("task_type", entry.get("task_type", ""))

                    before_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                    try:
                        pred, decoded, nframes, prefill_ms, e2e_ms = run_one(
                            model, processor, vp, dataset, question, choices,
                            use_audio, args.fps, args.max_pixels,
                            args.max_frames_vmme, args.max_frames_other,
                            args.max_new_tokens, args.temperature,
                        )
                        after_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                        delta = round(after_alloc - before_alloc, 4)
                        status = "ok"
                        err_msg = None
                    except Exception as exc:
                        tb = traceback.format_exc()
                        print(f"  ERROR [{dataset}/{task_type}]: {exc}")
                        pred, decoded = "ERROR", str(exc)
                        nframes = prefill_ms = e2e_ms = 0
                        after_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                        delta = round(after_alloc - before_alloc, 4)
                        status = "error"
                        err_msg = tb

                    is_correct = (pred.strip().upper() == answer)
                    if pred != "ERROR":
                        correct += int(is_correct)
                        total += 1
                    mark = "✓" if is_correct else "✗"
                    print(f"  [{mark}] {dataset}/{task_type} pred={pred} ans={answer} "
                          f"prefill={prefill_ms:.0f}ms e2e={e2e_ms:.0f}ms")

                    result = {
                        "dataset": dataset, "task_type": task_type,
                        "video": entry["file"], "question": question,
                        "answer": answer, "prediction": pred, "correct": is_correct,
                        "orig_nframes": nframes, "used_nframes": nframes,
                        "prefill_ms": prefill_ms, "e2e_ms": e2e_ms,
                        "vram_alloc_delta_gb": delta,
                        "method": ABLATION_NAME,
                        "config": {
                            "rho_video": FIXED_RHO_VIDEO, "rho_audio": FIXED_RHO_AUDIO,
                            "g": g, "contextual_ratio": FIXED_CONTEXTUAL_RATIO,
                        },
                        "ablation_value": g,
                        "status": status,
                    }
                    if err_msg:
                        result["error"] = err_msg[:500]
                    rf.write(json.dumps(result) + "\n"); rf.flush()

                    vram_entry = {
                        "question_idx": q_idx, "dataset": dataset,
                        "before_alloc_gb": round(before_alloc, 4),
                        "after_alloc_gb": round(after_alloc, 4),
                        "delta_gb": delta,
                    }
                    vf.write(json.dumps(vram_entry) + "\n"); vf.flush()
                    q_idx += 1

        acc = correct / total if total else 0.0
        summary = {
            "ablation": ABLATION_NAME, "value": g,
            "config": {"rho_video": FIXED_RHO_VIDEO, "rho_audio": FIXED_RHO_AUDIO,
                       "g": g, "contextual_ratio": FIXED_CONTEXTUAL_RATIO},
            "correct": correct, "total": total, "accuracy": round(acc, 4),
            "skipped": skipped,
        }
        Path(summary_path).write_text(json.dumps(summary, indent=2))
        print(f"\n  g={g}: {correct}/{total} = {acc:.2%}")
        return summary

    finally:
        unload_model(model, processor)
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        tee.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=f"Ablation: {ABLATION_NAME}")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--metadata", default="/data/armaan/purs/metadata.json")
    parser.add_argument("--videos",   default="/data/armaan/purs/videos")
    parser.add_argument("--output_root", default="/data/armaan/purs/ablation_outputs")
    parser.add_argument("--value",    type=int, default=None,
                        help="Single g value (e.g. 3).")
    parser.add_argument("--all_values", action="store_true")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--fps",              type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels",       type=int,   default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max_frames_vmme",  type=int,   default=DEFAULT_MAX_FRAMES_VMME)
    parser.add_argument("--max_frames_other", type=int,   default=DEFAULT_MAX_FRAMES_OTHER)
    parser.add_argument("--max_new_tokens",   type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature",      type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    if args.value is None and not args.all_values:
        parser.error("Specify --value <int> or --all_values")

    values = SWEEP_VALUES if args.all_values else [args.value]

    sweep_dir = os.path.join(args.output_root, ABLATION_NAME)
    os.makedirs(sweep_dir, exist_ok=True)

    all_summaries = []
    for v in values:
        out_dir = os.path.join(sweep_dir, f"g_{v}")
        print(f"\n{'='*60}")
        print(f"[{ABLATION_NAME}] g={v}  → {out_dir}")
        print(f"{'='*60}")
        summary = run_value(v, args, out_dir)
        all_summaries.append(summary)

    sweep_summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    Path(sweep_summary_path).write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[{ABLATION_NAME}] Sweep complete. Summary → {sweep_summary_path}")
    for s in all_summaries:
        print(f"  g={s['value']}: {s['correct']}/{s['total']} = {s['accuracy']:.2%}")


if __name__ == "__main__":
    main()
