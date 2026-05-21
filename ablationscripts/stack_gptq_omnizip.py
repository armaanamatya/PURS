"""
stack_gptq_omnizip.py
Ablation: GPTQ weight quantization (Int4) + OmniZip token compression.

GPTQ compresses weight matrices to Int4 (4x fewer bytes for weights, no sequence change).
OmniZip compresses input tokens (no weight change).
The two are orthogonal: GPTQ changes W tensors; OmniZip changes the forward pass logic.

To stack: load the GPTQ model USING the OmniZip model class
(OmniZip's class overrides the thinker forward; GPTQ just provides quantized weights).

Conditions:
  gptq_only                – GPTQ Int4 model, no OmniZip
  omnizip_only             – FP16 baseline + OmniZip (rho_v=0.6, rho_a=0.3)
  gptq_omnizip             – GPTQ + OmniZip (rho_v=0.6, rho_a=0.3)
  gptq_omnizip_aggressive  – GPTQ + OmniZip (rho_v=0.3, rho_a=0.3)

Usage:
  python stack_gptq_omnizip.py --metadata /data/armaan/purs/metadata.json \
      --videos /data/armaan/purs/videos --output_base ablation_outputs/stack_gptq_omnizip
  python stack_gptq_omnizip.py ... --condition gptq_only
"""

import argparse
import json
import os
import glob
import sys
import time
import traceback
import random
from datetime import datetime
from pathlib import Path

import torch

# ── sys.path setup ────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PURS_ROOT = os.path.join(_REPO_ROOT, "..")
OMNIZIP_DIR = os.path.join(_PURS_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, QWEN_OMNI_UTILS_SRC)
sys.path.insert(0, _PURS_ROOT)

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration as OmniZipModel
from transformers import Qwen2_5OmniForConditionalGeneration as VanillaModel
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ── Constants ─────────────────────────────────────────────────────────────────
BASELINE_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
GPTQ_MODEL_PATH     = "/data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4"

DEFAULT_FPS = 2.0
DEFAULT_MAX_PIXELS = 100352
DEFAULT_MAX_FRAMES_VIDEOMME = 768
DEFAULT_MAX_FRAMES_OTHER = 128
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1

CONDITIONS = {
    "gptq_only": {
        "model_path": GPTQ_MODEL_PATH,
        "use_omnizip_class": False,
        "use_omnizip_config": False,
        "rho_video": None, "rho_audio": None, "g": None, "contextual_ratio": None,
        "dtype": "float16",  # GPTQ requires float16
    },
    "omnizip_only": {
        "model_path": BASELINE_MODEL_PATH,
        "use_omnizip_class": True,
        "use_omnizip_config": True,
        "rho_video": 0.6, "rho_audio": 0.3, "g": 3, "contextual_ratio": 0.05,
        "dtype": "float16",
    },
    "gptq_omnizip": {
        "model_path": GPTQ_MODEL_PATH,
        "use_omnizip_class": True,
        "use_omnizip_config": True,
        "rho_video": 0.6, "rho_audio": 0.3, "g": 3, "contextual_ratio": 0.05,
        "dtype": "float16",  # GPTQ requires float16
    },
    "gptq_omnizip_aggressive": {
        "model_path": GPTQ_MODEL_PATH,
        "use_omnizip_class": True,
        "use_omnizip_config": True,
        "rho_video": 0.3, "rho_audio": 0.3, "g": 3, "contextual_ratio": 0.05,
        "dtype": "float16",
    },
}

# ── Answer parsing ─────────────────────────────────────────────────────────────
try:
    from mcq_answer_parse import parse_answer
except ImportError:
    def parse_answer(text, choices):
        import re
        t = text.strip()
        m = re.search(r'\b([A-D])\b', t)
        if m:
            return m.group(1).upper()
        return t[:1].upper() if t else "A"

# ── CUDA timing ───────────────────────────────────────────────────────────────
def cuda_time_ms(fn):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); out = fn(); e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out

# ── Tee loggers ───────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path, label=""):
        self.terminal = sys.stdout
        self.log = open(path, "a")
        self.log.write(f"\n{'='*60}\nRUN {label}: {datetime.now()}\n{'='*60}\n")
        self.log.flush()

    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
    def isatty(self): return self.terminal.isatty()
    def close(self): self.log.close()


class StderrTee:
    def __init__(self, log_file, terminal):
        self.log = log_file; self.terminal = terminal

    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
    def isatty(self): return self.terminal.isatty()

# ── Prompt builders ───────────────────────────────────────────────────────────
SYSTEM_PROMPT_DEFAULT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
SYSTEM_MCQ_SUFFIX = (
    "For multiple-choice questions, reply with only one letter: A, B, C, or D. "
    "Do not explain, do not ask follow-up questions, and do not add text after the letter."
)

def _format_choice_lines(choices):
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))

def _canon(dataset):
    return (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")

def build_user_prompt_for_dataset(dataset, question, choices):
    d = _canon(dataset)
    opts = _format_choice_lines(choices)
    post = "The best answer is:"
    if d in {"video-mme", "videomme"}:
        head = ("Select the best answer to the following multiple-choice question based on the video and the subtitles. "
                "Respond with only the letter (A, B, C, or D) of the correct option.")
        return f"{head}\n{question}\n{opts}\n{post}"
    if d == "worldsense":
        sys_ws = ("Carefully watch this video and pay attention to every detail. "
                  "Based on your observations, select the best option that accurately addresses the question.")
        frames = ("These are the frames of a video and the corresponding audio. "
                  "Select the best answer to the following multiple-choice question based on the video. "
                  "Respond with only the letter (A, B, C, or D) of the correct option.")
        return f"{sys_ws}\n{frames}\n{question}\n{opts}"
    if d in {"daily-omni", "dailyomni"}:
        head = ("Listen and watch the video carefully. "
                "Select the best answer to the following multiple-choice question. "
                "Respond with only the letter (A, B, C, or D) of the correct option.")
        return f"{head}\n{question}\n{opts}\n{post}"
    return ("Select the best answer to the following multiple-choice question based on the video. "
            f"Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{opts}\n{post}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_video_has_audio(video_path):
    try:
        import av
        c = av.open(video_path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False


def resolve_video_path(file_field, videos_dir):
    if os.path.exists(file_field):
        return file_field
    normalized = file_field.replace("\\", "/")
    if os.path.exists(normalized):
        return normalized
    rel = normalized
    for prefix in ("videos/", "videos\\"):
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):]
            break
    candidate = os.path.normpath(os.path.join(videos_dir, rel))
    if os.path.exists(candidate):
        return candidate
    rel_norm = rel.replace("\\", "/")
    filename = rel_norm.split("/")[-1]
    suffix_matches = [m for m in glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
                      if m.replace("\\", "/").endswith(rel_norm)]
    if suffix_matches:
        return suffix_matches[0]
    basename_matches = glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def capture_vram():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3

# ── Model loader ──────────────────────────────────────────────────────────────
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def load_model_for_condition(cond_cfg):
    model_path = cond_cfg["model_path"]
    dtype = _DTYPE_MAP[cond_cfg["dtype"]]
    is_gptq = "GPTQ" in model_path or "gptq" in model_path.lower()
    print(f"Loading model from {model_path}")
    print(f"  use_omnizip_class={cond_cfg['use_omnizip_class']}  dtype={cond_cfg['dtype']}")

    load_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
    )
    # GPTQ models use their own quantized attention kernels — flash_attention_2
    # can conflict; use sdpa or eager for GPTQ
    if is_gptq:
        load_kwargs["attn_implementation"] = "sdpa"
    else:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    if cond_cfg["use_omnizip_class"]:
        model = OmniZipModel.from_pretrained(model_path, **load_kwargs)
    else:
        model = VanillaModel.from_pretrained(model_path, **load_kwargs)

    if cond_cfg["use_omnizip_config"]:
        model.thinker.omnizip_config = {
            "rho_audio":        cond_cfg["rho_audio"],
            "rho_video":        cond_cfg["rho_video"],
            "g":                cond_cfg["g"],
            "contextual_ratio": cond_cfg["contextual_ratio"],
        }
        print(f"  OmniZip config: rho_video={cond_cfg['rho_video']}  "
              f"rho_audio={cond_cfg['rho_audio']}")

    # Processor always from base model (same tokenizer/image processor)
    processor = Qwen2_5OmniProcessor.from_pretrained(BASELINE_MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model, processor

# ── Inference ─────────────────────────────────────────────────────────────────
_DROP_KEYS = frozenset({"images", "return_tensors", "text"})

def _prepare_inputs(model, inputs):
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)
    return inputs


def run_inference(model, processor, video_path, dataset, question, choices,
                  use_audio, run_cfg):
    prompt = build_user_prompt_for_dataset(dataset, question, choices)
    ds = _canon(dataset)
    max_frames = (run_cfg["max_frames_videomme"] if ds in {"video-mme", "videomme"}
                  else run_cfg["max_frames_other"])

    sys_text = SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path,
             "fps": run_cfg["fps"], "max_pixels": run_cfg["max_pixels"],
             "max_frames": max_frames},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio)

    if not videos or videos[0] is None or videos[0].shape[0] <= 0:
        raise ValueError("Decoded 0 video frames")

    num_frames = int(videos[0].shape[0])
    if hasattr(model, "thinker"):
        model.thinker.nframes = num_frames

    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=use_audio)
    inputs = _prepare_inputs(model, inputs)

    gen_in = {k: v for k, v in inputs.items() if k not in _DROP_KEYS}
    tok = processor.tokenizer
    do_sample = run_cfg["temperature"] > 0
    gen_kw = dict(use_audio_in_video=use_audio, return_audio=False,
                  thinker_max_new_tokens=run_cfg["max_new_tokens"],
                  thinker_do_sample=do_sample,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if do_sample:
        gen_kw["thinker_temperature"] = run_cfg["temperature"]

    torch.cuda.reset_peak_memory_stats()
    e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))

    seq_ids = raw_out.sequences if hasattr(raw_out, "sequences") else raw_out
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, num_frames, round(e2e_ms, 2)

# ── Run one condition ─────────────────────────────────────────────────────────
def run_condition(cond_name, cond_cfg, meta, videos_dir, output_base, run_cfg, no_audio):
    out_dir = os.path.join(output_base, cond_name)
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, "results.jsonl")
    vram_path    = os.path.join(out_dir, "vram_log.jsonl")
    console_path = os.path.join(out_dir, "console.log")
    stderr_path  = os.path.join(out_dir, "stderr.log")
    summary_path = os.path.join(out_dir, "run_summary.json")

    _old_stdout = sys.stdout
    _old_stderr = sys.stderr
    tee = Tee(console_path, label=cond_name)
    sys.stdout = tee
    _sf = open(stderr_path, "a")
    sys.stderr = StderrTee(_sf, sys.__stderr__)

    print(f"\nCondition: {cond_name}")
    print(f"Config: {cond_cfg}")

    model, processor = load_model_for_condition(cond_cfg)
    model_loaded_alloc, model_loaded_reserved = capture_vram()
    print(f"Model loaded. VRAM: {model_loaded_alloc:.2f} GB alloc, "
          f"{model_loaded_reserved:.2f} GB reserved")

    runnable = [e for e in meta if e.get("questions")]
    correct = total = skipped = 0
    t_start = time.time()

    with open(results_path, "w") as rf, open(vram_path, "w") as vf:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], videos_dir)
            label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {label}: video not found")
                skipped += 1
                continue

            use_audio = (not no_audio) and check_video_has_audio(video_path)

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset   = entry.get("dataset", "")

                before_alloc, before_reserved = capture_vram()
                try:
                    pred, decoded, n_frames, e2e_ms = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        use_audio, run_cfg)
                    status = "ok"
                except Exception as exc:
                    print(f"  ERROR {label}: {exc!r}")
                    traceback.print_exc()
                    pred, decoded, n_frames, e2e_ms = "ERROR", str(exc), 0, 0.0
                    status = "error"

                peak_alloc  = torch.cuda.max_memory_allocated() / 1024**3
                peak_res    = torch.cuda.max_memory_reserved() / 1024**3
                after_alloc, after_res = capture_vram()

                vf.write(json.dumps({
                    "condition": cond_name, "entry": label, "task_type": task_type,
                    "status": status, "n_frames": n_frames, "e2e_ms": e2e_ms,
                    "model_path": cond_cfg["model_path"],
                    "model_loaded_alloc_gb": round(model_loaded_alloc, 2),
                    "model_loaded_reserved_gb": round(model_loaded_reserved, 2),
                    "before_alloc_gb": round(before_alloc, 2),
                    "before_reserved_gb": round(before_reserved, 2),
                    "peak_alloc_gb": round(peak_alloc, 2),
                    "peak_reserved_gb": round(peak_res, 2),
                    "after_alloc_gb": round(after_alloc, 2),
                    "after_reserved_gb": round(after_res, 2),
                }) + "\n"); vf.flush()

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                rf.write(json.dumps({
                    "condition": cond_name, "dataset": dataset, "task_type": task_type,
                    "question": question, "choices": choices, "answer": answer,
                    "prediction": pred, "correct": is_correct, "reasoning": decoded,
                    "n_frames": n_frames, "e2e_ms": e2e_ms,
                    "model_path": cond_cfg["model_path"],
                    "use_omnizip": cond_cfg["use_omnizip_config"],
                    "rho_video": cond_cfg["rho_video"],
                    "rho_audio": cond_cfg["rho_audio"],
                }) + "\n"); rf.flush()

                sym = "+" if is_correct else "-"
                print(f"  [{sym}] {label} pred={pred} ans={answer} e2e={e2e_ms:.0f}ms")

    elapsed = time.time() - t_start
    acc = correct / total if total else 0.0
    summary = {
        "condition": cond_name, "correct": correct, "total": total,
        "accuracy": round(acc, 4), "skipped": skipped,
        "elapsed_s": round(elapsed, 1),
        "model_path": cond_cfg["model_path"],
        "config": cond_cfg,
    }
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"\n[{cond_name}] Accuracy: {correct}/{total} = {acc:.2%}  ({elapsed:.0f}s)")

    sys.stdout = _old_stdout
    sys.stderr = _old_stderr
    tee.close(); _sf.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ablation: GPTQ + OmniZip stacking")
    parser.add_argument("--metadata",    default="/data/armaan/purs/metadata.json")
    parser.add_argument("--videos",      default="/data/armaan/purs/videos")
    parser.add_argument("--output_base", default="/data/armaan/purs/ablation_outputs/stack_gptq_omnizip")
    parser.add_argument("--condition",   default=None, choices=list(CONDITIONS.keys()),
                        help="Run a single condition (default: all)")
    parser.add_argument("--fps",                 type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels",          type=int,   default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max_frames_videomme", type=int,   default=DEFAULT_MAX_FRAMES_VIDEOMME)
    parser.add_argument("--max_frames_other",    type=int,   default=DEFAULT_MAX_FRAMES_OTHER)
    parser.add_argument("--max_new_tokens",      type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature",         type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no_audio",            action="store_true")
    parser.add_argument("--seed",                type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_cfg = {
        "fps": args.fps, "max_pixels": args.max_pixels,
        "max_frames_videomme": args.max_frames_videomme,
        "max_frames_other": args.max_frames_other,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    os.makedirs(args.output_base, exist_ok=True)
    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} metadata entries")

    to_run = {args.condition: CONDITIONS[args.condition]} if args.condition else CONDITIONS
    all_summaries = {}

    for cond_name, cond_cfg in to_run.items():
        summary = run_condition(cond_name, cond_cfg, meta, args.videos,
                                args.output_base, run_cfg, args.no_audio)
        all_summaries[cond_name] = summary

    cmp_path = os.path.join(args.output_base, "comparison_summary.json")
    with open(cmp_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nComparison summary: {cmp_path}")
    for name, s in all_summaries.items():
        print(f"  {name:<30} acc={s['accuracy']:.4f}  ({s['correct']}/{s['total']})")

if __name__ == "__main__":
    main()
