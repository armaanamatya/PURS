"""
rediprune_mode.py
=================
Ablation: frame-level vs token-level ReDiPrune pruning.

MATH:
  Both modes use the same ReDiPrune score:
    score_i = min_{j∈S} (1 - cos_sim(v_i, v_j)) + alpha * cos_sim(v_i, q_text)

  frame-level:
    - Feature = global average pool of frame pixels: features[i] = mean(frame_i, dims=[H,W])
    - Shape: (N_frames, C)  where C = number of color channels (typically 3)
    - Selects K = round(N_frames * keep_ratio) whole frames
    - Simple, no model internals access needed

  token-level:
    - Feature = flat pixel patches AFTER processor encoding: pixel_values_videos
    - Shape: (N_tokens, patch_dim)  where N_tokens = T_frames * H_patches * W_patches
    - Selects K = round(N_tokens * keep_ratio) individual patch tokens
    - More fine-grained than frame selection
    - Implementation: intercept pixel_values_videos from processor output,
      apply ReDiPrune to those rows, then zero-out or drop unselected tokens by
      replacing them with zero embeddings (null visual tokens) — avoids model surgery.
    - grid_thw is updated to reflect new token count

Fixed: keep_ratio=0.5, alpha=0.5, tau=0.1

Usage:
  # Run both modes:
  python rediprune_mode.py --metadata metadata.json --videos /data/videos

  # Run a single mode (for parallel GPU execution):
  python rediprune_mode.py --metadata metadata.json --videos /data/videos --value frame
  python rediprune_mode.py --metadata metadata.json --videos /data/videos --value token
"""

import argparse
import json
import glob
import os
import sys
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(_REPO_ROOT, "..", "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)
sys.path.insert(0, os.path.join(_REPO_ROOT, ".."))

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from mcq_answer_parse import parse_answer

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
FALLBACK_MODEL_PATH = "/workspace/model"
MODEL_PATH = os.environ.get("QWEN_OMNI_MODEL_PATH") or DEFAULT_MODEL_PATH

DEFAULT_FPS = 2.0
DEFAULT_MAX_PIXELS = 100352
DEFAULT_MAX_FRAMES_VIDEOMME = 768
DEFAULT_MAX_FRAMES_OTHER = 128
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1

FIXED_KEEP_RATIO = 0.5
FIXED_ALPHA = 0.5
FIXED_TAU = 0.1
MODE_VALUES = ["frame", "token"]

SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
SYSTEM_MCQ_SUFFIX = (
    "For multiple-choice questions, reply with only one letter: A, B, C, or D. "
    "Do not explain, do not ask follow-up questions, and do not add text after the letter."
)

MODEL_LOADED_ALLOC_GB = None
MODEL_LOADED_RESERVED_GB = None

# ── ReDiPrune core ───────────────────────────────────────────────────────────

@torch.no_grad()
def rediprune_select(visual_tokens, text_query, keep_ratio, alpha, tau):
    """ReDiPrune greedy farthest-point + text-relevance token selection.

    score_i = min_{j in S} (1 - cos_sim(v_i, v_j)) + alpha * cos_sim(v_i, q_text)
    """
    P, Cv = visual_tokens.shape
    K = max(1, int(round(P * keep_ratio)))
    if K >= P:
        return torch.arange(P, device=visual_tokens.device)

    dev = visual_tokens.device
    vf = visual_tokens.float()
    tq = text_query.float().to(dev)
    Vn = F.normalize(vf, p=2, dim=1)

    tq = tq.unsqueeze(0) if tq.dim() == 1 else tq
    Dq = tq.shape[1]
    if Dq != Cv:
        if Dq > Cv:
            tq = F.adaptive_avg_pool1d(tq.unsqueeze(1), Cv).squeeze(1) if Dq % Cv != 0 else tq.view(1, Cv, -1).mean(2)
        else:
            rep = (Cv + Dq - 1) // Dq
            tq = tq.repeat(1, rep)[:, :Cv]
    Tn = F.normalize(tq, p=2, dim=1)

    rel = (Vn @ Tn.t()).squeeze(1)  # (P,)

    if tau > 0:
        if K < P // 2:
            _, cand = torch.topk(rel, k=min(P, max(K * 3, K + 1)))
        else:
            mask = (rel >= tau)
            cand = torch.where(mask)[0] if mask.sum() >= K else torch.topk(rel, K)[1]
    else:
        cand = torch.arange(P, device=dev)

    div_mat = 1.0 - (Vn[cand] @ Vn[cand].t())
    rel_c = rel[cand]
    Nc = cand.shape[0]
    sel = torch.empty(K, dtype=torch.long, device=dev)
    sel[0] = torch.argmax(rel_c)

    for i in range(1, K):
        min_dist = div_mat[sel[:i]].min(dim=0).values
        score = min_dist + alpha * rel_c
        mask = torch.ones(Nc, dtype=torch.bool, device=dev)
        mask[sel[:i]] = False
        score[~mask] = -float('inf')
        sel[i] = torch.argmax(score)

    out_idx, _ = torch.sort(cand[sel])
    return out_idx


def get_text_query(model, processor, question_text):
    """Encode question text to (D,) query vector via mean-pooled embed_tokens."""
    tokenizer = processor.tokenizer
    tids = tokenizer(question_text, return_tensors="pt", truncation=True, max_length=128)["input_ids"]
    dev = next(model.parameters()).device
    tids = tids.to(dev)
    embed = model.thinker.model.embed_tokens if hasattr(model, "thinker") else model.model.embed_tokens
    with torch.no_grad():
        emb = embed(tids)
    return emb.mean(dim=1).squeeze(0)


# ── Utilities ────────────────────────────────────────────────────────────────

def cuda_time_ms(fn):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out


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


def _canonicalize(ds):
    return (ds or "").strip().lower().replace("_", "-").replace(" ", "-")


def _capture_vram():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3


def check_video_has_audio(path):
    try:
        import av
        c = av.open(path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False


# ── Prompt builders ──────────────────────────────────────────────────────────

def _fmt_choices(choices):
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))


def build_prompt(dataset, question, choices):
    n = _canonicalize(dataset)
    q_block = question + "\n" + _fmt_choices(choices)
    if n in {"video-mme", "videomme"}:
        return (
            "Select the best answer to the following multiple-choice question based on the video "
            "and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + q_block + "\nThe best answer is:"
        )
    if n == "worldsense":
        return (
            "Carefully watch this video and pay attention to every detail. "
            "Based on your observations, select the best option that accurately addresses the question.\n"
            "\nThese are the frames of a video and the corresponding audio. "
            "Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + q_block
        )
    if n in {"daily-omni", "dailyomni"}:
        return (
            "Listen and watch the video carefully. "
            "Select the best answer to the following multiple-choice question. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + q_block + "\nThe best answer is:"
        )
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        + q_block + "\nThe best answer is:"
    )


# ── Model loading ────────────────────────────────────────────────────────────

def load_model():
    global MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB
    print(f"Loading model from {MODEL_PATH} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB = _capture_vram()
    print(f"Model loaded. VRAM: {MODEL_LOADED_ALLOC_GB:.1f} GB alloc, {MODEL_LOADED_RESERVED_GB:.1f} GB reserved")
    return model, processor


# ── Inference ────────────────────────────────────────────────────────────────

_DROP_KEYS = frozenset({"images", "return_tensors", "text"})


def _apply_token_level_prune(inputs, text_query, keep_ratio, alpha, tau):
    """Apply ReDiPrune at the token (pixel patch) level.

    Intercepts pixel_values_videos from processor output (shape: (N_tokens, patch_dim)),
    selects K = round(N_tokens * keep_ratio) tokens via ReDiPrune, and replaces
    unselected tokens with zero vectors so sequence length is preserved (avoids
    breaking position encodings). Updates video_grid_thw to reflect kept temporal
    frames count.

    Returns (inputs, orig_ntokens, used_ntokens).
    """
    if "pixel_values_videos" not in inputs:
        return inputs, 0, 0

    pvv = inputs["pixel_values_videos"]
    if pvv is None or pvv.dim() < 2 or pvv.shape[0] <= 1:
        orig = pvv.shape[0] if pvv is not None else 0
        return inputs, orig, orig

    orig_ntokens = pvv.shape[0]
    feat = pvv.float()
    if feat.dim() > 2:
        feat = feat.reshape(feat.shape[0], -1)

    sel_idx = rediprune_select(feat, text_query, keep_ratio, alpha, tau)
    used_ntokens = sel_idx.shape[0]

    if used_ntokens < orig_ntokens:
        # Zero out unselected token positions to preserve sequence shape
        mask = torch.zeros(orig_ntokens, dtype=torch.bool, device=pvv.device)
        mask[sel_idx] = True
        zeroed = pvv.clone()
        zeroed[~mask] = 0.0
        inputs["pixel_values_videos"] = zeroed

        # Update grid_thw: scale the temporal dimension (T) proportionally
        if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
            grid = inputs["video_grid_thw"]
            if grid.dim() == 2 and grid.shape[1] >= 1:
                ratio = used_ntokens / orig_ntokens
                new_grid = grid.clone()
                new_grid[:, 0] = torch.clamp((grid[:, 0].float() * ratio).long(), min=1)
                inputs["video_grid_thw"] = new_grid

    return inputs, orig_ntokens, used_ntokens


def run_inference(model, processor, video_path, dataset, question, choices,
                  fps, max_pixels, max_new_tokens, use_audio, keep_ratio, alpha, tau,
                  prune_mode, max_frames=None, temperature=DEFAULT_TEMPERATURE):
    prompt = build_prompt(dataset, question, choices)
    system_text = SYSTEM_PROMPT + " " + SYSTEM_MCQ_SUFFIX

    video_element = {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels}
    if max_frames is not None:
        video_element["max_frames"] = max_frames

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [video_element, {"type": "text", "text": prompt}]},
    ]

    text_query = get_text_query(model, processor, question)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio)
    except Exception:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        use_audio = False

    if not videos or videos[0] is None or getattr(videos[0], "shape", None) is None:
        raise ValueError("Decoded 0 video frames")

    orig_nframes = videos[0].shape[0]
    used_nframes = orig_nframes

    # Frame-level pruning
    if prune_mode == "frame":
        pruned_videos = []
        for vid in videos:
            if vid is not None and vid.shape[0] > 1:
                features = vid.float().mean(dim=(-2, -1))  # (N, C)
                sel = rediprune_select(features, text_query, keep_ratio, alpha, tau)
                pruned_videos.append(vid[sel])
            else:
                pruned_videos.append(vid)
        videos = pruned_videos
        used_nframes = videos[0].shape[0] if videos[0] is not None else 0

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio,
    )
    dev = next(model.parameters()).device
    inputs = inputs.to(dev)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)

    # Token-level pruning — applied after processor, before model forward
    orig_ntokens = used_ntokens = 0
    if prune_mode == "token":
        inputs, orig_ntokens, used_ntokens = _apply_token_level_prune(
            inputs, text_query, keep_ratio, alpha, tau
        )
        used_nframes = used_ntokens  # report token count as "used frames" for token mode

    tokenizer = processor.tokenizer
    gen_kw = {
        "use_audio_in_video": use_audio,
        "return_audio": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    do_sample = temperature > 0
    if hasattr(model, "thinker"):
        gen_kw["thinker_max_new_tokens"] = max_new_tokens
        gen_kw["thinker_do_sample"] = do_sample
        if do_sample:
            gen_kw["thinker_temperature"] = temperature
    else:
        gen_kw["max_new_tokens"] = max_new_tokens
        gen_kw["do_sample"] = do_sample
        if do_sample:
            gen_kw["temperature"] = temperature

    gen_in = {k: v for k, v in inputs.items() if k not in _DROP_KEYS}

    # Measure prefill
    prefill_kw = dict(gen_kw)
    if "thinker_max_new_tokens" in prefill_kw:
        prefill_kw["thinker_max_new_tokens"] = 1
    else:
        prefill_kw["max_new_tokens"] = 1
    with torch.no_grad():
        prefill_ms, _ = cuda_time_ms(lambda: model.generate(**gen_in, **prefill_kw))

    with torch.no_grad():
        raw_out = model.generate(**gen_in, **gen_kw)

    seq_ids = raw_out.sequences if hasattr(raw_out, "sequences") else raw_out
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, orig_nframes, used_nframes, round(prefill_ms, 2)


# ── Run one mode ─────────────────────────────────────────────────────────────

def run_mode(prune_mode, args, model, processor, meta, out_base):
    out_dir = os.path.join(out_base, prune_mode)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    vram_path = os.path.join(out_dir, "vram_log.jsonl")

    print(f"\n{'='*60}")
    print(f"prune_mode={prune_mode} | keep_ratio={FIXED_KEEP_RATIO} | alpha={FIXED_ALPHA} | tau={FIXED_TAU}")
    if prune_mode == "frame":
        print("  Frame-level: selects whole frames via global avg-pool features (N_frames, C)")
    else:
        print("  Token-level: selects patch tokens from pixel_values_videos (N_patches, D)")
        print("  Unselected patches zeroed out to preserve sequence length")
    print(f"Output: {results_path}")
    print(f"{'='*60}")

    correct = total = 0
    results = []

    with open(results_path, "w") as out_f, open(vram_path, "w") as vram_f:
        for entry in meta:
            video_path = resolve_video_path(entry["file"], args.videos)
            if video_path is None:
                continue
            use_audio = (not args.no_audio) and check_video_has_audio(video_path)
            ds_name = _canonicalize(entry.get("dataset", ""))

            for q in entry.get("questions", []):
                question = q["question"]
                choices = q["choices"]
                answer = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset = entry.get("dataset", "")
                max_frames = DEFAULT_MAX_FRAMES_VIDEOMME if ds_name in {"video-mme", "videomme"} else DEFAULT_MAX_FRAMES_OTHER

                before_alloc, before_res = _capture_vram()
                torch.cuda.reset_peak_memory_stats()
                try:
                    pred, reasoning, orig_nf, used_nf, prefill_ms = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        args.fps, args.max_pixels, DEFAULT_MAX_NEW_TOKENS, use_audio,
                        FIXED_KEEP_RATIO, FIXED_ALPHA, FIXED_TAU, prune_mode,
                        max_frames=max_frames, temperature=DEFAULT_TEMPERATURE,
                    )
                    status = "ok"
                except Exception as e:
                    import traceback
                    print(f"  ERROR: {e!r}")
                    traceback.print_exc()
                    pred, reasoning, orig_nf, used_nf, prefill_ms = "ERROR", str(e), 0, 0, 0.0
                    status = "error"

                torch.cuda.empty_cache()
                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                peak_res = torch.cuda.max_memory_reserved() / 1024**3
                after_alloc, after_res = _capture_vram()

                rec = {
                    "dataset": dataset, "task_type": task_type,
                    "question": question, "answer": answer,
                    "prediction": pred, "correct": is_correct,
                    "orig_nframes": orig_nf, "used_nframes": used_nf,
                    "prefill_ms": prefill_ms,
                    "method": "rediprune_mode",
                    "config": {"alpha": FIXED_ALPHA, "keep_ratio": FIXED_KEEP_RATIO, "tau": FIXED_TAU, "prune_mode": prune_mode},
                    "ablation_param": "prune_mode", "ablation_value": prune_mode,
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                results.append(rec)

                vram_rec = {
                    "status": status, "prune_mode": prune_mode,
                    "orig_frames": orig_nf, "used_frames": used_nf,
                    "before_alloc_gb": round(before_alloc, 2), "before_res_gb": round(before_res, 2),
                    "peak_alloc_gb": round(peak_alloc, 2), "peak_res_gb": round(peak_res, 2),
                    "after_alloc_gb": round(after_alloc, 2), "after_res_gb": round(after_res, 2),
                    "model_loaded_alloc_gb": round(MODEL_LOADED_ALLOC_GB or 0.0, 2),
                }
                vram_f.write(json.dumps(vram_rec) + "\n")
                vram_f.flush()

                sym = "✓" if is_correct else "✗"
                print(f"  [{sym}] {dataset}/{task_type} pred={pred} ans={answer} units={orig_nf}->{used_nf} prefill={prefill_ms:.0f}ms")

    acc = correct / total if total else 0.0
    print(f"  mode={prune_mode}: {correct}/{total} = {acc:.2%}")
    return {"prune_mode": prune_mode, "accuracy": round(acc, 4), "correct": correct, "total": total}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ReDiPrune pruning mode ablation (frame vs token)")
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--videos", default="/data/videos")
    parser.add_argument("--output_base", default=None,
                        help="Base output dir. Defaults to ablation_outputs/rediprune_mode/")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--value", type=str, default=None, choices=["frame", "token"],
                        help="Run a single pruning mode (for parallel GPU execution). "
                             "If omitted, runs both: frame, token")
    args = parser.parse_args()

    out_base = args.output_base or os.path.join(
        os.path.dirname(_REPO_ROOT), "ablation_outputs", "rediprune_mode"
    )
    os.makedirs(out_base, exist_ok=True)

    global MODEL_PATH
    if not os.path.exists(MODEL_PATH) and os.path.exists(FALLBACK_MODEL_PATH):
        MODEL_PATH = FALLBACK_MODEL_PATH

    meta_raw = json.loads(Path(args.metadata).read_text())
    meta = [e for e in meta_raw if e.get("questions")]
    print(f"Loaded {len(meta)} runnable entries from {args.metadata}")

    if not meta:
        print("Nothing to run.")
        return

    model, processor = load_model()

    modes = [args.value] if args.value is not None else MODE_VALUES
    summary_results = []

    for mode in modes:
        res = run_mode(mode, args, model, processor, meta, out_base)
        summary_results.append(res)

    summary = {
        "ablation": "rediprune_mode",
        "fixed": {"keep_ratio": FIXED_KEEP_RATIO, "alpha": FIXED_ALPHA, "tau": FIXED_TAU},
        "results": summary_results,
    }
    summary_path = os.path.join(out_base, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSweep summary written to {summary_path}")

    print("\n=== Pruning Mode Comparison ===")
    for r in summary_results:
        print(f"  mode={r['prune_mode']}: accuracy={r['accuracy']:.4f} ({r['correct']}/{r['total']})")

    if len(summary_results) == 2:
        delta = summary_results[1]["accuracy"] - summary_results[0]["accuracy"]
        print(f"  delta (token - frame): {delta:+.4f}")


if __name__ == "__main__":
    main()
