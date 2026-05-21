"""
rediprune_random_query.py
=========================
Ablation: does the ACTUAL question text query matter, or does any direction work?

MATH:
  Normal ReDiPrune:
    rel_i = cos_sim(v_i, embed(question_text))     <- uses real question embedding

  Random query ReDiPrune:
    q_rand = normalize(randn(D))                   <- random unit vector
    rel_i = cos_sim(v_i, q_rand)                   <- measures alignment with noise

  Zero alpha (DivPrune):
    score_i = min_{j∈S} (1 - cos_sim(v_i, v_j))  <- no relevance term at all

Three conditions compared:
  1. real_query    — standard ReDiPrune: embed(question) used as query (alpha=0.5)
  2. random_query  — embed replaced with normalized Gaussian noise (alpha=0.5)
  3. zero_alpha    — alpha=0 (pure DivPrune), no query used at all (control)

For random_query: seed is set per-question via `torch.manual_seed(hash(question) % 2**31)`
to ensure reproducibility across runs while varying per question (simulating a realistic
random baseline rather than a fixed adversarial direction).

If real_query >> random_query ≈ zero_alpha:
  → The actual question text is meaningful; text-conditioning genuinely helps.
If real_query ≈ random_query >> zero_alpha:
  → Any relevance direction helps, but not the specific question text.
If all three ≈ equal:
  → Text-relevance adds no value beyond pure diversity.

Fixed: keep_ratio=0.5, tau=0.1, prune_mode=frame

Usage:
  # Run all three conditions:
  python rediprune_random_query.py --metadata metadata.json --videos /data/videos

  # Run a single condition (for parallel GPU execution):
  python rediprune_random_query.py --metadata metadata.json --videos /data/videos --value real_query
  python rediprune_random_query.py --metadata metadata.json --videos /data/videos --value random_query
  python rediprune_random_query.py --metadata metadata.json --videos /data/videos --value zero_alpha
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
FIXED_TAU = 0.1
FIXED_ALPHA = 0.5          # used for real_query and random_query
ZERO_ALPHA = 0.0           # used for zero_alpha (pure DivPrune)
FIXED_PRUNE_MODE = "frame"
CONDITION_VALUES = ["real_query", "random_query", "zero_alpha"]

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
    When alpha=0, degenerates to pure DivPrune (diversity only).
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


def get_random_query(model, question_text, dim=None):
    """Return a random unit vector as a fake query.

    Seed is derived from the question text hash so that the same question always
    gets the same random vector (reproducible across re-runs), while different
    questions get different vectors (not a single fixed adversarial direction).
    """
    dev = next(model.parameters()).device
    if dim is None:
        # Infer embedding dim from embed_tokens weight
        embed = model.thinker.model.embed_tokens if hasattr(model, "thinker") else model.model.embed_tokens
        dim = embed.weight.shape[1]

    seed = hash(question_text) % (2 ** 31)
    gen = torch.Generator()
    gen.manual_seed(seed)
    rand_vec = torch.randn(dim, generator=gen).to(dev)
    return F.normalize(rand_vec, p=2, dim=0)


def get_dummy_query(model):
    """Return a zero vector (for zero_alpha condition, though alpha=0 makes it irrelevant)."""
    dev = next(model.parameters()).device
    embed = model.thinker.model.embed_tokens if hasattr(model, "thinker") else model.model.embed_tokens
    dim = embed.weight.shape[1]
    return torch.zeros(dim, device=dev)


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


def run_inference(model, processor, video_path, dataset, question, choices,
                  fps, max_pixels, max_new_tokens, use_audio, condition,
                  max_frames=None, temperature=DEFAULT_TEMPERATURE):
    """
    condition: one of "real_query", "random_query", "zero_alpha"
    """
    prompt = build_prompt(dataset, question, choices)
    system_text = SYSTEM_PROMPT + " " + SYSTEM_MCQ_SUFFIX

    video_element = {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels}
    if max_frames is not None:
        video_element["max_frames"] = max_frames

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [video_element, {"type": "text", "text": prompt}]},
    ]

    # Select query and alpha based on condition
    if condition == "real_query":
        text_query = get_text_query(model, processor, question)
        alpha = FIXED_ALPHA
    elif condition == "random_query":
        text_query = get_random_query(model, question)
        alpha = FIXED_ALPHA
    elif condition == "zero_alpha":
        # alpha=0 means text_query is irrelevant, but we still need a valid tensor
        text_query = get_dummy_query(model)
        alpha = ZERO_ALPHA
    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio)
    except Exception:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        use_audio = False

    if not videos or videos[0] is None or getattr(videos[0], "shape", None) is None:
        raise ValueError("Decoded 0 video frames")

    orig_nframes = videos[0].shape[0]

    # Frame-level pruning
    pruned_videos = []
    for vid in videos:
        if vid is not None and vid.shape[0] > 1:
            features = vid.float().mean(dim=(-2, -1))  # (N, C)
            sel = rediprune_select(features, text_query, FIXED_KEEP_RATIO, alpha, FIXED_TAU)
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
    return letter, decoded, orig_nframes, used_nframes, round(prefill_ms, 2), alpha


# ── Run one condition ─────────────────────────────────────────────────────────

def run_condition(condition, args, model, processor, meta, out_base):
    out_dir = os.path.join(out_base, condition)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    vram_path = os.path.join(out_dir, "vram_log.jsonl")

    alpha_used = ZERO_ALPHA if condition == "zero_alpha" else FIXED_ALPHA

    print(f"\n{'='*60}")
    print(f"condition={condition} | keep_ratio={FIXED_KEEP_RATIO} | tau={FIXED_TAU} | alpha={alpha_used}")
    if condition == "real_query":
        print("  Query: actual question text embedded via embed_tokens")
    elif condition == "random_query":
        print("  Query: random unit vector, seeded from hash(question) for reproducibility")
    else:
        print("  Query: not used (alpha=0, pure DivPrune diversity selection)")
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
                    pred, reasoning, orig_nf, used_nf, prefill_ms, eff_alpha = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        args.fps, args.max_pixels, DEFAULT_MAX_NEW_TOKENS, use_audio,
                        condition, max_frames=max_frames, temperature=DEFAULT_TEMPERATURE,
                    )
                    status = "ok"
                except Exception as e:
                    import traceback
                    print(f"  ERROR: {e!r}")
                    traceback.print_exc()
                    pred, reasoning, orig_nf, used_nf, prefill_ms, eff_alpha = "ERROR", str(e), 0, 0, 0.0, alpha_used
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
                    "method": "rediprune_random_query",
                    "config": {
                        "alpha": eff_alpha,
                        "keep_ratio": FIXED_KEEP_RATIO,
                        "tau": FIXED_TAU,
                        "prune_mode": FIXED_PRUNE_MODE,
                        "condition": condition,
                    },
                    "ablation_param": "query_condition", "ablation_value": condition,
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                results.append(rec)

                vram_rec = {
                    "status": status, "condition": condition,
                    "orig_frames": orig_nf, "used_frames": used_nf,
                    "before_alloc_gb": round(before_alloc, 2), "before_res_gb": round(before_res, 2),
                    "peak_alloc_gb": round(peak_alloc, 2), "peak_res_gb": round(peak_res, 2),
                    "after_alloc_gb": round(after_alloc, 2), "after_res_gb": round(after_res, 2),
                    "model_loaded_alloc_gb": round(MODEL_LOADED_ALLOC_GB or 0.0, 2),
                }
                vram_f.write(json.dumps(vram_rec) + "\n")
                vram_f.flush()

                sym = "✓" if is_correct else "✗"
                print(f"  [{sym}] {dataset}/{task_type} pred={pred} ans={answer} frames={orig_nf}->{used_nf} prefill={prefill_ms:.0f}ms")

    acc = correct / total if total else 0.0
    print(f"  condition={condition}: {correct}/{total} = {acc:.2%}")
    return {"condition": condition, "accuracy": round(acc, 4), "correct": correct, "total": total}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ReDiPrune query condition ablation (real vs random vs no query)")
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--videos", default="/data/videos")
    parser.add_argument("--output_base", default=None,
                        help="Base output dir. Defaults to ablation_outputs/rediprune_random_query/")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--value", type=str, default=None,
                        choices=CONDITION_VALUES,
                        help="Run a single condition (for parallel GPU execution). "
                             "If omitted, runs all: " + str(CONDITION_VALUES))
    args = parser.parse_args()

    out_base = args.output_base or os.path.join(
        os.path.dirname(_REPO_ROOT), "ablation_outputs", "rediprune_random_query"
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

    conditions = [args.value] if args.value is not None else CONDITION_VALUES
    summary_results = []

    for condition in conditions:
        res = run_condition(condition, args, model, processor, meta, out_base)
        summary_results.append(res)

    summary = {
        "ablation": "rediprune_random_query",
        "fixed": {"keep_ratio": FIXED_KEEP_RATIO, "tau": FIXED_TAU, "prune_mode": FIXED_PRUNE_MODE},
        "notes": {
            "real_query": f"alpha={FIXED_ALPHA}, query=embed(question_text)",
            "random_query": f"alpha={FIXED_ALPHA}, query=random_unit_vec seeded by hash(question)",
            "zero_alpha": f"alpha={ZERO_ALPHA}, pure DivPrune (no text relevance)",
        },
        "results": summary_results,
    }
    summary_path = os.path.join(out_base, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSweep summary written to {summary_path}")

    print("\n=== Query Condition Comparison ===")
    for r in summary_results:
        print(f"  condition={r['condition']}: accuracy={r['accuracy']:.4f} ({r['correct']}/{r['total']})")

    # Compute interpretation gaps
    if len(summary_results) == 3:
        acc_map = {r["condition"]: r["accuracy"] for r in summary_results}
        if all(k in acc_map for k in ("real_query", "random_query", "zero_alpha")):
            d_real_vs_rand = acc_map["real_query"] - acc_map["random_query"]
            d_real_vs_zero = acc_map["real_query"] - acc_map["zero_alpha"]
            d_rand_vs_zero = acc_map["random_query"] - acc_map["zero_alpha"]
            print(f"\n  real_query - random_query = {d_real_vs_rand:+.4f}  (text specificity benefit)")
            print(f"  real_query - zero_alpha   = {d_real_vs_zero:+.4f}  (relevance vs diversity only)")
            print(f"  random_query - zero_alpha = {d_rand_vs_zero:+.4f}  (any relevance direction benefit)")


if __name__ == "__main__":
    main()
