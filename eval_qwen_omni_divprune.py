"""
eval_qwen_omni_divprune.py
Runs Qwen2.5-Omni-7B + DivPrune diversity-based visual token pruning.

DivPrune (CVPR 2025) selects the most diverse visual tokens using a greedy
farthest-point-sampling algorithm on cosine distance. This is training-free.

Two modes of operation:
  frame-level (default):  Select diverse VIDEO FRAMES before they enter the model.
                          Simpler, no model surgery required. Each frame maps to
                          multiple visual tokens, so pruning frames prunes token groups.
  token-level:            Hook into the thinker to prune individual visual token
                          embeddings after the vision encoder. More faithful to the
                          original DivPrune paper but requires model internals access.

Usage:
    # Frame-level (recommended starting point):
    python eval_qwen_omni_divprune.py --metadata metadata.json --videos /workspace/videos \\
        --output /workspace/results_divprune.jsonl --subset_ratio 0.5 --prune_mode frame

    # Token-level (more aggressive, closer to paper):
    python eval_qwen_omni_divprune.py --metadata metadata.json --videos /workspace/videos \\
        --output /workspace/results_divprune.jsonl --subset_ratio 0.3 --prune_mode token
"""

import argparse
import json
import math
import os
import glob
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(_REPO_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from mcq_answer_parse import parse_answer

# ── Constants ────────────────────────────────────────────────────────────────

ENV_MODEL_PATH_KEY = "QWEN_OMNI_MODEL_PATH"
DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
FALLBACK_MODEL_PATH = "/workspace/model"
MODEL_PATH = os.environ.get(ENV_MODEL_PATH_KEY) or DEFAULT_MODEL_PATH

SYSTEM_PROMPT_DEFAULT = (
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


# ── Prompt builders ──────────────────────────────────────────────────────────

def _format_choice_lines(choices):
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def _build_user_prompt_video_mme(q, c):
    return _VIDEO_MME_OPTION_PROMPT + "\n" + q + "\n" + _format_choice_lines(c) + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_worldsense(q, c):
    parts = [_WORLD_SENSE_SYS, _WORLD_SENSE_FRAMES_AUDIO, q + "\n"]
    for op in c:
        parts.append(op + "\n")
    return "".join(parts)


def _build_user_prompt_daily_omni(q, c):
    head = ("Listen and watch the video carefully. "
            "Select the best answer to the following multiple-choice question. "
            "Respond with only the letter (A, B, C, or D) of the correct option.")
    return head + "\n" + q + "\n" + _format_choice_lines(c) + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_default(q, c):
    return ("Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + q + "\n" + _format_choice_lines(c) + "\n" + _VIDEO_MME_POST_PROMPT)


def _canonicalize(ds):
    return (ds or "").strip().lower().replace("_", "-").replace(" ", "-")


def build_user_prompt_for_dataset(dataset, question, choices):
    n = _canonicalize(dataset)
    if n in {"video-mme", "videomme"}:
        return _build_user_prompt_video_mme(question, choices)
    if n == "worldsense":
        return _build_user_prompt_worldsense(question, choices)
    if n in {"daily-omni", "dailyomni"}:
        return _build_user_prompt_daily_omni(question, choices)
    return _build_user_prompt_default(question, choices)


# ── Utilities ────────────────────────────────────────────────────────────────

def resolve_model_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def check_video_has_audio(path):
    try:
        import av
        c = av.open(path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False


def resolve_video_path(file_field, videos_dir):
    if os.path.exists(file_field):
        return file_field
    normalized = file_field.replace("\\", "/")
    filename = normalized.split("/")[-1]
    stem = filename.rsplit(".", 1)[0]
    candidate = os.path.join(videos_dir, filename)
    if os.path.exists(candidate):
        return candidate
    for ext in ("mp4", "mkv", "webm", "avi"):
        matches = glob.glob(os.path.join(videos_dir, "**", f"{stem}.{ext}"), recursive=True)
        if matches:
            return matches[0]
    return None


class Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
        self.log.write(f"\n{'='*60}\nRUN (DivPrune): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        self.log.flush()

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
    def __init__(self, log_file, terminal):
        self.log = log_file
        self.terminal = terminal

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


# ══════════════════════════════════════════════════════════════════════════════
# DivPrune: Diversity-based Visual Token/Frame Pruning
# ══════════════════════════════════════════════════════════════════════════════

def pairwise_cosine_distance(features: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine distance matrix: 1 - cosine_similarity.

    Args:
        features: (N, D) tensor of feature vectors

    Returns:
        (N, N) distance matrix
    """
    normed = F.normalize(features, dim=-1)
    sim = torch.mm(normed, normed.t())
    return 1.0 - sim


def divprune_select(features: torch.Tensor, keep_count: int) -> torch.Tensor:
    """Greedy farthest-point diversity selection (DivPrune algorithm).

    Iteratively selects tokens that maximize the minimum cosine distance to
    all previously selected tokens. This is the core DivPrune algorithm from
    the CVPR 2025 paper.

    Args:
        features: (N, D) tensor of feature vectors
        keep_count: number of tokens to select

    Returns:
        (keep_count,) tensor of selected indices
    """
    N = features.shape[0]
    if keep_count >= N:
        return torch.arange(N, device=features.device)

    dist_matrix = pairwise_cosine_distance(features)

    selected = torch.empty(keep_count, dtype=torch.long, device=features.device)

    # First token: pick the one whose nearest neighbor is farthest (most isolated)
    topk_vals = torch.topk(dist_matrix, k=2, dim=0, largest=False).values
    scores = topk_vals[1, :]  # second-smallest = distance to nearest neighbor
    selected[0] = torch.argmax(scores)

    for i in range(1, keep_count):
        # Distance from each candidate to the closest already-selected token
        sel_dists = dist_matrix[selected[:i], :]  # (i, N)
        min_dists = sel_dists.min(dim=0).values    # (N,)
        # Pick the candidate farthest from any selected token
        selected[i] = torch.argmax(min_dists)

    return selected


def divprune_select_frames(video_tensor: torch.Tensor, subset_ratio: float) -> torch.Tensor:
    """Apply DivPrune at the frame level.

    Args:
        video_tensor: (nframes, C, H, W) video frames tensor
        subset_ratio: fraction of frames to keep (0.0 - 1.0)

    Returns:
        (selected_nframes, C, H, W) subset of frames selected for diversity
    """
    nframes = video_tensor.shape[0]
    keep_count = max(1, int(round(subset_ratio * nframes)))

    if keep_count >= nframes:
        return video_tensor

    # Create frame-level features via global average pooling
    # video_tensor: (nframes, C, H, W) -> features: (nframes, C)
    features = video_tensor.float().mean(dim=(-2, -1))  # (nframes, C)

    selected_idx = divprune_select(features, keep_count)
    # Sort to maintain temporal order
    selected_idx = selected_idx.sort().values

    return video_tensor[selected_idx]


def divprune_select_tokens(token_embeddings: torch.Tensor, subset_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply DivPrune at the token level on visual token embeddings.

    Args:
        token_embeddings: (N, D) visual token embeddings
        subset_ratio: fraction of tokens to keep

    Returns:
        (selected_embeddings, selected_indices)
    """
    N = token_embeddings.shape[0]
    keep_count = max(1, int(round(subset_ratio * N)))

    if keep_count >= N:
        return token_embeddings, torch.arange(N, device=token_embeddings.device)

    selected_idx = divprune_select(token_embeddings, keep_count)
    selected_idx = selected_idx.sort().values
    return token_embeddings[selected_idx], selected_idx


# ── Token-level DivPrune: Hook into the thinker's vision encoding ───────────

_ORIG_ENCODE_VISION = None  # Will store original method


def _make_divprune_vision_hook(subset_ratio: float):
    """Create a hook that prunes visual token embeddings after the vision encoder.

    This patches the thinker's _get_vision_info or equivalent method to apply
    DivPrune on the vision encoder output before it gets merged into text embeddings.
    """
    def hook_fn(module, input, output):
        # output is typically (visual_features, grid_thw) or just visual_features
        if isinstance(output, tuple):
            features = output[0]
        else:
            features = output

        if features is None or features.dim() < 2:
            return output

        # Apply DivPrune to each batch item
        pruned_features = []
        for b in range(features.shape[0]):
            feat = features[b]  # (N_tokens, D)
            pruned, _ = divprune_select_tokens(feat, subset_ratio)
            pruned_features.append(pruned)

        # Stack if all same size, otherwise return first (batch=1 typical in eval)
        if all(p.shape[0] == pruned_features[0].shape[0] for p in pruned_features):
            result = torch.stack(pruned_features)
        else:
            result = pruned_features[0].unsqueeze(0)

        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result

    return hook_fn


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(dtype_name: str):
    dt = resolve_model_dtype(dtype_name)
    print(f"Loading model from {MODEL_PATH} (dtype={dtype_name}) ...")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dt,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor


# ── Inference ────────────────────────────────────────────────────────────────

_OMNI_GENERATE_BATCH_DROP = frozenset({"images", "return_tensors", "text"})


def _prepare_omni_inputs(model, inputs):
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)
    return inputs


def _generation_output_token_ids(gen_out):
    return gen_out.sequences if hasattr(gen_out, "sequences") else gen_out


def run_inference(model, processor, video_path, dataset, question, choices,
                  fps, max_pixels, max_new_tokens, use_audio, subset_ratio, prune_mode):
    prompt = build_user_prompt_for_dataset(dataset, question, choices)
    system_text = SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    effective_use_audio = use_audio
    try:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=effective_use_audio)
    except Exception:
        if not effective_use_audio:
            raise
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        effective_use_audio = False

    if not videos or videos[0] is None or getattr(videos[0], "shape", None) is None:
        raise ValueError("Decoded 0 video frames")

    orig_nframes = videos[0].shape[0]

    # ── DivPrune: Frame-level pruning ──
    if prune_mode == "frame" and videos and videos[0] is not None:
        videos_pruned = []
        for vid in videos:
            if vid is not None and vid.shape[0] > 1:
                pruned = divprune_select_frames(vid, subset_ratio)
                videos_pruned.append(pruned)
            else:
                videos_pruned.append(vid)
        videos = videos_pruned
        pruned_nframes = videos[0].shape[0] if videos[0] is not None else 0
    else:
        pruned_nframes = orig_nframes

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=effective_use_audio,
    )
    inputs = _prepare_omni_inputs(model, inputs)

    # ── DivPrune: Token-level pruning via processor output manipulation ──
    # For token-level mode, we prune the pixel_values_videos after processor
    # encodes them but before the model forward. This is an approximation;
    # true token-level pruning would require hooking inside the vision encoder.
    if prune_mode == "token" and "pixel_values_videos" in inputs:
        pvv = inputs["pixel_values_videos"]  # (total_patches, C*patch_H*patch_W) or similar
        if pvv is not None and pvv.dim() >= 2 and pvv.shape[0] > 1:
            keep_count = max(1, int(round(subset_ratio * pvv.shape[0])))
            if keep_count < pvv.shape[0]:
                # Use the pixel features themselves for diversity selection
                feat = pvv.float()
                if feat.dim() > 2:
                    feat = feat.view(feat.shape[0], -1)
                selected_idx = divprune_select(feat, keep_count)
                selected_idx = selected_idx.sort().values
                inputs["pixel_values_videos"] = pvv[selected_idx]
                # Update grid_thw if present (reduce temporal dimension)
                if "video_grid_thw" in inputs:
                    grid = inputs["video_grid_thw"]  # (num_videos, 3) -> [T, H_grid, W_grid]
                    if grid is not None and grid.dim() == 2:
                        # Approximate: scale down temporal dim proportionally
                        ratio = keep_count / pvv.shape[0]
                        new_grid = grid.clone()
                        new_grid[:, 0] = torch.clamp((grid[:, 0].float() * ratio).long(), min=1)
                        inputs["video_grid_thw"] = new_grid

    tokenizer = processor.tokenizer
    gen_kw = {
        "use_audio_in_video": effective_use_audio,
        "return_audio": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if hasattr(model, "thinker"):
        gen_kw["thinker_max_new_tokens"] = max_new_tokens
        gen_kw["thinker_do_sample"] = False
    else:
        gen_kw["max_new_tokens"] = max_new_tokens
        gen_kw["do_sample"] = False

    gen_in = {k: v for k, v in inputs.items() if k not in _OMNI_GENERATE_BATCH_DROP}
    with torch.no_grad():
        raw_out = model.generate(**gen_in, **gen_kw)

    seq_ids = _generation_output_token_ids(raw_out)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, orig_nframes, pruned_nframes


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni + DivPrune")
    parser.add_argument("--model", default=None)
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--videos", default="/workspace/videos")
    parser.add_argument("--output", default="/workspace/results_divprune.jsonl")
    parser.add_argument("--log", default="/workspace/eval_divprune.log")
    parser.add_argument("--errors_log", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=360*420)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--vram_log", default="/workspace/vram_log_divprune.jsonl")
    parser.add_argument("--stderr_log", default=None)
    # DivPrune-specific
    parser.add_argument("--subset_ratio", type=float, default=0.5,
                        help="Fraction of visual tokens/frames to KEEP (0.0-1.0). "
                             "Paper default for LLaVA is 0.098 but Qwen2.5-Omni uses fewer "
                             "frames to begin with, so 0.3-0.5 is a better starting point.")
    parser.add_argument("--prune_mode", default="frame", choices=["frame", "token"],
                        help="'frame': prune video frames before model (simpler, recommended). "
                             "'token': prune visual token patches after processor (experimental).")
    args = parser.parse_args()

    global MODEL_PATH
    if args.model:
        MODEL_PATH = args.model
    elif not os.path.exists(MODEL_PATH) and os.path.exists(FALLBACK_MODEL_PATH):
        MODEL_PATH = FALLBACK_MODEL_PATH

    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found. Use --no_audio or install ffmpeg.\n")

    errors_log_path = args.errors_log or os.path.join(os.path.dirname(args.log) or ".", "errors.log")
    for path in [args.log, args.output, args.vram_log, errors_log_path]:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if args.stderr_log:
        os.makedirs(os.path.dirname(args.stderr_log) or ".", exist_ok=True)
        _stderr_f = open(args.stderr_log, "a", encoding="utf-8")
        sys.stderr = StderrTee(_stderr_f, sys.__stderr__)

    tee = Tee(args.log)
    sys.stdout = tee

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} entries")

    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    runnable = [e for e in meta if e.get("questions")]
    skipped_no_qa = len(meta) - len(runnable)
    if skipped_no_qa:
        print(f"Skipping {skipped_no_qa} entries with no Q&A")
    print(f"Running {len(runnable)} entries with DivPrune ({args.prune_mode}, ratio={args.subset_ratio})\n")

    if not runnable:
        print("Nothing to run.")
        return

    model, processor = load_model(args.dtype)

    correct = total = skipped_no_video = 0
    total_orig_frames = total_pruned_frames = 0
    results = []

    with open(args.output, "w") as out_f, open(args.vram_log, "w") as vram_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video not found")
                skipped_no_video += 1
                continue

            use_audio = (not args.no_audio) and check_video_has_audio(video_path)

            for q in entry["questions"]:
                question = q["question"]
                choices = q["choices"]
                answer = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset = entry.get("dataset", "")

                try:
                    torch.cuda.reset_peak_memory_stats()
                    pred, reasoning, orig_nf, pruned_nf = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        args.fps, args.max_pixels, args.max_new_tokens, use_audio,
                        args.subset_ratio, args.prune_mode,
                    )
                    total_orig_frames += orig_nf
                    total_pruned_frames += pruned_nf
                    vram_entry = {
                        "entry": entry_label, "task_type": task_type,
                        "duration_s": entry.get("duration_s"),
                        "orig_frames": orig_nf, "pruned_frames": pruned_nf,
                        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                        "after_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                        "after_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                except Exception as e:
                    import traceback
                    print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                    with open(errors_log_path, "a") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{traceback.format_exc()}\n")
                    pred, reasoning = "ERROR", str(e)
                    orig_nf = pruned_nf = 0

                torch.cuda.empty_cache()
                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "model_variant": "qwen2.5-omni",
                    "dataset": entry.get("dataset"),
                    "task_type": task_type,
                    "duration_s": entry.get("duration_s"),
                    "question": question, "choices": choices,
                    "answer": answer, "prediction": pred,
                    "correct": is_correct, "reasoning": reasoning,
                    "method": f"divprune-{args.prune_mode}",
                    "subset_ratio": args.subset_ratio,
                    "orig_frames": orig_nf, "pruned_frames": pruned_nf,
                }
                out_f.write(json.dumps(result) + "\n")
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer} frames={orig_nf}->{pruned_nf}")

    acc = correct / total if total else 0
    frame_ratio = total_pruned_frames / total_orig_frames if total_orig_frames else 0
    print(f"\n{'='*50}")
    print(f"Model: qwen2.5-omni + DivPrune ({args.prune_mode})")
    print(f"Subset ratio: {args.subset_ratio} | Effective frame ratio: {frame_ratio:.2%}")
    print(f"Total frames: {total_orig_frames} -> {total_pruned_frames}")
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Results: {args.output}")

    datasets = {}
    for r in results:
        d = r["dataset"]
        datasets.setdefault(d, {"correct": 0, "total": 0})
        datasets[d]["total"] += 1
        if r["correct"]:
            datasets[d]["correct"] += 1
    print("\nPer-dataset:")
    for ds, s in sorted(datasets.items()):
        print(f"  {ds:<20} {s['correct']}/{s['total']} = {s['correct']/s['total']:.2%}")


if __name__ == "__main__":
    main()
