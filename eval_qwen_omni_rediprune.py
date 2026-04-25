"""
eval_qwen_omni_rediprune.py
Runs Qwen2.5-Omni-7B + ReDiPrune text-conditioned diversity-based visual token pruning.

ReDiPrune combines text-relevance scoring with diversity-based selection:
  score = min_dist + alpha * relevance
where relevance = cosine similarity between visual tokens and a text query
derived from the question. This is training-free.

Two modes of operation:
  frame-level (default):  Select diverse+relevant VIDEO FRAMES before they enter the model.
                          Simpler, no model surgery required.
  token-level:            Prune individual visual token patches after processor
                          but before model forward. More aggressive.

Usage:
    # Frame-level (recommended):
    python eval_qwen_omni_rediprune.py --metadata metadata.json --videos /workspace/videos \\
        --output /workspace/results_rediprune.jsonl --subset_ratio 0.5 --prune_mode frame

    # Token-level:
    python eval_qwen_omni_rediprune.py --metadata metadata.json --videos /workspace/videos \\
        --output /workspace/results_rediprune.jsonl --subset_ratio 0.3 --prune_mode token

    # Tune relevance weight:
    python eval_qwen_omni_rediprune.py ... --alpha 0.8 --tau 0.1
"""

import argparse
import json
import math
import os
import glob
import random
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


def cuda_time_ms(fn):
    """Run fn() with CUDA event timing. Returns (elapsed_ms, result)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out

# ── Constants ────────────────────────────────────────────────────────────────

ENV_MODEL_PATH_KEY = "QWEN_OMNI_MODEL_PATH"
DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
FALLBACK_MODEL_PATH = "/workspace/model"
MODEL_PATH = os.environ.get(ENV_MODEL_PATH_KEY) or DEFAULT_MODEL_PATH
MODEL_LOADED_ALLOC_GB: float | None = None
MODEL_LOADED_RESERVED_GB: float | None = None

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


def set_run_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_current_vram_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def _record_model_loaded_vram() -> None:
    global MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB
    MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB = _capture_current_vram_gb()
    print(
        f"Model loaded. VRAM: {MODEL_LOADED_ALLOC_GB:.1f} GB allocated, "
        f"{MODEL_LOADED_RESERVED_GB:.1f} GB reserved"
    )


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
        self.log.write(f"\n{'='*60}\nRUN (ReDiPrune): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
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
# ReDiPrune: Text-Conditioned Diversity-based Visual Token/Frame Pruning
# Ported from ReDiPrune-main/LLaVA-NeXT/llavavid/model/llava_arch.py
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def rediprune_select_text_cond(
    visual_tokens: torch.Tensor,
    text_query: torch.Tensor,
    keep_ratio: float,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """Select visual tokens that are diverse AND relevant to a text query.

    Scoring: score = min_dist + alpha * relevance
        min_dist  - minimum cosine distance to the current selection (diversity)
        relevance - cosine similarity with the text query

    Args:
        visual_tokens: (P, Cv) visual embeddings
        text_query: (Dq,) or (1, Dq) query embedding
        keep_ratio: fraction of tokens to retain (0 < keep_ratio <= 1)
        alpha: weight of the relevance term (0 = pure diversity like DivPrune)
        tau: relevance threshold; tokens with relevance < tau are filtered

    Returns:
        1-D tensor of selected indices, sorted to preserve original order
    """
    P, Cv = visual_tokens.shape
    K = max(1, int(round(P * keep_ratio)))
    if K >= P:
        return torch.arange(P, device=visual_tokens.device)

    # Ensure both tensors are on the same device and in float32
    dev = visual_tokens.device
    visual_tokens = visual_tokens.float()
    text_query = text_query.float().to(dev)

    Vn = F.normalize(visual_tokens, p=2, dim=1)

    # Match text query dimensionality to Cv
    tq = text_query.unsqueeze(0) if text_query.dim() == 1 else text_query
    Dq = tq.shape[1]
    if Dq != Cv:
        if Dq > Cv:
            if Dq % Cv == 0:
                tq = tq.view(1, Cv, Dq // Cv).mean(dim=2)
            else:
                tq = F.adaptive_avg_pool1d(tq.unsqueeze(1), Cv).squeeze(1)
        else:
            rep = (Cv + Dq - 1) // Dq
            tq = tq.repeat(1, rep)[:, :Cv]
    Tn = F.normalize(tq, p=2, dim=1)

    # Relevance scores: cosine similarity with text query
    rel = (Vn @ Tn.t()).squeeze(1)  # (P,)

    # Candidate pre-selection via relevance threshold
    if tau > 0:
        if K < P // 2:
            _, cand = torch.topk(rel, k=min(P, max(K * 3, K + 1)))
        else:
            mask = (rel >= tau)
            if mask.sum() < K:
                _, cand = torch.topk(rel, K)
            else:
                cand = torch.where(mask)[0]
    else:
        cand = torch.arange(P, device=visual_tokens.device)

    # Pairwise cosine distance among candidates
    div_mat = 1.0 - (Vn[cand] @ Vn[cand].t())
    rel_c = rel[cand]

    # Greedy selection: first token = most relevant
    Nc = cand.shape[0]
    sel = torch.empty(K, dtype=torch.long, device=visual_tokens.device)
    sel[0] = torch.argmax(rel_c)

    for i in range(1, K):
        min_dist = div_mat[sel[:i]].min(dim=0).values  # (Nc,)
        score = min_dist + alpha * rel_c
        mask = torch.ones(Nc, dtype=torch.bool, device=visual_tokens.device)
        mask[sel[:i]] = False
        score[~mask] = -float('inf')
        sel[i] = torch.argmax(score)

    out_idx = cand[sel]
    out_idx, _ = torch.sort(out_idx)
    return out_idx


def rediprune_select_frames(
    video_tensor: torch.Tensor,
    text_query: torch.Tensor,
    keep_ratio: float,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """Apply ReDiPrune at the frame level.

    Args:
        video_tensor: (nframes, C, H, W) video frames tensor
        text_query: (D,) text query embedding
        keep_ratio: fraction of frames to keep (0.0 - 1.0)
        alpha: relevance weight
        tau: relevance threshold

    Returns:
        (selected_nframes, C, H, W) subset of frames
    """
    nframes = video_tensor.shape[0]
    if max(1, int(round(keep_ratio * nframes))) >= nframes:
        return video_tensor

    # Frame-level features via global average pooling: (nframes, C, H, W) -> (nframes, C)
    features = video_tensor.float().mean(dim=(-2, -1))

    selected_idx = rediprune_select_text_cond(features, text_query, keep_ratio, alpha, tau)
    return video_tensor[selected_idx]


def _get_text_query_embedding(model, processor, question_text):
    """Encode question text into a single query vector using the model's embed_tokens.

    Uses the thinker's text embedding layer and mean-pools over the sequence
    to produce a single (D,) vector for relevance scoring.
    """
    tokenizer = processor.tokenizer
    token_ids = tokenizer(
        question_text, return_tensors="pt", truncation=True, max_length=128,
    )["input_ids"]

    device = next(model.parameters()).device
    token_ids = token_ids.to(device)

    if hasattr(model, "thinker"):
        embed_layer = model.thinker.model.embed_tokens
    else:
        embed_layer = model.model.embed_tokens

    with torch.no_grad():
        embeddings = embed_layer(token_ids)  # (1, seq_len, D)
        query = embeddings.mean(dim=1).squeeze(0)  # (D,)

    return query


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(dtype_name: str):
    dt = resolve_model_dtype(dtype_name)
    print(f"Loading model from {MODEL_PATH} (dtype={dtype_name}) ...")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dt,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    _record_model_loaded_vram()
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
                  fps, max_pixels, max_new_tokens, use_audio,
                  subset_ratio, prune_mode, alpha, tau,
                  max_frames=None, temperature=0.0, measure_prefill=False):
    prompt = build_user_prompt_for_dataset(dataset, question, choices)
    system_text = SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX
    video_element = {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels}
    if max_frames is not None:
        video_element["max_frames"] = max_frames
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            video_element,
            {"type": "text", "text": prompt},
        ]},
    ]

    # Extract text query embedding for ReDiPrune relevance scoring
    text_query = _get_text_query_embedding(model, processor, question)

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

    # ── ReDiPrune: Frame-level pruning ──
    if prune_mode == "frame" and videos and videos[0] is not None:
        videos_pruned = []
        for vid in videos:
            if vid is not None and vid.shape[0] > 1:
                pruned = rediprune_select_frames(vid, text_query, subset_ratio, alpha, tau)
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

    # ── ReDiPrune: Token-level pruning via processor output manipulation ──
    if prune_mode == "token" and "pixel_values_videos" in inputs:
        pvv = inputs["pixel_values_videos"]
        if pvv is not None and pvv.dim() >= 2 and pvv.shape[0] > 1:
            feat = pvv.float()
            if feat.dim() > 2:
                feat = feat.view(feat.shape[0], -1)
            selected_idx = rediprune_select_text_cond(
                feat, text_query, subset_ratio, alpha, tau,
            )
            if selected_idx.shape[0] < pvv.shape[0]:
                inputs["pixel_values_videos"] = pvv[selected_idx]
                pruned_nframes = selected_idx.shape[0]
                if "video_grid_thw" in inputs:
                    grid = inputs["video_grid_thw"]
                    if grid is not None and grid.dim() == 2:
                        ratio = selected_idx.shape[0] / pvv.shape[0]
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

    gen_in = {k: v for k, v in inputs.items() if k not in _OMNI_GENERATE_BATCH_DROP}

    timing = {}
    if measure_prefill:
        prefill_kw = dict(gen_kw)
        if "thinker_max_new_tokens" in prefill_kw:
            prefill_kw["thinker_max_new_tokens"] = 1
        else:
            prefill_kw["max_new_tokens"] = 1
        with torch.no_grad():
            prefill_ms, _ = cuda_time_ms(lambda: model.generate(**gen_in, **prefill_kw))
        timing["prefill_ms"] = round(prefill_ms, 2)

    with torch.no_grad():
        if measure_prefill:
            e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))
            timing["e2e_ms"] = round(e2e_ms, 2)
        else:
            raw_out = model.generate(**gen_in, **gen_kw)

    seq_ids = _generation_output_token_ids(raw_out)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, orig_nframes, pruned_nframes, timing


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni + ReDiPrune")
    parser.add_argument("--model", default=None)
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--videos", default="/workspace/videos")
    parser.add_argument("--output", default="/workspace/results_rediprune.jsonl")
    parser.add_argument("--log", default="/workspace/eval_rediprune.log")
    parser.add_argument("--errors_log", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=360*420)
    parser.add_argument("--max_frames_videomme", type=int, default=768,
                        help="Max frames for VideoMME videos (paper default: 768)")
    parser.add_argument("--max_frames_other", type=int, default=128,
                        help="Max frames for non-VideoMME datasets (paper default: 128)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Text generation temperature. 0 keeps greedy decoding; >0 enables sampling.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional RNG seed for reproducible sampled-temperature runs.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--vram_log", default="/workspace/vram_log_rediprune.jsonl")
    parser.add_argument("--stderr_log", default=None)
    # ReDiPrune-specific
    parser.add_argument("--subset_ratio", type=float, default=0.5,
                        help="Fraction of visual tokens/frames to KEEP (0.0-1.0). "
                             "ReDiPrune paper uses 0.1 for LLaVA but Qwen2.5-Omni uses fewer "
                             "frames, so 0.3-0.5 is a better starting point.")
    parser.add_argument("--prune_mode", default="frame", choices=["frame", "token"],
                        help="'frame': prune video frames before model (simpler, recommended). "
                             "'token': prune visual token patches after processor (experimental).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight of text relevance vs diversity in scoring. "
                             "0 = pure diversity (like DivPrune), higher = more relevance. "
                             "Paper default: 0.5")
    parser.add_argument("--tau", type=float, default=0.0,
                        help="Relevance threshold for candidate pre-filtering. "
                             "0 = no filtering (use all candidates). Paper default: 0.0")
    parser.add_argument("--measure_prefill", action="store_true",
                        help="Measure prefill time (TTFT) via generate(max_new_tokens=1) with CUDA events.")
    args = parser.parse_args()
    set_run_seed(args.seed)

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
    print(
        f"Running {len(runnable)} entries with ReDiPrune ({args.prune_mode}, ratio={args.subset_ratio}, "
        f"alpha={args.alpha}, tau={args.tau}); temperature={args.temperature}; seed={args.seed}\n"
    )

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
                    before_alloc_gb, before_reserved_gb = _capture_current_vram_gb()
                    torch.cuda.reset_peak_memory_stats()
                    ds_name = _canonicalize(dataset)
                    max_frames = args.max_frames_videomme if ds_name in {"video-mme", "videomme"} else args.max_frames_other
                    pred, reasoning, orig_nf, pruned_nf, timing = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        args.fps, args.max_pixels, args.max_new_tokens, use_audio,
                        args.subset_ratio, args.prune_mode, args.alpha, args.tau,
                        max_frames=max_frames,
                        temperature=args.temperature,
                        measure_prefill=args.measure_prefill,
                    )
                    total_orig_frames += orig_nf
                    total_pruned_frames += pruned_nf
                    vram_entry = {
                        "entry": entry_label, "task_type": task_type,
                        "status": "ok",
                        "method": f"rediprune-{args.prune_mode}",
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "duration_s": entry.get("duration_s"),
                        "orig_frames": orig_nf, "pruned_frames": pruned_nf,
                        "model_loaded_alloc_gb": round(MODEL_LOADED_ALLOC_GB or 0.0, 2),
                        "model_loaded_reserved_gb": round(MODEL_LOADED_RESERVED_GB or 0.0, 2),
                        "before_alloc_gb": round(before_alloc_gb, 2),
                        "before_reserved_gb": round(before_reserved_gb, 2),
                        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                        "after_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                        "after_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                        **timing,
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                except Exception as e:
                    import traceback
                    print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
                    curr_alloc_gb, curr_reserved_gb = _capture_current_vram_gb()
                    before_alloc_gb = locals().get("before_alloc_gb", 0.0)
                    before_reserved_gb = locals().get("before_reserved_gb", 0.0)
                    vram_entry = {
                        "entry": entry_label, "task_type": task_type,
                        "status": "error",
                        "method": f"rediprune-{args.prune_mode}",
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "duration_s": entry.get("duration_s"),
                        "orig_frames": 0,
                        "pruned_frames": 0,
                        "model_loaded_alloc_gb": round(MODEL_LOADED_ALLOC_GB or 0.0, 2),
                        "model_loaded_reserved_gb": round(MODEL_LOADED_RESERVED_GB or 0.0, 2),
                        "before_alloc_gb": round(before_alloc_gb, 2),
                        "before_reserved_gb": round(before_reserved_gb, 2),
                        "peak_alloc_gb": round(peak_alloc_gb, 2),
                        "peak_reserved_gb": round(peak_reserved_gb, 2),
                        "after_alloc_gb": round(curr_alloc_gb, 2),
                        "after_reserved_gb": round(curr_reserved_gb, 2),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                    with open(errors_log_path, "a") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{traceback.format_exc()}\n")
                    pred, reasoning = "ERROR", str(e)
                    orig_nf = pruned_nf = 0
                    timing = {}

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
                    "method": f"rediprune-{args.prune_mode}",
                    "subset_ratio": args.subset_ratio,
                    "alpha": args.alpha,
                    "tau": args.tau,
                    "orig_frames": orig_nf, "pruned_frames": pruned_nf,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    **timing,
                }
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                results.append(result)

                status = "\u2713" if is_correct else "\u2717"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer} frames={orig_nf}->{pruned_nf}")

    acc = correct / total if total else 0
    frame_ratio = total_pruned_frames / total_orig_frames if total_orig_frames else 0
    print(f"\n{'='*50}")
    print(f"Model: qwen2.5-omni + ReDiPrune ({args.prune_mode})")
    print(f"Subset ratio: {args.subset_ratio} | alpha: {args.alpha} | tau: {args.tau}")
    print(f"Effective frame ratio: {frame_ratio:.2%}")
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
