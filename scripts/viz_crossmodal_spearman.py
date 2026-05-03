"""
Cross-modal Spearman + Jaccard comparison: all four attention directions.

At 8 Thinker depth checkpoints, for each video with ≥2 questions, measures
cross-question agreement for four cross-modal Q·K directions:

  text→audio   Q=text,  K=audio  → salience per audio token  (L6 query-inv. result)
  text→video   Q=text,  K=video  → salience per video token  (SparseVLM paradigm)
  audio→video  Q=audio, K=video  → salience per video token  (novel: audio attending video)
  video→audio  Q=video, K=audio  → salience per audio token  (novel: video attending audio)

Metrics computed per direction per layer (all pairwise across questions on same video):

  Spearman ρ          — global rank correlation (inflated by easy-to-rank bottom tokens)
  Top-k Jaccard       — |kept(Q1)∩kept(Q2)| / |kept(Q1)∪kept(Q2)| at OmniZip keep ratio
                        Directly measures pruning decision consistency (replaces Spearman)
  Separation score    — (mean_top_k − mean_bottom) / std; how cleanly signal separates at
                        pruning threshold (replaces Gini)
  Temporal autocorr   — Pearson(scores[:-1], scores[1:]); are adjacent tokens scored alike?
                        High = block-structured saliency, good for segment-level pruning

Layer-to-layer Jaccard (L6 vs L14, per direction):
  Using the per-question mean score as the representative vector, measures whether
  pruning at L6 selects the same tokens as pruning at L14.
  High Jaccard(L6,L14) → L6 is sufficient and avoids deeper forward computation.

Memory note: audio→video avoids materializing T_q × T_k using mean-Q equivalence:
  mean_q(Q·K^T) = mean(Q)·K^T  →  O(H·T_k) instead of O(H·T_q·T_k)

Outputs:
  {out_dir}/crossmodal_stats.jsonl      — one JSON record per video (all metrics + raw scores)
  {out_dir}/crossmodal_jaccard.png      — top-k Jaccard (primary metric)
  {out_dir}/crossmodal_spearman.png     — Spearman ρ (kept for comparison)
  {out_dir}/crossmodal_separation.png   — separation score at pruning threshold
  {out_dir}/crossmodal_autocorr.png     — temporal autocorrelation of scores
  {out_dir}/crossmodal_l6l14_jaccard.png — layer-to-layer Jaccard (L6 vs L14)

Run:
  python viz_crossmodal_spearman.py
  python viz_crossmodal_spearman.py --dataset worldsense --limit_videos 20
  python viz_crossmodal_spearman.py --gpus 0,1,2,3 --limit_videos 50
  python viz_crossmodal_spearman.py --keep_ratio 0.7 --min_questions 3
"""

import argparse
import glob
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNIZIP_DIR = os.path.join(REPO_ROOT, "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None
from qwen_omni_utils import process_mm_info

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--metadata",      default=os.path.join(REPO_ROOT, "videos", "metadata.json"))
parser.add_argument("--videos",        default=os.path.join(REPO_ROOT, "videos"))
parser.add_argument("--model_path",    default="/data/armaan/models/Qwen2.5-Omni-7B")
parser.add_argument("--out_dir",       default=os.path.join(REPO_ROOT, "vizzing", "crossmodal_spearman"))
parser.add_argument("--dataset",       default=None, help="Filter to one dataset (e.g. worldsense)")
parser.add_argument("--limit_videos",  type=int, default=None)
parser.add_argument("--min_questions", type=int, default=1)
parser.add_argument("--gpus",          default="0,1")
parser.add_argument("--fps",                  type=float, default=2.0)
parser.add_argument("--max_frames_videomme",  type=int,   default=768,
                    help="max_frames for video-mme (matches eval_qwen_omni_zip.py)")
parser.add_argument("--max_frames_other",     type=int,   default=128,
                    help="max_frames for all other datasets")
parser.add_argument("--max_pixels",           type=int,   default=360 * 420)
# OmniZip default: rho_audio=0.3 → drops 30% → keeps 70% of audio tokens.
# Used for top-k Jaccard and separation score.
parser.add_argument("--keep_ratio",   type=float, default=0.70,
                    help="Fraction of tokens kept at pruning threshold (OmniZip default=0.70)")
args = parser.parse_args()

GPU_IDS         = [int(g) for g in args.gpus.split(",")]
HOOK_LAYERS     = [0, 1, 3, 6, 10, 14, 20, 27]
SAVE_RAW_LAYERS = [6, 14]
LAYER_PAIR      = (6, 14)   # layer-to-layer Jaccard comparison
# L6 is where text→audio cross-attention was found query-invariant (Spearman ≥ 0.999)
# in prior experiments. OmniZip itself uses audio-encoder self-attention (pre-LLM).
HIGHLIGHT_LAYER = 6

os.makedirs(args.out_dir, exist_ok=True)
LOG_PATH = os.path.join(args.out_dir, "crossmodal_stats.jsonl")

# Direction definitions: (label, Q-modality, K-modality)
# Score semantics: for each K token, how strongly do Q tokens attend to it on average.
DIRECTIONS = [
    ("text→audio",  "text",  "audio"),
    ("text→video",  "text",  "video"),
    ("audio→video", "audio", "video"),
    ("video→audio", "video", "audio"),
]
DIR_LABELS   = [d[0] for d in DIRECTIONS]
DIR_COLORS   = ["#27ae60", "#2980b9", "#e67e22", "#8e44ad"]
DIR_MARKERS  = ["o",       "s",       "^",       "D"]
DIR_LINES    = ["-",       "--",      "-.",      ":"]

# ── Pure-numpy helpers ────────────────────────────────────────────────────────
def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v))
        return r
    n = len(x)
    if n < 2:
        return float("nan")
    d = rank(x) - rank(y)
    return float(1.0 - 6.0 * float(np.dot(d, d)) / (n * (n ** 2 - 1)))


def gini(arr: np.ndarray) -> float:
    a = np.sort(np.abs(arr))
    n, total = len(a), a.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2.0 * np.dot(np.arange(1, n + 1), a)) / (n * total) - (n + 1) / n)


def topk_jaccard(x: np.ndarray, y: np.ndarray, keep_ratio: float) -> float:
    """Jaccard similarity of top-k sets at the pruning threshold.

    k = int(keep_ratio * len(x)).  Returns NaN if k==0 or inputs are too short.
    Directly measures whether two questions would make the same pruning decision.
    Unlike Spearman, this is insensitive to rank agreement in the discarded tail.
    """
    n = len(x)
    k = max(1, int(round(keep_ratio * n)))
    if k >= n:
        return 1.0
    top_x = set(np.argpartition(x, -k)[-k:].tolist())
    top_y = set(np.argpartition(y, -k)[-k:].tolist())
    inter = len(top_x & top_y)
    union = len(top_x | top_y)
    return float(inter / union) if union > 0 else float("nan")


def separation_score(arr: np.ndarray, keep_ratio: float) -> float:
    """(mean_top_k − mean_bottom) / std(arr) at the pruning threshold.

    Measures how cleanly the signal separates kept tokens from dropped tokens.
    More interpretable than Gini: directly captures signal quality at the
    decision boundary rather than global concentration.
    Returns NaN for constant or very short arrays.
    """
    n = len(arr)
    if n < 2:
        return float("nan")
    k = max(1, int(round(keep_ratio * n)))
    idx = np.argpartition(arr, -k)
    top_mean = float(arr[idx[-k:]].mean())
    bot_mean = float(arr[idx[:-k]].mean()) if n - k > 0 else top_mean
    std = float(arr.std())
    if std < 1e-10:
        return float("nan")
    return (top_mean - bot_mean) / std


def temporal_autocorr(arr: np.ndarray) -> float:
    """Pearson correlation between adjacent elements (lag-1 autocorrelation).

    High value → saliency is smooth / block-structured (speech segments vs silence).
    Low value  → saliency is noisy, less useful for segment-level pruning.
    """
    if len(arr) < 3:
        return float("nan")
    x, y = arr[:-1], arr[1:]
    if x.std() < 1e-10 or y.std() < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def layer_jaccard(scores_li_a: np.ndarray, scores_li_b: np.ndarray,
                  keep_ratio: float) -> float:
    """Jaccard between top-k sets of two layers for the same video.

    Uses per-question mean score as the representative vector per layer.
    Answers: does pruning at layer A select the same tokens as layer B?
    """
    return topk_jaccard(scores_li_a, scores_li_b, keep_ratio)


def resolve_video_path(file_field: str, videos_dir: str):
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
    filename  = rel_norm.split("/")[-1]
    for match in glob.glob(os.path.join(videos_dir, "**", filename), recursive=True):
        if match.replace("\\", "/").endswith(rel_norm):
            return match
    matches = glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
    return matches[0] if len(matches) == 1 else None


def safe_dir_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


THINKER_KEYS = {
    "input_ids", "attention_mask",
    "pixel_values", "pixel_values_videos",
    "image_grid_thw", "video_grid_thw",
    "input_features", "feature_attention_mask",
    "audio_feature_lengths", "video_second_per_grid",
    "inputs_embeds", "position_ids",
    "past_key_values", "rope_deltas",
    "use_cache", "cache_position", "labels",
}

# ── Load model ────────────────────────────────────────────────────────────────
max_memory = {i: "10GiB" for i in GPU_IDS}
print(f"Loading Qwen2.5-Omni-7B across GPUs {GPU_IDS} (≤10GiB each) …")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,
    attn_implementation="flash_attention_2",
)
INPUT_DEVICE   = model.thinker.model.embed_tokens.weight.device
model.eval()
if hasattr(model, "token2wav"):
    try:
        has_meta = any(p.is_meta for p in model.token2wav.parameters())
        if not has_meta:
            model.token2wav = model.token2wav.to("cpu")
    except Exception:
        pass
processor      = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id
print(f"Model ready. Input device: {INPUT_DEVICE}\n")

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(args.metadata) as f:
    raw = json.load(f)

entries = list(raw.values()) if isinstance(raw, dict) else raw
if args.dataset:
    canon   = args.dataset.strip().lower().replace("_", "-").replace(" ", "-")
    entries = [e for e in entries if e.get("dataset", "").strip().lower().replace("_", "-") == canon]
entries = [e for e in entries if len(e.get("questions", [])) >= args.min_questions]
if args.limit_videos:
    entries = entries[:args.limit_videos]
print(f"Processing {len(entries)} video entries (≥{args.min_questions} Q each; cross-Q metrics require ≥2) …\n")


# ── Score one (video, question) across all hook layers and all 4 directions ───
def score_question(video_path: str, question_text: str, dataset: str = ""):
    """
    Returns (layer_scores, meta) where:
      layer_scores: {layer_idx: {direction_label: np.ndarray | None}}

    For direction "X→Y": score[j] is proportional to how much X tokens
    attend to Y-token j on average (mean over X queries and heads).
    Uses mean-Q trick: mean_q(Q·K^T) = mean(Q)·K^T — avoids materializing
    the full T_q × T_k matrix, critical for audio→video where both are large.
    """
    ds_canon = dataset.strip().lower().replace("_", "-").replace(" ", "-")
    max_frames = (args.max_frames_videomme
                  if ds_canon in {"video-mme", "videomme"}
                  else args.max_frames_other)
    system_text = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    video_element = {
        "type": "video", "video": video_path,
        "fps": args.fps, "max_frames": max_frames, "max_pixels": args.max_pixels,
    }
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": [video_element, {"type": "text", "text": question_text}]},
    ]
    text   = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    model.thinker.nframes  = videos[0].shape[0]

    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(INPUT_DEVICE)
    for k, v in inputs.items():
        if v.is_floating_point():
            inputs[k] = v.to(torch.float16)
    thinker_inputs = {k: v for k, v in inputs.items() if k in THINKER_KEYS}

    ids       = inputs["input_ids"][0].cpu()
    audio_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
    video_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
    text_pos  = ((ids != audio_token_id) & (ids != video_token_id)).nonzero(as_tuple=True)[0]

    audio_pos_gpu = audio_pos.to(INPUT_DEVICE)
    video_pos_gpu = video_pos.to(INPUT_DEVICE)
    text_pos_gpu  = text_pos.to(INPUT_DEVICE)

    layer_scores: dict[int, dict] = {}

    def make_hook(li: int):
        def hook(module, args_h, kwargs, output):
            with torch.no_grad():
                hidden_states = args_h[0] if args_h else kwargs["hidden_states"]
                hs  = hidden_states[0]   # (seq, D) — drop batch dim
                dev = hs.device

                num_heads = module.num_heads
                num_kv    = module.num_key_value_heads
                head_dim  = module.head_dim
                scale     = head_dim ** 0.5

                pos_map = {
                    "text":  text_pos_gpu.to(dev),
                    "audio": audio_pos_gpu.to(dev),
                    "video": video_pos_gpu.to(dev),
                }

                def _cross(q_pos, k_pos):
                    """
                    Cross-modal score: for each K-token, mean attention logit
                    from all Q-tokens (averaged over heads).

                    mean-Q trick: mean_q(Q·K^T) = mean(Q)·K^T
                    Reduces memory from O(H·T_q·T_k) to O(H·T_k).
                    """
                    if q_pos.numel() == 0 or k_pos.numel() == 0:
                        return None

                    T_q    = q_pos.numel()
                    q_raw  = module.q_proj(hs[q_pos]).float()           # (T_q, H*d)
                    q_all  = q_raw.view(T_q, num_heads, head_dim).permute(1, 0, 2)  # (H, T_q, d)
                    q_mean = q_all.mean(dim=1, keepdim=True)            # (H, 1, d)

                    T_k   = k_pos.numel()
                    k_raw = module.k_proj(hs[k_pos]).float()            # (T_k, Hkv*d)
                    k     = k_raw.view(T_k, num_kv, head_dim).permute(1, 0, 2)  # (Hkv, T_k, d)
                    if num_heads != num_kv:
                        k = k.repeat_interleave(num_heads // num_kv, dim=0)     # (H, T_k, d)

                    # (H, 1, T_k) → squeeze → (H, T_k) → mean heads → (T_k,)
                    scores = torch.matmul(q_mean, k.transpose(-2, -1)) / scale
                    return scores.squeeze(1).mean(dim=0).cpu().numpy()

                layer_scores[li] = {
                    "text→audio":  _cross(pos_map["text"],  pos_map["audio"]),
                    "text→video":  _cross(pos_map["text"],  pos_map["video"]),
                    "audio→video": _cross(pos_map["audio"], pos_map["video"]),
                    "video→audio": _cross(pos_map["video"], pos_map["audio"]),
                }
            return output
        return hook

    hooks = [
        model.thinker.model.layers[li].self_attn.register_forward_hook(
            make_hook(li), with_kwargs=True
        )
        for li in HOOK_LAYERS
    ]
    try:
        with torch.no_grad():
            model.thinker(
                **thinker_inputs,
                output_attentions=False,
                use_audio_in_video=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

    meta = {
        "seq_len": int(ids.shape[0]),
        "n_audio": int(audio_pos.numel()),
        "n_video": int(video_pos.numel()),
        "n_text":  int(text_pos.numel()),
    }
    return layer_scores, meta


# ── Accumulators: direction → layer → list of per-video values ───────────────
acc_sp  = {d: {li: [] for li in HOOK_LAYERS} for d in DIR_LABELS}  # Spearman ρ
acc_jk  = {d: {li: [] for li in HOOK_LAYERS} for d in DIR_LABELS}  # top-k Jaccard (primary)
acc_sep = {d: {li: [] for li in HOOK_LAYERS} for d in DIR_LABELS}  # separation score
acc_tac = {d: {li: [] for li in HOOK_LAYERS} for d in DIR_LABELS}  # temporal autocorr
acc_gi  = {d: {li: [] for li in HOOK_LAYERS} for d in DIR_LABELS}  # Gini (kept for ref)
acc_sd  = {d: {li: [] for li in HOOK_LAYERS} for d in DIR_LABELS}  # spread
# Layer-to-layer Jaccard: direction → list of per-video values (one scalar per video)
acc_l2l = {d: [] for d in DIR_LABELS}
n_videos_processed = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
with open(LOG_PATH, "a") as log_f:
    for v_idx, entry in enumerate(entries):
        video_path = resolve_video_path(entry["file"], args.videos)
        dataset    = entry.get("dataset", "unknown")
        video_id   = safe_dir_name(Path(entry["file"]).stem)
        questions  = [q["question"] for q in entry["questions"]]

        if video_path is None:
            print(f"SKIP (no video): {entry['file']}")
            continue

        print(f"\n[Video {v_idx+1}/{len(entries)}] {video_id}  ({len(questions)} Qs)")

        q_scores: list[dict] = []
        seq_meta = None

        for q_idx, question in enumerate(questions):
            print(f"  Q{q_idx}: {question[:70]}")
            try:
                layer_scores, meta = score_question(video_path, question, dataset)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            if seq_meta is None:
                seq_meta = meta
            q_scores.append(layer_scores)

        if len(q_scores) < 1:
            print("  No successful question runs — skipping")
            continue
        if len(q_scores) < 2:
            print("  Single question only — skipping cross-Q pairwise metrics, still saving raw scores")

        # -- Per-layer, per-direction metrics for this video -------------------
        v_sp  = {d: {} for d in DIR_LABELS}
        v_jk  = {d: {} for d in DIR_LABELS}
        v_sep = {d: {} for d in DIR_LABELS}
        v_tac = {d: {} for d in DIR_LABELS}
        v_gi  = {d: {} for d in DIR_LABELS}
        v_sd  = {d: {} for d in DIR_LABELS}

        for li in HOOK_LAYERS:
            for dir_label in DIR_LABELS:
                vecs = [
                    q[li][dir_label]
                    for q in q_scores
                    if li in q and q[li].get(dir_label) is not None
                ]
                if len(vecs) < 1:
                    for acc in (v_sp, v_jk, v_sep, v_tac, v_gi, v_sd):
                        acc[dir_label][li] = None
                    continue

                # Cross-Q pairwise metrics — only if ≥2 questions
                if len(vecs) >= 2:
                    pairs_sp = [spearman_r(x, y)                    for x, y in combinations(vecs, 2)]
                    pairs_jk = [topk_jaccard(x, y, args.keep_ratio) for x, y in combinations(vecs, 2)]
                    sp_val   = float(np.nanmean(pairs_sp))
                    jk_val   = float(np.nanmean(pairs_jk))
                else:
                    sp_val = None
                    jk_val = None

                # Per-question metrics — work with any number of questions
                sep_val = float(np.mean([separation_score(v, args.keep_ratio) for v in vecs]))
                tac_val = float(np.mean([temporal_autocorr(v) for v in vecs]))
                gi_val  = float(np.mean([gini(v) for v in vecs]))
                sd_val  = float(np.mean([v.std() for v in vecs]))

                v_sp[dir_label][li]  = sp_val
                v_jk[dir_label][li]  = jk_val
                v_sep[dir_label][li] = sep_val
                v_tac[dir_label][li] = tac_val
                v_gi[dir_label][li]  = gi_val
                v_sd[dir_label][li]  = sd_val
                if sp_val  is not None: acc_sp[dir_label][li].append(sp_val)
                if jk_val  is not None: acc_jk[dir_label][li].append(jk_val)
                acc_sep[dir_label][li].append(sep_val)
                acc_tac[dir_label][li].append(tac_val)
                acc_gi[dir_label][li].append(gi_val)
                acc_sd[dir_label][li].append(sd_val)

        # -- Layer-to-layer Jaccard: L6 vs L14 per direction ------------------
        v_l2l = {}
        la, lb = LAYER_PAIR
        for dir_label in DIR_LABELS:
            # Use per-question mean score as the representative vector per layer
            vecs_a = [q[la][dir_label] for q in q_scores
                      if la in q and q[la].get(dir_label) is not None]
            vecs_b = [q[lb][dir_label] for q in q_scores
                      if lb in q and q[lb].get(dir_label) is not None]
            if vecs_a and vecs_b:
                mean_a = np.mean(vecs_a, axis=0)
                mean_b = np.mean(vecs_b, axis=0)
                jval = layer_jaccard(mean_a, mean_b, args.keep_ratio)
                v_l2l[dir_label] = jval
                acc_l2l[dir_label].append(jval)
            else:
                v_l2l[dir_label] = None

        n_videos_processed += 1

        # Terminal table (Jaccard as primary column)
        col_w = 10
        print(f"\n  {'Layer':>6}  " +
              "  ".join(f"{'Jk:'+d:>{col_w+3}}" for d in DIR_LABELS) +
              "  (top-k Jaccard, keep={:.0%})".format(args.keep_ratio))
        for li in HOOK_LAYERS:
            row = f"  {li:>6}  "
            for d in DIR_LABELS:
                v = v_jk[d].get(li)
                row += f"  {v:>{col_w}.4f}" if v is not None else f"  {'N/A':>{col_w}}"
            marker = "  ← L6" if li == HIGHLIGHT_LAYER else ""
            print(row + marker)
        print(f"  L{la}→L{lb} Jaccard: " +
              "  ".join(f"{d}={v_l2l[d]:.4f}" if v_l2l[d] is not None else f"{d}=N/A"
                        for d in DIR_LABELS))

        # Raw score arrays at SAVE_RAW_LAYERS for offline analysis
        raw_scores = {
            str(li): {
                d: [q[li][d].tolist() for q in q_scores if li in q and q[li].get(d) is not None]
                for d in DIR_LABELS
            }
            for li in SAVE_RAW_LAYERS
        }

        record = {
            "video_id":     video_id,
            "dataset":      dataset,
            "file":         entry["file"],
            "n_questions":  len(questions),
            "n_successful": len(q_scores),
            "questions":    questions,
            "keep_ratio":   args.keep_ratio,
            "seq_meta":     seq_meta,
            # Primary metrics
            "topk_jaccard":    {d: {str(li): v for li, v in v_jk[d].items()  if v is not None} for d in DIR_LABELS},
            "separation":      {d: {str(li): v for li, v in v_sep[d].items() if v is not None} for d in DIR_LABELS},
            "temporal_autocorr": {d: {str(li): v for li, v in v_tac[d].items() if v is not None} for d in DIR_LABELS},
            "l2l_jaccard":     {d: v_l2l[d] for d in DIR_LABELS if v_l2l[d] is not None},
            # Reference metrics (kept for comparison)
            "spearman": {d: {str(li): v for li, v in v_sp[d].items() if v is not None} for d in DIR_LABELS},
            "gini":     {d: {str(li): v for li, v in v_gi[d].items() if v is not None} for d in DIR_LABELS},
            "spread":   {d: {str(li): v for li, v in v_sd[d].items() if v is not None} for d in DIR_LABELS},
            "raw_scores": raw_scores,
        }
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

print(f"\nProcessed {n_videos_processed} videos.")

# ── Aggregate helpers ─────────────────────────────────────────────────────────
def agg(acc_dict):
    means = np.array([np.mean(acc_dict[li]) if acc_dict[li] else np.nan for li in HOOK_LAYERS])
    stds  = np.array([np.std(acc_dict[li])  if acc_dict[li] else np.nan for li in HOOK_LAYERS])
    return means, stds


layers = np.array(HOOK_LAYERS)


def _make_plot(metric_acc, ylabel, title, filename, ylim=None, extra_hlines=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    for (dir_label, _, _), color, marker, ls in zip(DIRECTIONS, DIR_COLORS, DIR_MARKERS, DIR_LINES):
        m, s = agg(metric_acc[dir_label])
        ax.plot(layers, m, marker=marker, linestyle=ls, color=color, lw=2.2,
                label=dir_label, zorder=3)
        ax.fill_between(layers, m - s, m + s, alpha=0.12, color=color)

    ax.axvline(HIGHLIGHT_LAYER, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.text(HIGHLIGHT_LAYER + 0.25, ax.get_ylim()[0] if ylim is None else ylim[0],
            f"L{HIGHLIGHT_LAYER}\n(query-inv.)", fontsize=8.5, color="gray", va="bottom")

    if extra_hlines:
        for y_val, label, color in extra_hlines:
            ax.axhline(y_val, color=color, lw=0.9, ls=":", alpha=0.6)
            ax.text(layers[-1] + 0.3, y_val, label, fontsize=8, color=color, va="center")

    ax.set_xlabel("Thinker layer", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xticks(layers)
    if ylim:
        ax.set_ylim(ylim)
    ax.text(0.99, 0.02, f"n={n_videos_processed} videos",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="gray")
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


print("\nGenerating summary plots …")

# Primary: top-k Jaccard
_make_plot(
    acc_jk,
    ylabel=f"Cross-Q top-k Jaccard  (k={args.keep_ratio:.0%} kept)",
    title=(
        "Cross-Modal Attention: Pruning Decision Consistency vs. Layer Depth\n"
        f"Mean pairwise Jaccard of top-{args.keep_ratio:.0%} token sets across questions on same video\n"
        "(1.0 = identical pruning decision regardless of question)"
    ),
    filename="crossmodal_jaccard.png",
    ylim=(-0.05, 1.08),
    extra_hlines=[(0.90, "J=0.90", "red"), (0.70, "J=0.70", "orange")],
)

# Separation score
_make_plot(
    acc_sep,
    ylabel="Separation score  (mean_top − mean_bot) / std",
    title=(
        "Cross-Modal Attention: Signal Quality at Pruning Threshold vs. Layer Depth\n"
        f"(mean_top_{args.keep_ratio:.0%} − mean_bottom) / std  —  higher = cleaner separation"
    ),
    filename="crossmodal_separation.png",
)

# Temporal autocorrelation
_make_plot(
    acc_tac,
    ylabel="Temporal autocorrelation (lag-1 Pearson)",
    title=(
        "Cross-Modal Attention: Score Smoothness vs. Layer Depth\n"
        "Pearson(scores[t], scores[t+1])  —  high = block-structured saliency (good for segment pruning)"
    ),
    filename="crossmodal_autocorr.png",
    ylim=(-0.15, 1.08),
    extra_hlines=[(0.5, "r=0.5", "gray")],
)

# Spearman (kept for reference)
_make_plot(
    acc_sp,
    ylabel="Cross-Q Spearman ρ  (reference only)",
    title=(
        "Cross-Modal Attention: Global Rank Correlation vs. Layer Depth  [reference]\n"
        "Inflated by agreement in discarded tail — use Jaccard for pruning decisions"
    ),
    filename="crossmodal_spearman.png",
    ylim=(-0.15, 1.08),
    extra_hlines=[(0.99, "ρ=0.99", "red")],
)

# Layer-to-layer Jaccard bar chart (L6 vs L14)
la, lb = LAYER_PAIR
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(DIR_LABELS))
means = [float(np.mean(acc_l2l[d])) if acc_l2l[d] else np.nan for d in DIR_LABELS]
stds  = [float(np.std(acc_l2l[d]))  if acc_l2l[d] else 0.0   for d in DIR_LABELS]
bars = ax.bar(x, means, yerr=stds, color=DIR_COLORS, alpha=0.82, capsize=5,
              error_kw={"elinewidth": 1.5})
ax.axhline(0.90, color="red",    lw=1.0, ls="--", alpha=0.7, label="J=0.90")
ax.axhline(0.70, color="orange", lw=1.0, ls="--", alpha=0.7, label="J=0.70")
ax.set_xticks(x)
ax.set_xticklabels(DIR_LABELS, fontsize=11)
ax.set_ylabel(f"Jaccard(L{la}, L{lb})", fontsize=12)
ax.set_title(
    f"Layer-to-Layer Jaccard: L{la} vs L{lb}  (mean ± std across videos)\n"
    f"Do top-{args.keep_ratio:.0%} sets match at the early vs. deeper layer?  "
    f"High → L{la} is sufficient.",
    fontsize=11,
)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar, m, s in zip(bars, means, stds):
    if not np.isnan(m):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
ax.text(0.99, 0.02, f"n={n_videos_processed} videos",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="gray")
plt.tight_layout()
out_l2l = os.path.join(args.out_dir, "crossmodal_l6l14_jaccard.png")
plt.savefig(out_l2l, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out_l2l}")

# ── Aggregate tables ──────────────────────────────────────────────────────────
col_w = 16
for metric_name, metric_acc in [("Top-k Jaccard (primary)", acc_jk),
                                  ("Separation score",        acc_sep),
                                  ("Temporal autocorr",       acc_tac),
                                  ("Spearman ρ (reference)",  acc_sp)]:
    print(f"\n── {metric_name}  (mean ± std across videos) ──")
    print(f"{'Layer':>6}  " + "  ".join(f"{d:>{col_w}}" for d in DIR_LABELS))
    for i, li in enumerate(HOOK_LAYERS):
        row = f"{li:>6}  "
        for d in DIR_LABELS:
            m, s = agg(metric_acc[d])
            val = f"{m[i]:.4f}±{s[i]:.4f}" if not np.isnan(m[i]) else "N/A"
            row += f"  {val:>{col_w}}"
        marker = "  ← L6" if li == HIGHLIGHT_LAYER else ""
        print(row + marker)

print(f"\n── Layer-to-layer Jaccard L{la}↔L{lb} ──")
for d in DIR_LABELS:
    vals = acc_l2l[d]
    if vals:
        print(f"  {d}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  n={len(vals)}")
    else:
        print(f"  {d}: N/A")

print(f"\nLog   → {LOG_PATH}")
print(f"Plots → {args.out_dir}/")
