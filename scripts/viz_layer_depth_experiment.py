"""
Layer depth experiment: at what Thinker depth does question-sensitivity emerge?

For each video with ≥2 questions, hooks 8 depth checkpoints and measures per layer:
  1. Cross-Q Spearman ρ  — mean pairwise rank correlation between score vectors from
     different questions on the SAME video.
     High ρ → question-INsensitive (same map regardless of Q)
     Low ρ  → question-SENSITIVE (map shifts with the question)
  2. Audio/Video Gini    — how concentrated scores are (0=uniform, 1=single spike).
     High Gini → few tokens dominate → potentially more compressible.
  3. Score spread (std)  — absolute dispersion; higher = more dynamic range for ranking.

Outputs:
  {out_dir}/layer_depth_stats.jsonl   — one JSON record per video
  {out_dir}/layer_depth_summary.png   — 3-panel aggregate plot (mean ± std across videos)

Run:
  python viz_layer_depth_experiment.py --metadata videos/metadata.json --gpus 0,1,2,3
  python viz_layer_depth_experiment.py --dataset worldsense --limit_videos 10
  python viz_layer_depth_experiment.py --limit_videos 5 --min_questions 2
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
parser.add_argument("--metadata",      default=os.path.join(REPO_ROOT, "metadata.json"))
parser.add_argument("--videos",        default=os.path.join(REPO_ROOT, "videos"))
parser.add_argument("--model_path",    default="/data/armaan/models/Qwen2.5-Omni-7B")
parser.add_argument("--out_dir",       default=os.path.join(REPO_ROOT, "vizzing", "layer_depth_experiment"))
parser.add_argument("--dataset",       default=None, help="Filter to one dataset (e.g. worldsense)")
parser.add_argument("--limit_videos",  type=int, default=None, help="Max multi-Q videos to process")
parser.add_argument("--min_questions", type=int, default=2,    help="Min questions per video")
parser.add_argument("--gpus",          default="0,1", help="Comma-separated GPU IDs")
parser.add_argument("--fps",           type=float, default=2.0)
parser.add_argument("--max_frames",    type=int,   default=128)
parser.add_argument("--max_pixels",    type=int,   default=360 * 420)
args = parser.parse_args()

GPU_IDS     = [int(g) for g in args.gpus.split(",")]
HOOK_LAYERS  = [0, 1, 3, 6, 10, 14, 20, 27]
SAVE_RAW_LAYERS = [6, 14]   # save per-question score arrays for offline Jaccard/separation/autocorr
os.makedirs(args.out_dir, exist_ok=True)
LOG_PATH = os.path.join(args.out_dir, "layer_depth_stats.jsonl")

# ── Pure-numpy Spearman (no scipy dependency) ─────────────────────────────────
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
    """Gini coefficient of absolute values in arr. 0 = uniform, 1 = single spike."""
    a = np.sort(np.abs(arr))
    n = len(a)
    total = a.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2.0 * np.dot(np.arange(1, n + 1), a)) / (n * total) - (n + 1) / n)


# ── Video path resolver (mirrors eval_qwen_omni.py) ───────────────────────────
def resolve_video_path(file_field: str, videos_dir: str) -> str | None:
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
INPUT_DEVICE = model.thinker.model.embed_tokens.weight.device
model.eval()
if hasattr(model, "token2wav"):
    model.token2wav = model.token2wav.to("cpu")
processor      = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id
print(f"Model ready. Input device: {INPUT_DEVICE}\n")

# ── Load & filter metadata ────────────────────────────────────────────────────
with open(args.metadata) as f:
    raw = json.load(f)

entries = list(raw.values()) if isinstance(raw, dict) else raw
if args.dataset:
    canon   = args.dataset.strip().lower().replace("_", "-").replace(" ", "-")
    entries = [e for e in entries
               if e.get("dataset", "").strip().lower().replace("_", "-") == canon]

entries = [e for e in entries if len(e.get("questions", [])) >= args.min_questions]
if args.limit_videos:
    entries = entries[: args.limit_videos]

print(f"Processing {len(entries)} multi-question video entries (≥{args.min_questions} Qs each) …\n")


# ── Score one (video, question) pair across all HOOK_LAYERS ──────────────────
def score_question(video_path: str, question_text: str):
    """
    Returns (layer_scores, meta) where:
      layer_scores: {layer_idx: {'audio': np.ndarray|None, 'video': np.ndarray|None}}
      meta:         {'seq_len', 'n_audio', 'n_video', 'n_text'}
    """
    system_text = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    video_element = {
        "type": "video", "video": video_path,
        "fps": args.fps, "max_frames": args.max_frames, "max_pixels": args.max_pixels,
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
                hs  = hidden_states[0]   # (seq, D)
                dev = hs.device

                t_pos = text_pos_gpu.to(dev)
                if t_pos.numel() == 0:
                    layer_scores[li] = {"audio": None, "video": None}
                    return output

                num_heads = module.num_heads
                num_kv    = module.num_key_value_heads
                head_dim  = module.head_dim
                T_text    = t_pos.numel()

                q_all = module.q_proj(hs[t_pos]).float()           # (T_text, H*d)
                q     = q_all.view(T_text, num_heads, head_dim).permute(1, 0, 2)

                def _cross(pos_gpu):
                    pos = pos_gpu.to(dev)
                    if pos.numel() == 0:
                        return None
                    k_all = module.k_proj(hs[pos]).float()         # (T_mod, Hkv*d)
                    T_mod = pos.numel()
                    k = k_all.view(T_mod, num_kv, head_dim).permute(1, 0, 2)
                    if num_heads != num_kv:
                        k = k.repeat_interleave(num_heads // num_kv, dim=0)
                    # (num_heads, T_text, T_mod) → mean heads & text → (T_mod,)
                    s = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                    return s.mean(dim=(0, 1)).cpu().numpy()

                layer_scores[li] = {
                    "audio": _cross(audio_pos_gpu),
                    "video": _cross(video_pos_gpu),
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


# ── Per-layer aggregate accumulators ─────────────────────────────────────────
acc_audio_sp  = {li: [] for li in HOOK_LAYERS}  # cross-Q Spearman (one value per video)
acc_video_sp  = {li: [] for li in HOOK_LAYERS}
acc_audio_gi  = {li: [] for li in HOOK_LAYERS}  # Gini (mean across Qs per video)
acc_video_gi  = {li: [] for li in HOOK_LAYERS}
acc_audio_std = {li: [] for li in HOOK_LAYERS}  # spread (mean std per video)
acc_video_std = {li: [] for li in HOOK_LAYERS}

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

        print(f"\n[Video {v_idx+1}/{len(entries)}] {video_id}  ({len(questions)} questions)")

        # -- Collect per-Q, per-layer score arrays for this video ---------------
        q_audio_by_layer: list[dict[int, np.ndarray]] = []
        q_video_by_layer: list[dict[int, np.ndarray]] = []
        seq_meta = None

        for q_idx, question in enumerate(questions):
            print(f"  Q{q_idx}: {question[:70]}")
            try:
                layer_scores, meta = score_question(video_path, question)
            except Exception as exc:
                print(f"    ERROR: {exc}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            if seq_meta is None:
                seq_meta = meta

            q_audio_by_layer.append({
                li: layer_scores[li]["audio"]
                for li in HOOK_LAYERS
                if layer_scores.get(li, {}).get("audio") is not None
            })
            q_video_by_layer.append({
                li: layer_scores[li]["video"]
                for li in HOOK_LAYERS
                if layer_scores.get(li, {}).get("video") is not None
            })

        if len(q_audio_by_layer) < 2:
            print("  Not enough successful questions for cross-Q analysis — skipping")
            continue

        # -- Compute per-layer metrics for this video --------------------------
        v_audio_sp = {}; v_video_sp = {}
        v_audio_gi = {}; v_video_gi = {}
        v_audio_sd = {}; v_video_sd = {}

        for li in HOOK_LAYERS:
            a_vecs = [q[li] for q in q_audio_by_layer if li in q]
            v_vecs = [q[li] for q in q_video_by_layer if li in q]

            # Cross-Q Spearman: all pairs
            a_pairs = [spearman_r(x, y) for x, y in combinations(a_vecs, 2)] if len(a_vecs) >= 2 else []
            v_pairs = [spearman_r(x, y) for x, y in combinations(v_vecs, 2)] if len(v_vecs) >= 2 else []
            v_audio_sp[li] = float(np.nanmean(a_pairs)) if a_pairs else None
            v_video_sp[li] = float(np.nanmean(v_pairs)) if v_pairs else None

            # Gini & spread averaged across questions
            v_audio_gi[li] = float(np.mean([gini(v) for v in a_vecs])) if a_vecs else None
            v_video_gi[li] = float(np.mean([gini(v) for v in v_vecs])) if v_vecs else None
            v_audio_sd[li] = float(np.mean([v.std() for v in a_vecs])) if a_vecs else None
            v_video_sd[li] = float(np.mean([v.std() for v in v_vecs])) if v_vecs else None

            # Push into global accumulators
            if v_audio_sp[li] is not None: acc_audio_sp[li].append(v_audio_sp[li])
            if v_video_sp[li] is not None: acc_video_sp[li].append(v_video_sp[li])
            if v_audio_gi[li] is not None: acc_audio_gi[li].append(v_audio_gi[li])
            if v_video_gi[li] is not None: acc_video_gi[li].append(v_video_gi[li])
            if v_audio_sd[li] is not None: acc_audio_std[li].append(v_audio_sd[li])
            if v_video_sd[li] is not None: acc_video_std[li].append(v_video_sd[li])

        # -- Terminal table for this video -------------------------------------
        print(f"\n  {'Layer':>6}  {'AudSpear':>9}  {'AudGini':>8}  {'AudStd':>8}"
              f"  {'VidSpear':>9}  {'VidGini':>8}  {'VidStd':>8}")
        for li in HOOK_LAYERS:
            def _f(v):
                return f"{v:.4f}" if v is not None else "  N/A  "
            print(f"  {li:>6}  {_f(v_audio_sp[li]):>9}  {_f(v_audio_gi[li]):>8}  {_f(v_audio_sd[li]):>8}"
                  f"  {_f(v_video_sp[li]):>9}  {_f(v_video_gi[li]):>8}  {_f(v_video_sd[li]):>8}")

        # -- Log to JSONL -------------------------------------------------------
        # Save raw per-question score arrays for SAVE_RAW_LAYERS so that
        # Jaccard, separation, and autocorrelation can be computed offline.
        raw_audio = {
            str(li): [q[li].tolist() for q in q_audio_by_layer if li in q]
            for li in SAVE_RAW_LAYERS
        }
        raw_video = {
            str(li): [q[li].tolist() for q in q_video_by_layer if li in q]
            for li in SAVE_RAW_LAYERS
        }

        record = {
            "video_id":     video_id,
            "dataset":      dataset,
            "file":         entry["file"],
            "n_questions":  len(questions),
            "n_successful": len(q_audio_by_layer),
            "seq_meta":     seq_meta,
            "audio_spearman": {str(li): v for li, v in v_audio_sp.items() if v is not None},
            "video_spearman": {str(li): v for li, v in v_video_sp.items() if v is not None},
            "audio_gini":     {str(li): v for li, v in v_audio_gi.items() if v is not None},
            "video_gini":     {str(li): v for li, v in v_video_gi.items() if v is not None},
            "audio_spread":   {str(li): v for li, v in v_audio_sd.items() if v is not None},
            "video_spread":   {str(li): v for li, v in v_video_sd.items() if v is not None},
            # Raw arrays at SAVE_RAW_LAYERS — one list per question
            "raw_audio_scores": raw_audio,
            "raw_video_scores": raw_video,
        }
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()

# ── Aggregate summary plots ───────────────────────────────────────────────────
print("\nGenerating summary plots …")

layers = np.array(HOOK_LAYERS)

def agg(d: dict) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([np.mean(d[li]) if d[li] else np.nan for li in HOOK_LAYERS])
    stds  = np.array([np.std(d[li])  if d[li] else np.nan for li in HOOK_LAYERS])
    return means, stds

audio_sp_m, audio_sp_s = agg(acc_audio_sp)
video_sp_m, video_sp_s = agg(acc_video_sp)
audio_gi_m, audio_gi_s = agg(acc_audio_gi)
video_gi_m, video_gi_s = agg(acc_video_gi)
audio_sd_m, audio_sd_s = agg(acc_audio_std)
video_sd_m, video_sd_s = agg(acc_video_std)

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
n_videos = len([e for e in entries])
fig.suptitle(
    f"Layer Depth Experiment — Text→Audio/Video Relevance\n"
    f"({len(entries)} videos, ≥{args.min_questions} questions each)",
    fontsize=13, y=1.01,
)

# Panel 1: Cross-Q Spearman
ax = axes[0]
ax.plot(layers, audio_sp_m, "o-", color="#27ae60", lw=2, label="Audio")
ax.fill_between(layers, audio_sp_m - audio_sp_s, audio_sp_m + audio_sp_s,
                alpha=0.18, color="#27ae60")
ax.plot(layers, video_sp_m, "s--", color="#2980b9", lw=2, label="Video")
ax.fill_between(layers, video_sp_m - video_sp_s, video_sp_m + video_sp_s,
                alpha=0.18, color="#2980b9")
ax.axhline(0, color="gray", lw=0.6, ls=":")
ax.set_ylabel("Cross-Q Spearman ρ", fontsize=11)
ax.set_title(
    "Question-Sensitivity  (lower ρ = more question-relevant)\n"
    "Mean pairwise Spearman between score vectors of different Qs on same video",
    fontsize=10,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_ylim(-0.15, 1.05)

# Panel 2: Gini
ax = axes[1]
ax.plot(layers, audio_gi_m, "o-", color="#27ae60", lw=2, label="Audio")
ax.fill_between(layers, audio_gi_m - audio_gi_s, audio_gi_m + audio_gi_s,
                alpha=0.18, color="#27ae60")
ax.plot(layers, video_gi_m, "s--", color="#2980b9", lw=2, label="Video")
ax.fill_between(layers, video_gi_m - video_gi_s, video_gi_m + video_gi_s,
                alpha=0.18, color="#2980b9")
ax.set_ylabel("Gini coefficient", fontsize=11)
ax.set_title(
    "Score Concentration  (higher = fewer tokens dominate → better for pruning)\n"
    "Gini of |score| distribution averaged across questions",
    fontsize=10,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 3: Spread
ax = axes[2]
ax.plot(layers, audio_sd_m, "o-", color="#27ae60", lw=2, label="Audio")
ax.fill_between(layers, audio_sd_m - audio_sd_s, audio_sd_m + audio_sd_s,
                alpha=0.18, color="#27ae60")
ax.plot(layers, video_sd_m, "s--", color="#2980b9", lw=2, label="Video")
ax.fill_between(layers, video_sd_m - video_sd_s, video_sd_m + video_sd_s,
                alpha=0.18, color="#2980b9")
ax.set_xlabel("Thinker layer", fontsize=11)
ax.set_ylabel("Mean score std", fontsize=11)
ax.set_title(
    "Score Spread  (higher = more dynamic range for ranking)\n"
    "Mean std of score vectors across questions",
    fontsize=10,
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.xticks(layers)
plt.tight_layout()
out_summary = os.path.join(args.out_dir, "layer_depth_summary.png")
plt.savefig(out_summary, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out_summary}")

# ── Print aggregate table ─────────────────────────────────────────────────────
print("\n── Aggregate Summary ──")
print(f"{'Layer':>6}  {'AudSpear':>13}  {'VidSpear':>13}  "
      f"{'AudGini':>12}  {'VidGini':>12}  {'AudStd':>12}  {'VidStd':>12}")
for i, li in enumerate(HOOK_LAYERS):
    def _fmt(m, s):
        return f"{m:.4f}±{s:.4f}" if not np.isnan(m) else "    N/A    "
    print(f"{li:>6}  "
          f"{_fmt(audio_sp_m[i], audio_sp_s[i]):>13}  "
          f"{_fmt(video_sp_m[i], video_sp_s[i]):>13}  "
          f"{_fmt(audio_gi_m[i], audio_gi_s[i]):>12}  "
          f"{_fmt(video_gi_m[i], video_gi_s[i]):>12}  "
          f"{_fmt(audio_sd_m[i], audio_sd_s[i]):>12}  "
          f"{_fmt(video_sd_m[i], video_sd_s[i]):>12}")

print(f"\nLog  → {LOG_PATH}")
print(f"Plot → {out_summary}")
