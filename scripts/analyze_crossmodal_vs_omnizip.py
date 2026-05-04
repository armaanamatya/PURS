"""
AUC analysis: do LLM cross-modal scores predict OmniZip's pruning mask?

OmniZip prunes audio/video tokens using audio-encoder self-attention (pre-LLM).
This script compares that mask against the LLM-internal cross-modal scores
saved by viz_crossmodal_spearman.py, computing ROC-AUC for each direction
at each saved layer.

AUC interpretation:
  ≈ 0.5  → signal is orthogonal to OmniZip (different pruning behavior)
  ≈ 0.7  → partial agreement
  ≈ 0.9  → signal largely replicates what OmniZip already does

For audio tokens:  text→audio and video→audio scores predict audio mask
For video tokens:  text→video and audio→video scores predict video mask

Outputs:
  {out_dir}/auc_vs_omnizip.jsonl    — one JSON line per video
  {out_dir}/auc_vs_omnizip.png      — AUC bar chart per direction per saved layer

Run:
  python analyze_crossmodal_vs_omnizip.py
  python analyze_crossmodal_vs_omnizip.py --crossmodal_log vizzing/crossmodal_spearman/crossmodal_stats.jsonl
  python analyze_crossmodal_vs_omnizip.py --limit 50
"""

import argparse
import glob as _glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMNIZIP_DIR = os.path.join(REPO_ROOT, "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

# Must use OmniZip's model fork — it is the one that calls omnizip_units.omnizip()
# inside its thinker forward. The transformers version does not.
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
import omnizip.modeling_qwen2_5_omni as _omnizip_mod
_omnizip_mod.check_torch_load_is_safe = lambda: None
import omnizip.omnizip_units as _omnizip_units
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--crossmodal_log", default=os.path.join(
    REPO_ROOT, "vizzing", "crossmodal_spearman", "crossmodal_stats.jsonl"))
parser.add_argument("--model_path", default="/data/armaan/models/Qwen2.5-Omni-7B")
parser.add_argument("--videos",     default=os.path.join(REPO_ROOT, "videos"))
parser.add_argument("--out_dir",    default=os.path.join(REPO_ROOT, "vizzing", "auc_vs_omnizip"))
parser.add_argument("--gpus",       default="0,1")
parser.add_argument("--limit",      type=int, default=None, help="Max videos to process")
# OmniZip pruning config
parser.add_argument("--rho_audio",        type=float, default=0.3)
parser.add_argument("--rho_video",        type=float, default=0.6)
parser.add_argument("--contextual_ratio", type=float, default=0.05)
parser.add_argument("--g",                type=int,   default=3)
parser.add_argument("--fps",                  type=float, default=2.0)
parser.add_argument("--max_frames_videomme",  type=int,   default=768)
parser.add_argument("--max_frames_other",     type=int,   default=128)
parser.add_argument("--max_pixels",           type=int,   default=360 * 420)
args = parser.parse_args()

GPU_IDS      = [int(g) for g in args.gpus.split(",")]
SAVED_LAYERS = ["6", "14"]

AUDIO_SCORE_DIRS = ["text→audio", "video→audio"]
VIDEO_SCORE_DIRS = ["text→video", "audio→video"]

os.makedirs(args.out_dir, exist_ok=True)
LOG_PATH = os.path.join(args.out_dir, "auc_vs_omnizip.jsonl")

# Keys passed to model.generate() — multimodal inputs the generate path needs
GENERATE_KEYS = {
    "input_ids", "attention_mask",
    "pixel_values", "pixel_values_videos",
    "image_grid_thw", "video_grid_thw",
    "input_features", "feature_attention_mask",
    "audio_feature_lengths", "video_second_per_grid",
    "rope_deltas", "position_ids",
}


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
    for match in _glob.glob(os.path.join(videos_dir, "**", filename), recursive=True):
        if match.replace("\\", "/").endswith(rel_norm):
            return match
    matches = _glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
    return matches[0] if len(matches) == 1 else None


# ── ROC-AUC (Wilcoxon-Mann-Whitney, no sklearn) ───────────────────────────────
def roc_auc_manual(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    neg_sorted = np.sort(neg)
    wins = np.searchsorted(neg_sorted, pos, side="left")
    ties = np.searchsorted(neg_sorted, pos, side="right") - wins
    return float((wins.sum() + 0.5 * ties.sum()) / (len(pos) * len(neg)))


# ── OmniZip mask capture ──────────────────────────────────────────────────────
def install_omnizip_mask_capture():
    """
    Wraps omnizip_units.omnizip to intercept global_mask at runtime.
    Works because omnizip.modeling_qwen2_5_omni imports omnizip_units as a
    module reference (not a direct function import), so patching the module
    attribute affects all subsequent calls from within the model.
    """
    _capture = {}
    _orig    = _omnizip_units.omnizip

    def _wrapped(input_embeds, attn_logits, input_ids, audio_token_id,
                 video_token_id, num_input_frames, **kw):
        out_embeds, global_mask = _orig(
            input_embeds=input_embeds,
            attn_logits=attn_logits,
            input_ids=input_ids,
            audio_token_id=audio_token_id,
            video_token_id=video_token_id,
            num_input_frames=num_input_frames,
            **kw,
        )
        _capture["global_mask"]    = global_mask.detach().cpu()
        _capture["input_ids"]      = input_ids.detach().cpu()
        _capture["audio_token_id"] = int(audio_token_id)
        _capture["video_token_id"] = int(video_token_id)
        return out_embeds, global_mask

    _omnizip_units.omnizip = _wrapped
    return _capture, _orig


# ── Load model ────────────────────────────────────────────────────────────────
max_memory = {i: "10GiB" for i in GPU_IDS}
print(f"Loading Qwen2.5-Omni-7B (OmniZip fork) across GPUs {GPU_IDS} …")
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
    try:
        has_meta = any(p.is_meta for p in model.token2wav.parameters())
        if not has_meta:
            model.token2wav = model.token2wav.to("cpu")
    except Exception:
        pass
# Enable OmniZip pruning inside the thinker forward
model.thinker.omnizip_config = {
    "rho_audio":        args.rho_audio,
    "rho_video":        args.rho_video,
    "g":                args.g,
    "contextual_ratio": args.contextual_ratio,
}
processor      = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id
print(f"Model ready. Input device: {INPUT_DEVICE}\n")

# Install the capture hook (must be before any generate call)
capture, _orig_omnizip = install_omnizip_mask_capture()

# ── Load crossmodal log ───────────────────────────────────────────────────────
records = []
with open(args.crossmodal_log) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
if args.limit:
    records = records[:args.limit]
print(f"Loaded {len(records)} records from {args.crossmodal_log}\n")


# ── Get OmniZip mask for one video + question ─────────────────────────────────
def get_omnizip_mask(video_path: str, question_text: str, dataset: str = ""):
    """
    Runs model.generate() with OmniZip enabled (via omnizip_config).
    Returns (audio_mask, video_mask) as bool arrays, or (None, None) on failure.
    audio_mask[i]=True  → audio token i is KEPT by OmniZip
    video_mask[i]=True  → video token i is KEPT by OmniZip
    """
    ds_canon   = dataset.strip().lower().replace("_", "-").replace(" ", "-")
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

    capture.clear()
    try:
        gen_inputs = {k: v for k, v in inputs.items() if k in GENERATE_KEYS}
        with torch.no_grad():
            model.generate(
                **gen_inputs,
                use_audio_in_video=True,
                return_audio=False,
                thinker_max_new_tokens=1,
                thinker_do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
    finally:
        torch.cuda.empty_cache()

    if "global_mask" not in capture:
        return None, None

    global_mask = capture["global_mask"].reshape(-1).numpy().astype(bool)
    ids_np      = capture["input_ids"].reshape(-1).numpy()

    audio_pos = np.flatnonzero(ids_np == audio_token_id)
    video_pos = np.flatnonzero(ids_np == video_token_id)

    audio_mask = global_mask[audio_pos].astype(bool) if len(audio_pos) > 0 else None
    video_mask = global_mask[video_pos].astype(bool) if len(video_pos) > 0 else None
    return audio_mask, video_mask


# ── AUC accumulators ──────────────────────────────────────────────────────────
auc_acc  = {li: {d: [] for d in AUDIO_SCORE_DIRS + VIDEO_SCORE_DIRS} for li in SAVED_LAYERS}
n_processed = 0

try:
    with open(LOG_PATH, "a") as log_f:
        for rec in records:
            video_file = rec["file"]
            raw_scores = rec.get("raw_scores", {})
            if not raw_scores:
                print(f"SKIP (no raw scores): {video_file}")
                continue

            video_path = resolve_video_path(video_file, args.videos)
            if video_path is None:
                print(f"SKIP (no video): {video_file}")
                continue

            dataset         = rec.get("dataset", "")
            questions_saved = rec.get("questions", [])
            q_text          = questions_saved[0] if questions_saved else "Describe the video."

            print(f"[{n_processed+1}] {rec.get('video_id','?')}  —  getting OmniZip mask …")
            try:
                audio_mask, video_mask = get_omnizip_mask(video_path, q_text, dataset)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue

            if audio_mask is None and video_mask is None:
                print("  SKIP (OmniZip did not fire)")
                continue

            v_auc = {}

            for li_str in SAVED_LAYERS:
                layer_raw = raw_scores.get(li_str, {})
                v_auc[li_str] = {}

                if audio_mask is not None:
                    for dir_label in AUDIO_SCORE_DIRS:
                        q_vecs = layer_raw.get(dir_label, [])
                        if not q_vecs:
                            continue
                        scores = np.mean(q_vecs, axis=0)
                        if len(scores) != len(audio_mask):
                            print(f"  WARN: {dir_label}@L{li_str}: score len {len(scores)} ≠ mask {len(audio_mask)}")
                            continue
                        auc = roc_auc_manual(scores, audio_mask.astype(float))
                        v_auc[li_str][dir_label] = auc
                        auc_acc[li_str][dir_label].append(auc)
                        print(f"  L{li_str}  {dir_label}: AUC={auc:.4f}  (kept={audio_mask.mean():.1%})")

                if video_mask is not None:
                    for dir_label in VIDEO_SCORE_DIRS:
                        q_vecs = layer_raw.get(dir_label, [])
                        if not q_vecs:
                            continue
                        scores = np.mean(q_vecs, axis=0)
                        if len(scores) != len(video_mask):
                            print(f"  WARN: {dir_label}@L{li_str}: score len {len(scores)} ≠ mask {len(video_mask)}")
                            continue
                        auc = roc_auc_manual(scores, video_mask.astype(float))
                        v_auc[li_str][dir_label] = auc
                        auc_acc[li_str][dir_label].append(auc)
                        print(f"  L{li_str}  {dir_label}: AUC={auc:.4f}  (kept={video_mask.mean():.1%})")

            record = {
                "video_id":   rec.get("video_id"),
                "dataset":    rec.get("dataset"),
                "file":       video_file,
                "audio_kept": float(audio_mask.mean()) if audio_mask is not None else None,
                "video_kept": float(video_mask.mean()) if video_mask is not None else None,
                "auc":        v_auc,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            n_processed += 1

finally:
    _omnizip_units.omnizip = _orig_omnizip  # always restore

print(f"\nProcessed {n_processed} videos.")

# ── Aggregate plot ────────────────────────────────────────────────────────────
all_dirs   = AUDIO_SCORE_DIRS + VIDEO_SCORE_DIRS
dir_colors = {"text→audio": "#27ae60", "video→audio": "#8e44ad",
              "text→video":  "#2980b9", "audio→video": "#e67e22"}

fig, axes = plt.subplots(1, len(SAVED_LAYERS), figsize=(5 * len(SAVED_LAYERS), 5), sharey=True)
if len(SAVED_LAYERS) == 1:
    axes = [axes]

for ax, li_str in zip(axes, SAVED_LAYERS):
    x      = np.arange(len(all_dirs))
    means  = [float(np.mean(auc_acc[li_str][d])) if auc_acc[li_str][d] else np.nan for d in all_dirs]
    stds   = [float(np.std(auc_acc[li_str][d]))  if auc_acc[li_str][d] else 0.0    for d in all_dirs]
    colors = [dir_colors[d] for d in all_dirs]
    bars   = ax.bar(x, means, yerr=stds, color=colors, alpha=0.82, capsize=5,
                    error_kw={"elinewidth": 1.5})
    ax.axhline(0.5, color="gray", lw=1.0, ls="--", alpha=0.7, label="AUC=0.5 (random)")
    ax.axhline(0.9, color="red",  lw=1.0, ls=":",  alpha=0.6, label="AUC=0.9")
    ax.set_xticks(x)
    ax.set_xticklabels(all_dirs, rotation=20, ha="right", fontsize=9)
    ax.set_title(f"Layer {li_str}", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

axes[0].set_ylabel("ROC-AUC vs. OmniZip mask", fontsize=12)
axes[0].legend(fontsize=8)
fig.suptitle(
    "AUC: LLM Cross-Modal Scores vs. OmniZip Pruning Mask\n"
    "0.5=orthogonal (different signal)  0.9=replicates OmniZip  1.0=identical",
    fontsize=11, y=1.02,
)
fig.text(0.99, 0.01, f"n={n_processed} videos", ha="right", fontsize=9, color="gray")
plt.tight_layout()
out_fig = os.path.join(args.out_dir, "auc_vs_omnizip.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out_fig}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── ROC-AUC vs. OmniZip mask  (mean ± std) ──")
print(f"{'Direction':<14}  " + "  ".join(f"L{li_str:>2}" for li_str in SAVED_LAYERS))
for d in all_dirs:
    row = f"{d:<14}  "
    for li_str in SAVED_LAYERS:
        vals = auc_acc[li_str][d]
        row += f"  {np.mean(vals):.4f}±{np.std(vals):.4f}" if vals else "      N/A      "
    print(row)

print(f"\nLog  → {LOG_PATH}")
print(f"Plot → {out_fig}")
