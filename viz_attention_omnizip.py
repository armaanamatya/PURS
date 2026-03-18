"""
Attention visualization for OmniZip-compressed Qwen2.5-Omni.

Shows, for a single video+audio input:
  1. Fraction of attention going to audio vs video vs text tokens (per layer)
  2. Per-audio-token attention – with dropped tokens marked
  3. Per-video-frame attention – with dropped frames marked
  4. Token retention summary (how many audio/video tokens survived)

The key difference from viz_attention_qwen.py:
  - OmniZip prunes/merges audio and video tokens before the LLM sees them.
  - We monkey-patch the omnizip() call to capture the global_mask, then remap
    original token positions → compressed positions for the attention lookup.

Run from: C:/Users/armaa/OneDrive/Desktop/PURS/
  python viz_attention_omnizip.py
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
OMNIZIP_DIR = os.path.join(os.path.dirname(__file__), "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

# Load OmniZip's custom model (not the transformers stock version)
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "videos", "worldsense",
                          "attribute_reasoning", "video.mp4")
QUESTION   = "What is the profession of the man with a beard wearing a suit in the video?"
MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

OMNIZIP_CONFIG = {           # same defaults as demo.py
    "rho_audio":       0.4,  # fraction of audio tokens to DROP
    "rho_video":       0.7,  # fraction of video tokens to DROP
    "g":               3,
    "contextual_ratio": 0.05,
}

VIZ_LAYERS = [0, 13, 27]
OUT_DIR    = os.path.join(os.path.dirname(__file__), "attention_viz_omnizip")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Monkey-patch omnizip() to capture global_mask ─────────────────────────────
import omnizip.omnizip_units as omnizip_units_module
_original_omnizip = omnizip_units_module.omnizip
_captured_mask    = [None]   # will hold (orig_seq_len,) bool tensor after forward

def _patched_omnizip(*args, **kwargs):
    result = _original_omnizip(*args, **kwargs)
    _captured_mask[0] = result[1].cpu()   # result = (embeds, global_mask)
    return result

omnizip_units_module.omnizip = _patched_omnizip

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading OmniZip Qwen2.5-Omni-7B with eager attention …")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    attn_implementation="eager",
)
model.eval()
model.thinker.omnizip_config = OMNIZIP_CONFIG   # enable OmniZip compression

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

# ── Prepare inputs ────────────────────────────────────────────────────────────
conversation = [
    {
        "role": "system",
        "content": [{"type": "text", "text": (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating "
            "text and speech."
        )}],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": VIDEO_PATH},
            {"type": "text",  "text":  QUESTION},
        ],
    },
]

text   = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

num_input_frames = videos[0].shape[0]
model.thinker.nframes = num_input_frames
print(f"Video frames sampled: {num_input_frames}")

inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors="pt", padding=True, use_audio_in_video=True,
)
inputs = inputs.to("cuda:0")
for k, v in inputs.items():
    if v.is_floating_point():
        inputs[k] = v.to(torch.float16)

input_ids      = inputs["input_ids"]   # (1, ORIGINAL seq_len) – before compression
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id
orig_seq_len   = input_ids.shape[1]

ids       = input_ids[0].cpu()
n_audio   = (ids == audio_token_id).sum().item()
n_video   = (ids == video_token_id).sum().item()
n_text    = orig_seq_len - n_audio - n_video
print(f"Original sequence : {orig_seq_len}  (audio={n_audio}, video={n_video}, text={n_text})")

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
thinker_inputs = {k: v for k, v in inputs.items() if k in THINKER_KEYS}

# ── Forward pass ──────────────────────────────────────────────────────────────
print("Running prefill forward pass (OmniZip + eager attn – may take ~1 min) …")
with torch.no_grad():
    output = model.thinker(
        **thinker_inputs,
        output_attentions=True,
        use_audio_in_video=True,
        use_cache=False,
        return_dict=True,
    )

# Restore original omnizip function
omnizip_units_module.omnizip = _original_omnizip

attentions    = [a.float().cpu() for a in output.attentions]
global_mask   = _captured_mask[0]   # (orig_seq_len,) bool – True = token kept
compressed_len = global_mask.sum().item()

print(f"Captured {len(attentions)} layers, compressed shape: {attentions[0].shape}")
print(f"Compressed sequence: {compressed_len}  "
      f"(dropped {orig_seq_len - compressed_len} tokens)")

# ── Build position remapping ──────────────────────────────────────────────────
# orig_to_comp[i] = compressed index for original position i (if kept), else -1
orig_to_comp = torch.full((orig_seq_len,), -1, dtype=torch.long)
comp_idx = 0
for orig_idx in range(orig_seq_len):
    if global_mask[orig_idx].item():
        orig_to_comp[orig_idx] = comp_idx
        comp_idx += 1

audio_orig  = (ids == audio_token_id).nonzero(as_tuple=True)[0]
video_orig  = (ids == video_token_id).nonzero(as_tuple=True)[0]
text_orig   = ((ids != audio_token_id) & (ids != video_token_id)).nonzero(as_tuple=True)[0]

audio_kept  = audio_orig[global_mask[audio_orig]]
audio_drop  = audio_orig[~global_mask[audio_orig]]
audio_comp  = orig_to_comp[audio_kept]   # compressed indices for kept audio tokens

video_kept  = video_orig[global_mask[video_orig]]
video_drop  = video_orig[~global_mask[video_orig]]
video_comp  = orig_to_comp[video_kept]

text_comp   = orig_to_comp[text_orig[global_mask[text_orig]]]

n_audio_kept = len(audio_kept);  n_audio_drop = len(audio_drop)
n_video_kept = len(video_kept);  n_video_drop = len(video_drop)
print(f"Audio: {n_audio_kept} kept / {n_audio_drop} dropped  "
      f"({100*n_audio_drop/max(n_audio,1):.1f}% pruned)")
print(f"Video: {n_video_kept} kept / {n_video_drop} dropped  "
      f"({100*n_video_drop/max(n_video,1):.1f}% pruned)")

# Last query position in COMPRESSED sequence
last_q = compressed_len - 1


def modality_fractions(attn_layer, query_idx):
    row   = attn_layer[0, :, query_idx, :].mean(0)  # (compressed_seq,)
    a_sum = row[audio_comp].sum().item() if len(audio_comp) > 0 else 0.0
    v_sum = row[video_comp].sum().item() if len(video_comp) > 0 else 0.0
    t_sum = row[text_comp].sum().item()  if len(text_comp)  > 0 else 0.0
    total = a_sum + v_sum + t_sum + 1e-9
    return a_sum / total, v_sum / total, t_sum / total


# ── Plot 1 : Modality fractions per layer ────────────────────────────────────
fig, axes = plt.subplots(1, len(VIZ_LAYERS), figsize=(5 * len(VIZ_LAYERS), 4),
                         sharey=True)
cats   = ["Audio", "Video", "Text"]
colors = ["#e74c3c", "#3498db", "#2ecc71"]

for ax, li in zip(axes, VIZ_LAYERS):
    fa, fv, ft = modality_fractions(attentions[li], last_q)
    vals = [fa, fv, ft]
    bars = ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(f"Layer {li}", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel("Fraction of attention" if li == VIZ_LAYERS[0] else "")
    ax.tick_params(axis="x", labelsize=10)

fig.suptitle(
    f"OmniZip Qwen2.5-Omni | Attention → modality (last prompt token)\n"
    f"ρ_audio={OMNIZIP_CONFIG['rho_audio']} ρ_video={OMNIZIP_CONFIG['rho_video']}  |  "
    f"orig={orig_seq_len} → compressed={compressed_len} tok",
    fontsize=10, y=1.02,
)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "1_modality_fractions.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out1}")


# ── Plot 2 : Per-audio-token attention with dropped tokens marked ─────────────
if len(audio_orig) > 0:
    li  = 27
    row = attentions[li][0, :, last_q, :].mean(0)  # (compressed_seq,)

    # Build a full-length array (original audio indices) with NaN for dropped
    audio_full_attn = np.full(n_audio, np.nan)
    for rank, (orig_idx, comp_idx) in enumerate(zip(audio_kept.tolist(),
                                                      audio_comp.tolist())):
        # rank in original audio sequence
        orig_audio_rank = (audio_orig == orig_idx).nonzero(as_tuple=True)[0].item()
        audio_full_attn[orig_audio_rank] = row[comp_idx].item()

    kept_mask = ~np.isnan(audio_full_attn)
    drop_mask = np.isnan(audio_full_attn)

    fig, ax = plt.subplots(figsize=(min(14, max(8, n_audio // 15)), 3.5))
    x = np.arange(n_audio)

    # Draw kept tokens as filled area + line
    kept_vals = np.where(kept_mask, audio_full_attn, 0.0)
    ax.fill_between(x, np.where(kept_mask, audio_full_attn, np.nan),
                    alpha=0.35, color="#e74c3c", label="kept tokens")
    ax.plot(x[kept_mask], audio_full_attn[kept_mask],
            "o", color="#c0392b", markersize=2, linewidth=0)

    # Mark dropped tokens with vertical grey lines
    for xi in x[drop_mask]:
        ax.axvline(xi, color="#bdc3c7", linewidth=0.5, alpha=0.7)
    # Add legend proxy for dropped
    ax.axvline(-999, color="#bdc3c7", linewidth=1.5, label=f"dropped ({drop_mask.sum()})")

    ax.set_xlabel("Audio token index  (≈ time →)", fontsize=10)
    ax.set_ylabel("Attention weight", fontsize=10)
    ax.set_title(
        f"OmniZip | Layer {li} | Attention over audio tokens\n"
        f"{n_audio_kept} kept (blue), {n_audio_drop} dropped (grey lines)",
        fontsize=11,
    )
    ax.set_xlim(0, n_audio - 1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "2_audio_token_attention.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out2}")


# ── Plot 3 : Per-video-frame attention with drop info ────────────────────────
if len(video_orig) > 0:
    li  = 27
    row = attentions[li][0, :, last_q, :].mean(0)

    tpf = n_video // num_input_frames   # tokens per frame (original)

    frame_kept_attn  = np.zeros(num_input_frames)
    frame_drop_count = np.zeros(num_input_frames, dtype=int)

    for orig_idx, comp_idx_val in zip(video_kept.tolist(), video_comp.tolist()):
        # which frame does this original video token belong to?
        rank_in_video = (video_orig == orig_idx).nonzero(as_tuple=True)[0].item()
        frame_id = rank_in_video // tpf
        if frame_id < num_input_frames:
            frame_kept_attn[frame_id] += row[comp_idx_val].item()

    for orig_idx in video_drop.tolist():
        rank_in_video = (video_orig == orig_idx).nonzero(as_tuple=True)[0].item()
        frame_id = rank_in_video // tpf
        if frame_id < num_input_frames:
            frame_drop_count[frame_id] += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, num_input_frames // 2 + 2), 5),
                                    sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    bar_colors = plt.cm.Blues(
        0.4 + 0.5 * (frame_kept_attn - frame_kept_attn.min()) /
        (frame_kept_attn.max() - frame_kept_attn.min() + 1e-9)
    )
    bars = ax1.bar(np.arange(num_input_frames), frame_kept_attn,
                   color=bar_colors, edgecolor="black", linewidth=0.5)
    top_frame = int(np.argmax(frame_kept_attn))
    bars[top_frame].set_edgecolor("red")
    bars[top_frame].set_linewidth(2)
    ax1.set_ylabel("Σ attention weight (kept tokens)", fontsize=9)
    ax1.set_title(
        f"OmniZip | Layer {li} | Attention per video frame\n"
        f"(most attended: frame {top_frame})",
        fontsize=11,
    )

    # Bottom panel: drop count per frame
    ax2.bar(np.arange(num_input_frames), frame_drop_count,
            color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.4)
    ax2.set_ylabel("Dropped\ntokens", fontsize=8)
    ax2.set_xlabel("Frame index", fontsize=10)
    ax2.set_xticks(np.arange(num_input_frames))

    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "3_video_frame_attention.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out3}")


# ── Plot 4 : Modality attention across all layers ────────────────────────────
all_audio = np.array([
    attentions[li][0, :, last_q, audio_comp].mean().item()
    if len(audio_comp) > 0 else 0.0
    for li in range(len(attentions))
])
all_video = np.array([
    attentions[li][0, :, last_q, video_comp].mean().item()
    if len(video_comp) > 0 else 0.0
    for li in range(len(attentions))
])
all_text = np.array([
    attentions[li][0, :, last_q, text_comp].mean().item()
    if len(text_comp) > 0 else 0.0
    for li in range(len(attentions))
])

fig, ax = plt.subplots(figsize=(10, 3))
x = np.arange(len(attentions))
ax.plot(x, all_audio, "o-", color="#e74c3c", label="Audio (kept)", markersize=4)
ax.plot(x, all_video, "s-", color="#3498db", label="Video (kept)", markersize=4)
ax.plot(x, all_text,  "^-", color="#2ecc71", label="Text",         markersize=4)
ax.set_xlabel("Layer index", fontsize=10)
ax.set_ylabel("Mean attention weight per token", fontsize=10)
ax.set_title("OmniZip | Mean per-token attention to each modality across all layers", fontsize=11)
ax.legend(fontsize=9)
ax.set_xticks(x)
ax.tick_params(axis="x", labelsize=7)
plt.tight_layout()
out4 = os.path.join(OUT_DIR, "4_modality_across_layers.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out4}")


# ── Plot 5 : Token retention summary ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))
categories = ["Audio\noriginal", "Audio\nkept", "Video\noriginal", "Video\nkept"]
vals       = [n_audio, n_audio_kept, n_video, n_video_kept]
clrs       = ["#e74c3c", "#c0392b", "#3498db", "#2980b9"]
bars       = ax.bar(categories, vals, color=clrs, edgecolor="black", linewidth=0.6)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Token count", fontsize=10)
ax.set_title(
    f"OmniZip token retention\n"
    f"ρ_audio={OMNIZIP_CONFIG['rho_audio']}, ρ_video={OMNIZIP_CONFIG['rho_video']}",
    fontsize=11,
)
plt.tight_layout()
out5 = os.path.join(OUT_DIR, "5_token_retention.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out5}")


print(f"\nAll visualizations saved to: {OUT_DIR}/")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
