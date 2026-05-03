"""
Attention visualization for normal Qwen2.5-Omni (no token compression).

Shows, for a single video+audio input:
  1. Fraction of attention going to audio vs video vs text tokens (per layer)
  2. Per-audio-token attention across the sequence (last layer)
  3. Per-video-frame attention (last layer)

Run from: C:/Users/armaa/OneDrive/Desktop/PURS/
  python viz_attention_qwen.py
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup so we can import qwen_omni_utils
OMNIZIP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

# Patch after import: the modeling module holds its own reference via `from ... import`
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None  # bypass CVE-2025-32434 (torch < 2.6)
from qwen_omni_utils import process_mm_info

# ── Config ────────────────────────────────────────────────────────────────────
# Shortest video in the eval set (24s, 1 question)
VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos", "worldsense",
                          "attribute_reasoning", "video.mp4")
QUESTION   = "What is the profession of the man with a beard wearing a suit in the video?"
MODEL_PATH = "/workspace/model"

# Which decoder layers to visualise (0-indexed out of 28)
VIZ_LAYERS = [0, 13, 27]   # first, middle, last
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention_viz_qwen")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model with EAGER attention (required to extract attn weights) ────────
print("Loading Qwen2.5-Omni-7B with eager attention …")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    attn_implementation="eager",   # flash_attention_2 does NOT return weights
)
model.eval()
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
# Keep fp16 for non-integer tensors
for k, v in inputs.items():
    if v.is_floating_point():
        inputs[k] = v.to(torch.float16)

input_ids        = inputs["input_ids"]   # (1, seq_len)
audio_token_id   = model.thinker.config.audio_token_id
video_token_id   = model.thinker.config.video_token_id
seq_len          = input_ids.shape[1]

n_audio = (input_ids[0] == audio_token_id).sum().item()
n_video = (input_ids[0] == video_token_id).sum().item()
n_text  = seq_len - n_audio - n_video
print(f"Sequence length : {seq_len}")
print(f"  Audio tokens  : {n_audio}")
print(f"  Video tokens  : {n_video}")
print(f"  Text tokens   : {n_text}")

# Keys accepted by the thinker's forward()
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
print("Running prefill forward pass (eager attn – may take ~1 min) …")
with torch.no_grad():
    output = model.thinker(
        **thinker_inputs,
        output_attentions=True,
        use_audio_in_video=True,
        use_cache=False,
        return_dict=True,
    )

# output.attentions: tuple[28] of (1, heads, seq, seq) in fp16
# Move each to CPU in fp32 immediately to free VRAM
attentions = [a.float().cpu() for a in output.attentions]
print(f"Captured {len(attentions)} layers, shape per layer: {attentions[0].shape}")

# ── Identify per-position modality masks ─────────────────────────────────────
ids        = input_ids[0].cpu()
audio_pos  = (ids == audio_token_id).nonzero(as_tuple=True)[0]   # absolute indices
video_pos  = (ids == video_token_id).nonzero(as_tuple=True)[0]
text_mask  = (ids != audio_token_id) & (ids != video_token_id)
text_pos   = text_mask.nonzero(as_tuple=True)[0]

# The last token of the prefill is where generation would start from
last_q = seq_len - 1


def modality_fractions(attn_layer, query_idx):
    """Average attention from query_idx across all heads, return modality fractions."""
    row    = attn_layer[0, :, query_idx, :].mean(0)   # (seq,)
    a_sum  = row[audio_pos].sum().item() if len(audio_pos) > 0 else 0.0
    v_sum  = row[video_pos].sum().item() if len(video_pos) > 0 else 0.0
    t_sum  = row[text_pos].sum().item()
    total  = a_sum + v_sum + t_sum + 1e-9
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
    f"Normal Qwen2.5-Omni | Attention → modality (last prompt token)\n"
    f"Video: {os.path.basename(os.path.dirname(VIDEO_PATH))}/video.mp4  |  "
    f"audio={n_audio} vid={n_video} txt={n_text} tok",
    fontsize=10, y=1.02,
)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "1_modality_fractions.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out1}")


# ── Plot 2 : Per-audio-token attention (last layer) ───────────────────────────
if len(audio_pos) > 0:
    li  = 27
    row = attentions[li][0, :, last_q, :].mean(0)
    audio_attn = row[audio_pos].numpy()

    fig, ax = plt.subplots(figsize=(min(14, max(8, len(audio_attn) // 15)), 3))
    x = np.arange(len(audio_attn))
    ax.fill_between(x, audio_attn, alpha=0.35, color="#e74c3c")
    ax.plot(x, audio_attn, color="#c0392b", linewidth=0.9, label="attention weight")

    # Annotate top-5 peaks
    top5 = np.argsort(audio_attn)[-5:]
    for idx in top5:
        ax.annotate(f"{idx}", (x[idx], audio_attn[idx]),
                    textcoords="offset points", xytext=(0, 5),
                    fontsize=7, ha="center", color="#c0392b")

    ax.set_xlabel("Audio token index  (≈ time →)", fontsize=10)
    ax.set_ylabel("Attention weight", fontsize=10)
    ax.set_title(f"Normal Qwen | Layer {li} | Attention over {len(audio_attn)} audio tokens", fontsize=11)
    ax.set_xlim(0, len(audio_attn) - 1)
    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, "2_audio_token_attention.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out2}")


# ── Plot 3 : Per-video-frame attention (last layer) ───────────────────────────
if len(video_pos) > 0:
    li  = 27
    row = attentions[li][0, :, last_q, :].mean(0)
    video_attn = row[video_pos].numpy()

    tpf = len(video_pos) // num_input_frames  # tokens per frame
    frame_attn = np.array([
        video_attn[f * tpf: (f + 1) * tpf].sum()
        for f in range(num_input_frames)
    ])

    fig, ax = plt.subplots(figsize=(max(8, num_input_frames // 2 + 2), 3))
    bar_colors = plt.cm.Blues(
        0.4 + 0.5 * (frame_attn - frame_attn.min()) / (frame_attn.max() - frame_attn.min() + 1e-9)
    )
    bars = ax.bar(np.arange(num_input_frames), frame_attn,
                  color=bar_colors, edgecolor="black", linewidth=0.5)
    top_frame = int(np.argmax(frame_attn))
    bars[top_frame].set_edgecolor("red")
    bars[top_frame].set_linewidth(2)

    ax.set_xlabel("Frame index", fontsize=10)
    ax.set_ylabel("Σ attention weight", fontsize=10)
    ax.set_title(f"Normal Qwen | Layer {li} | Attention per video frame  "
                 f"(most attended: frame {top_frame})", fontsize=11)
    ax.set_xticks(np.arange(num_input_frames))
    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "3_video_frame_attention.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out3}")


# ── Plot 4 : Attention heatmap across layers (audio bucket) ───────────────────
if len(audio_pos) > 0:
    # (num_layers, seq_len) – mean attention from last_q to every token, per layer
    all_layers_audio = np.array([
        attentions[li][0, :, last_q, audio_pos].mean(0).mean().item()
        for li in range(len(attentions))
    ])
    all_layers_video = np.array([
        attentions[li][0, :, last_q, video_pos].mean(0).mean().item()
        for li in range(len(attentions))
    ])
    all_layers_text = np.array([
        attentions[li][0, :, last_q, text_pos].mean(0).mean().item()
        for li in range(len(attentions))
    ])

    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(attentions))
    ax.plot(x, all_layers_audio, "o-", color="#e74c3c", label="Audio", markersize=4)
    ax.plot(x, all_layers_video, "s-", color="#3498db", label="Video", markersize=4)
    ax.plot(x, all_layers_text,  "^-", color="#2ecc71", label="Text",  markersize=4)
    ax.set_xlabel("Layer index", fontsize=10)
    ax.set_ylabel("Mean attention weight per token", fontsize=10)
    ax.set_title("Normal Qwen | Mean per-token attention to each modality across all layers", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout()
    out4 = os.path.join(OUT_DIR, "4_modality_across_layers.png")
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out4}")


print(f"\nAll visualizations saved to: {OUT_DIR}/")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
