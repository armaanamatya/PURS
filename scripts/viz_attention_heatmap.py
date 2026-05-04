"""
Reproduce OmniZip paper Figure 2: full T×T attention heatmaps for Qwen2.5-Omni.

Shows the head-averaged attention matrix at two decoder layers (default: 4 & 20)
with log-scale coloring and a zoomed inset, plus x-axis token-type labels.

Works with the UNCOMPRESSED (normal) model so we see the raw audio-dominance
pattern that motivates OmniZip.

Run from: C:/Users/Armaan/Desktop/PURS/
  python viz_attention_heatmap.py
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ── Path setup ────────────────────────────────────────────────────────────────
OMNIZIP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None
from qwen_omni_utils import process_mm_info

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos", "worldsense",
                          "attribute_reasoning", "video.mp4")
QUESTION   = "What is the profession of the man with a beard wearing a suit in the video?"
MODEL_PATH = "/workspace/model"

VIZ_LAYERS = [3, 19]          # 0-indexed → "Layer 4" and "Layer 20" in the paper
ZOOM_SIZE  = 10               # NxN window for the zoomed inset
OUT_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "attention_viz_heatmap")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model (eager attention required) ─────────────────────────────────────
print("Loading Qwen2.5-Omni-7B with eager attention …")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    attn_implementation="eager",
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
for k, v in inputs.items():
    if v.is_floating_point():
        inputs[k] = v.to(torch.float16)

input_ids      = inputs["input_ids"]
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id
seq_len        = input_ids.shape[1]

ids     = input_ids[0].cpu()
n_audio = (ids == audio_token_id).sum().item()
n_video = (ids == video_token_id).sum().item()
n_text  = seq_len - n_audio - n_video
print(f"Sequence length: {seq_len}  (audio={n_audio}, video={n_video}, text={n_text})")

# ── Forward pass ──────────────────────────────────────────────────────────────
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

print("Running prefill forward pass (eager attn – may take ~1 min) …")
with torch.no_grad():
    output = model.thinker(
        **thinker_inputs,
        output_attentions=True,
        use_audio_in_video=True,
        use_cache=False,
        return_dict=True,
    )

# Only keep the layers we need (save memory)
attn_layers = {}
for li in VIZ_LAYERS:
    attn_layers[li] = output.attentions[li].float().cpu()
del output
torch.cuda.empty_cache()
print(f"Captured layers {VIZ_LAYERS}, shape: {attn_layers[VIZ_LAYERS[0]].shape}")

# ── Build token-type labels for x-axis ────────────────────────────────────────
token_types = []   # 'A', 'V', or 'T' for each position
for i in range(seq_len):
    tok = ids[i].item()
    if tok == audio_token_id:
        token_types.append("A")
    elif tok == video_token_id:
        token_types.append("V")
    else:
        token_types.append("T")

# Build colour strip for x-axis annotation
type_colors = {"A": "#e74c3c", "V": "#3498db", "T": "#2ecc71"}


def find_audio_block_for_zoom(token_types, zoom_size):
    """Find a contiguous audio block to use for the zoomed inset."""
    for i in range(len(token_types) - zoom_size):
        if all(t == "A" for t in token_types[i:i + zoom_size]):
            return i
    # Fallback: first audio token
    for i, t in enumerate(token_types):
        if t == "A":
            return max(0, i - zoom_size // 2)
    return 0


# ── Plot heatmaps ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(VIZ_LAYERS),
                         figsize=(7 * len(VIZ_LAYERS), 6.5))
if len(VIZ_LAYERS) == 1:
    axes = [axes]

for ax, li in zip(axes, VIZ_LAYERS):
    # Head-averaged attention matrix: (seq, seq)
    attn = attn_layers[li][0].mean(dim=0).numpy()   # (seq, seq)

    # Causal mask: lower-triangular (query attends to keys ≤ its position)
    # Values above diagonal are 0 from the causal mask; set to NaN so they
    # don't distort the log scale
    attn_masked = np.where(attn > 0, attn, np.nan)

    # Clamp floor for log scale
    vmin = 1e-4
    vmax = attn_masked[np.isfinite(attn_masked)].max()

    im = ax.imshow(
        attn_masked,
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        origin="upper",
    )

    ax.set_title(f"Qwen2.5-Omni Layer {li + 1}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Key position", fontsize=10)
    ax.set_ylabel("Query position", fontsize=10)

    # ── Zoomed inset ──────────────────────────────────────────────────────
    zoom_start = find_audio_block_for_zoom(token_types, ZOOM_SIZE)
    zoom_end   = zoom_start + ZOOM_SIZE

    axins = inset_axes(ax, width="30%", height="30%", loc="upper right",
                       borderpad=1.5)
    zoom_patch = attn_masked[zoom_start:zoom_end, zoom_start:zoom_end]
    axins.imshow(
        zoom_patch,
        aspect="auto",
        interpolation="nearest",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        origin="upper",
    )
    axins.set_title("Zoomed", fontsize=8)
    axins.tick_params(labelsize=6)
    # Label inset axes with token positions
    axins.set_xticks(range(ZOOM_SIZE))
    axins.set_xticklabels(range(zoom_start, zoom_end), fontsize=5, rotation=90)
    axins.set_yticks(range(ZOOM_SIZE))
    axins.set_yticklabels(range(zoom_start, zoom_end), fontsize=5)

    # Draw red dashed rectangle on main plot showing zoom region
    from matplotlib.patches import Rectangle
    rect = Rectangle((zoom_start - 0.5, zoom_start - 0.5),
                      ZOOM_SIZE, ZOOM_SIZE,
                      linewidth=1.5, edgecolor="red", facecolor="none",
                      linestyle="--")
    ax.add_patch(rect)

    # ── Token-type colour strip along x-axis ──────────────────────────────
    strip_y = seq_len + seq_len * 0.01
    strip_h = seq_len * 0.02
    for i in range(seq_len):
        ax.add_patch(Rectangle(
            (i - 0.5, strip_y), 1, strip_h,
            facecolor=type_colors[token_types[i]],
            edgecolor="none",
            clip_on=False,
        ))

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=9)

# Legend for token types
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label=f"Audio ({n_audio})"),
    Patch(facecolor="#3498db", label=f"Video ({n_video})"),
    Patch(facecolor="#2ecc71", label=f"Text ({n_text})"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Audio tokens dominate attention heatmaps",
    fontsize=14, fontweight="bold", y=1.01,
)
plt.tight_layout()
out = os.path.join(OUT_DIR, "fig2_attention_heatmap.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
plt.close()
print(f"\nSaved → {out}")

print(f"\nAll visualizations saved to: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
