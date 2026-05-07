"""
Cross-modal attention depth curve for Qwen2.5-Omni Thinker.
(Moved into mechinterp/ on 2026-05-03 — was previously /data/armaan/purs/viz_attention_depth_curve.py.)

Plots how attention between modalities evolves CONTINUOUSLY across ALL 28
decoder layers — not just sampled at [0, 13, 27].

Produces, in --out_dir (default mechinterp/outputs/depth_curve/):
  1. Cross-modal attention flow: 6 directional curves
     (audio→video, audio→text, video→audio, video→text, text→audio, text→video)
  2. Self-attention vs cross-attention ratio per layer
  3. Attention concentration (Gini coefficient) per modality across depth
  4. Modality attention share heatmap (layers x modality-pairs)
  5. Last-token attention share across depth (generation-relevant)

Run from anywhere; defaults assume the repo root is mechinterp/'s parent
(i.e. this file lives at <repo_root>/mechinterp/viz_attention_depth_curve.py).
  python mechinterp/viz_attention_depth_curve.py
"""

import argparse
import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Path setup. Script now lives one level below the repo root, so OmniZip-main
# and videos/ are siblings of mechinterp/.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
OMNIZIP_DIR = os.path.join(REPO_ROOT, "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None
from qwen_omni_utils import process_mm_info

# ── Config ───────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(REPO_ROOT, "videos", "worldsense",
                          "attribute_reasoning", "video.mp4")
QUESTION   = "What is the profession of the man with a beard wearing a suit in the video?"
MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
OUT_DIR    = os.path.join(SCRIPT_DIR, "outputs", "depth_curve")

parser = argparse.ArgumentParser(description="Visualize Qwen2.5-Omni Thinker cross-modal attention depth for one video/question.")
parser.add_argument("--video_path", default=VIDEO_PATH)
parser.add_argument("--question", default=QUESTION)
parser.add_argument("--model_path", default=MODEL_PATH)
parser.add_argument("--out_dir", default=OUT_DIR)
parser.add_argument("--fps", type=float, default=None, help="Optional video sampling FPS cap.")
parser.add_argument("--max_pixels", type=int, default=None, help="Optional max pixels per sampled frame.")
parser.add_argument("--max_frames", type=int, default=None, help="Optional max sampled frames.")
args = parser.parse_args()

VIDEO_PATH = args.video_path
QUESTION = args.question
MODEL_PATH = args.model_path
OUT_DIR = args.out_dir

os.makedirs(OUT_DIR, exist_ok=True)

# Drop a run_meta.json so the folder is self-describing.
import json as _json, datetime as _dt
_meta = {
    "video_path": VIDEO_PATH,
    "question":   QUESTION,
    "model_path": MODEL_PATH,
    "fps":        args.fps,
    "max_pixels": args.max_pixels,
    "max_frames": args.max_frames,
    "timestamp":  _dt.datetime.now().isoformat(timespec="seconds"),
}
with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as _f:
    _json.dump(_meta, _f, indent=2)

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading Qwen2.5-Omni-7B with eager attention …")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    attn_implementation="eager",
)
model.eval()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)

# ── Prepare inputs ───────────────────────────────────────────────────────────
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
video_element = {"type": "video", "video": VIDEO_PATH}
if args.fps is not None:
    video_element["fps"] = args.fps
if args.max_pixels is not None:
    video_element["max_pixels"] = args.max_pixels
if args.max_frames is not None:
    video_element["max_frames"] = args.max_frames
conversation[1]["content"][0] = video_element

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

n_audio = (input_ids[0] == audio_token_id).sum().item()
n_video = (input_ids[0] == video_token_id).sum().item()
n_text  = seq_len - n_audio - n_video
print(f"Sequence length : {seq_len}")
print(f"  Audio tokens  : {n_audio}")
print(f"  Video tokens  : {n_video}")
print(f"  Text tokens   : {n_text}")

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

# Build modality masks before the forward pass so hooks can stream compact
# statistics and immediately drop the full attention matrices.
ids       = input_ids[0].cpu()
audio_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
video_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
text_mask = (ids != audio_token_id) & (ids != video_token_id)
text_pos  = text_mask.nonzero(as_tuple=True)[0]

modalities = {
    "audio": audio_pos,
    "video": video_pos,
    "text":  text_pos,
}
modalities_gpu = {name: pos.to("cuda:0") for name, pos in modalities.items()}
mod_colors = {"audio": "#e74c3c", "video": "#3498db", "text": "#2ecc71"}

directions = []
for src_name, src_pos in modalities.items():
    for dst_name, dst_pos in modalities.items():
        directions.append((src_name, dst_name))

layer_data = {d: [] for d in directions}
gini_data = {mod: [] for mod in modalities}
last_token_data = {mod: [] for mod in modalities}
last_q = seq_len - 1


def _mean_attention_block(head_mean, src_pos, dst_pos):
    if src_pos.numel() == 0 or dst_pos.numel() == 0:
        return 0.0
    return head_mean.index_select(0, src_pos).index_select(1, dst_pos).mean().item()


def _gini_for_queries(head_mean, query_pos):
    """Mean per-query Gini over the full key distribution."""
    if query_pos.numel() == 0:
        return 0.0
    rows = head_mean.index_select(0, query_pos).float()
    sorted_rows = torch.sort(rows, dim=-1).values
    n = sorted_rows.shape[-1]
    index = torch.arange(1, n + 1, device=sorted_rows.device, dtype=sorted_rows.dtype).view(1, -1)
    denom = (n * sorted_rows.sum(dim=-1)).clamp_min(1e-12)
    gini = (2 * (index * sorted_rows).sum(dim=-1) / denom) - (n + 1) / n
    return gini.mean().item()


def _make_thinker_attn_stats_hook(layer_idx):
    def hook(module, args, kwargs, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            return output

        with torch.no_grad():
            # output[1]: (batch, heads, query_seq, key_seq)
            head_mean = output[1][0].mean(dim=0)

            for src_name, dst_name in directions:
                layer_data[(src_name, dst_name)].append(
                    _mean_attention_block(head_mean, modalities_gpu[src_name], modalities_gpu[dst_name])
                )

            for mod_name, mod_pos in modalities_gpu.items():
                gini_data[mod_name].append(_gini_for_queries(head_mean, mod_pos))
                if mod_pos.numel() == 0:
                    last_token_data[mod_name].append(0.0)
                else:
                    last_token_data[mod_name].append(head_mean[last_q].index_select(0, mod_pos).sum().item())

        # Keep the full attention tensor out of the outer model's return tuple.
        return (output[0], None) + output[2:]

    return hook


hooks = []
for i, layer in enumerate(model.thinker.model.layers):
    hooks.append(layer.self_attn.register_forward_hook(_make_thinker_attn_stats_hook(i), with_kwargs=True))

# ── Forward pass with ALL layer attentions ───────────────────────────────────
print("Running forward pass (streaming compact stats across all 28 layers) ...")
try:
    with torch.no_grad():
        output = model.thinker(
            **thinker_inputs,
            output_attentions=True,
            use_audio_in_video=True,
            use_cache=False,
            return_dict=True,
        )
finally:
    for h in hooks:
        h.remove()

num_layers = len(next(iter(layer_data.values())))
print(f"Captured compact attention stats for {num_layers} layers")
if num_layers == 0:
    raise RuntimeError("No Thinker attention stats were captured; check hook registration.")

# ── Modality masks ───────────────────────────────────────────────────────────
ids       = input_ids[0].cpu()
audio_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
video_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
text_mask = (ids != audio_token_id) & (ids != video_token_id)
text_pos  = text_mask.nonzero(as_tuple=True)[0]

modalities = {
    "audio": audio_pos,
    "video": video_pos,
    "text":  text_pos,
}
mod_colors = {"audio": "#e74c3c", "video": "#3498db", "text": "#2ecc71"}


# ── Compute cross-modal attention matrix per layer ───────────────────────────
# For each layer, compute mean attention from modality A → modality B
# Result: cross_attn[layer_idx][(src, dst)] = mean attention weight

print("Using streamed cross-modal attention statistics ...")

# Convert to arrays
for k in layer_data:
    layer_data[k] = np.array(layer_data[k])
for k in gini_data:
    gini_data[k] = np.array(gini_data[k])
for k in last_token_data:
    last_token_data[k] = np.array(last_token_data[k])


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: Cross-modal attention flow (6 directional curves)
# ══════════════════════════════════════════════════════════════════════════════
cross_directions = [(s, d) for s, d in directions if s != d]

fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(num_layers)

line_styles = {
    ("audio", "video"): ("o-",  "#e74c3c", "audio → video"),
    ("audio", "text"):  ("o--", "#c0392b", "audio → text"),
    ("video", "audio"): ("s-",  "#3498db", "video → audio"),
    ("video", "text"):  ("s--", "#2980b9", "video → text"),
    ("text",  "audio"): ("^-",  "#2ecc71", "text → audio"),
    ("text",  "video"): ("^--", "#27ae60", "text → video"),
}

for d in cross_directions:
    style, color, label = line_styles[d]
    ax.plot(x, layer_data[d], style, color=color, label=label,
            markersize=4, linewidth=1.5, alpha=0.85)

ax.set_xlabel("Decoder Layer", fontsize=11)
ax.set_ylabel("Mean Attention Weight (query→key)", fontsize=11)
ax.set_title("Cross-Modal Attention Flow Across All 28 Thinker Layers\n"
             "How each modality attends to other modalities at every depth",
             fontsize=12)
ax.legend(fontsize=8, ncol=2, loc="upper right")
ax.set_xticks(x)
ax.tick_params(axis="x", labelsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "1_crossmodal_attention_curves.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out1}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: Self-attention vs cross-attention ratio
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, mod_name in zip(axes, ["audio", "video", "text"]):
    self_attn = layer_data[(mod_name, mod_name)]
    cross_total = sum(
        layer_data[(mod_name, other)]
        for other in modalities if other != mod_name
    )
    total = self_attn + cross_total + 1e-12

    ax.fill_between(x, 0, self_attn / total, alpha=0.4, color=mod_colors[mod_name],
                    label="Self-attention")
    ax.fill_between(x, self_attn / total, 1.0, alpha=0.25, color="gray",
                    label="Cross-attention")
    ax.plot(x, self_attn / total, color=mod_colors[mod_name], linewidth=1.5)

    ax.set_xlabel("Layer", fontsize=10)
    ax.set_title(f"{mod_name.capitalize()} tokens", fontsize=11)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    if mod_name == "audio":
        ax.set_ylabel("Fraction", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

fig.suptitle("Self-Attention vs Cross-Attention Ratio Across Depth\n"
             "How much each modality talks to itself vs others at each layer",
             fontsize=11, y=1.04)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "2_self_vs_cross_attention.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Attention Gini coefficient (concentration) per modality
# ══════════════════════════════════════════════════════════════════════════════
print("Using streamed attention Gini coefficients ...")

fig, ax = plt.subplots(figsize=(14, 4))
for mod_name in ["audio", "video", "text"]:
    marker = {"audio": "o", "video": "s", "text": "^"}[mod_name]
    ax.plot(x, gini_data[mod_name], f"{marker}-", color=mod_colors[mod_name],
            label=f"{mod_name.capitalize()}", markersize=4, linewidth=1.5)

ax.set_xlabel("Decoder Layer", fontsize=11)
ax.set_ylabel("Gini Coefficient", fontsize=11)
ax.set_title("Attention Concentration (Gini) Across Depth\n"
             "Higher = attention focused on fewer keys  |  Lower = attention spread widely",
             fontsize=11)
ax.legend(fontsize=9)
ax.set_xticks(x)
ax.tick_params(axis="x", labelsize=7)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "3_attention_gini_depth.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out3}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: Modality attention share heatmap (layers x directions)
# ══════════════════════════════════════════════════════════════════════════════
dir_labels = [f"{s}→{d}" for s, d in directions]
heatmap = np.array([layer_data[d] for d in directions]).T  # (layers, 9)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax.set_xticks(range(len(dir_labels)))
ax.set_xticklabels(dir_labels, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(num_layers))
ax.set_yticklabels(range(num_layers), fontsize=7)
ax.set_xlabel("Attention Direction (query → key)", fontsize=11)
ax.set_ylabel("Decoder Layer", fontsize=11)
ax.set_title("Cross-Modal Attention Heatmap\n"
             "Mean attention weight for each modality pair at every layer", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Mean attention weight")

# Annotate max per row
for li in range(num_layers):
    max_col = np.argmax(heatmap[li])
    ax.text(max_col, li, "*", ha="center", va="center", fontsize=8, color="white",
            fontweight="bold")

plt.tight_layout()
out4 = os.path.join(OUT_DIR, "4_crossmodal_heatmap.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out4}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5: Where does the LAST token look? (generation-relevant attention)
# ══════════════════════════════════════════════════════════════════════════════
last_q = seq_len - 1

last_token_audio = np.array(last_token_data["audio"])
last_token_video = np.array(last_token_data["video"])
last_token_text = np.array(last_token_data["text"])
total = last_token_audio + last_token_video + last_token_text + 1e-12

fig, ax = plt.subplots(figsize=(14, 5))

ax.stackplot(x,
             last_token_audio / total,
             last_token_video / total,
             last_token_text / total,
             labels=["Audio", "Video", "Text"],
             colors=["#e74c3c", "#3498db", "#2ecc71"],
             alpha=0.7)

ax.set_xlabel("Decoder Layer", fontsize=11)
ax.set_ylabel("Attention Share", fontsize=11)
ax.set_title("Last Token's Attention Distribution Across Depth\n"
             "Where the generation token looks at each layer (stacked area)",
             fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.set_xticks(x)
ax.tick_params(axis="x", labelsize=7)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(alpha=0.2)

plt.tight_layout()
out5 = os.path.join(OUT_DIR, "5_last_token_attention_depth.png")
plt.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out5}")


print(f"\nAll depth-curve visualizations saved to: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
