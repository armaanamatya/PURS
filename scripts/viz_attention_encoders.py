"""
Encoder attention visualization for Qwen2.5-Omni.

Visualizes attention patterns INSIDE the Audio Encoder (32 layers) and
Vision Encoder (32 layers) — the components that run *before* the Thinker.

Since neither encoder exposes attention weights through its API, we use
forward hooks on the Q/K projections to compute attention manually.

Produces:
  1. Audio encoder: attention entropy across 32 layers (windowed vs global)
  2. Audio encoder: heatmaps at selected layers (early / mid / late)
  3. Vision encoder: attention entropy across 32 layers (full vs windowed)
  4. Vision encoder: heatmaps at selected layers

Run from: /data/armaan/purs/
  python viz_attention_encoders.py
"""

import argparse
import os, sys, math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
OMNIZIP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration

import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None
from qwen_omni_utils import process_mm_info

# ── Config ───────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "videos", "worldsense",
                          "attribute_reasoning", "video.mp4")
QUESTION   = "What is the profession of the man with a beard wearing a suit in the video?"
MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"

# Layers to show heatmaps for (0-indexed, out of 32)
AUDIO_VIZ_LAYERS  = [0, 15, 31]   # early, mid, late
VISION_VIZ_LAYERS = [0, 15, 31]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vizzing", "encoder_attention")

parser = argparse.ArgumentParser(description="Visualize Qwen2.5-Omni encoder attention for one video/question.")
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

# ── Hook infrastructure ──────────────────────────────────────────────────────
# We capture the attention weights by hooking the attention modules.
# For eager attention, we intercept the module's forward and compute
# attn_weights = softmax(Q @ K^T / sqrt(d)) ourselves from Q, K.

_captured_audio_attn  = {}   # layer_idx -> (seq, seq) mean-over-heads
_captured_vision_attn = {}


def _hook_arg(args, kwargs, index, name, default=None):
    """Read a forward-hook argument from either positional args or kwargs."""
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _make_audio_attn_hook(layer_idx):
    """Hook for Qwen2_5OmniAudioAttention.forward().

    The eager attention computes:
        q = q_proj(x).reshape(seq, num_heads, head_dim)
        k = k_proj(x).reshape(seq, num_heads, head_dim)
        attn = softmax(q @ k^T / sqrt(d))
    We replicate this from the module's projections.
    """
    def hook(module, args, kwargs, output):
        hidden_states = _hook_arg(args, kwargs, 0, "hidden_states")
        cu_seqlens = _hook_arg(args, kwargs, 1, "cu_seqlens")
        if hidden_states is None:
            return
        seq_len = hidden_states.shape[0]
        num_heads = module.num_heads
        head_dim  = module.head_dim

        with torch.no_grad():
            q = module.q_proj(hidden_states).reshape(seq_len, num_heads, head_dim)
            k = module.k_proj(hidden_states).reshape(seq_len, num_heads, head_dim)
            # (num_heads, seq, head_dim) @ (num_heads, head_dim, seq) -> (num_heads, seq, seq)
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Apply windowed masking via cu_seqlens if present.
            if cu_seqlens is not None:
                mask = torch.full((seq_len, seq_len), torch.finfo(attn_weights.dtype).min,
                                 device=attn_weights.device, dtype=attn_weights.dtype)
                for i in range(len(cu_seqlens) - 1):
                    s, e = cu_seqlens[i].item(), cu_seqlens[i+1].item()
                    mask[s:e, s:e] = 0.0
                attn_weights = attn_weights + mask.unsqueeze(0)

            attn_weights = torch.softmax(attn_weights, dim=-1)
            # Mean over heads -> (seq, seq), move to CPU
            _captured_audio_attn[layer_idx] = attn_weights.mean(0).float().cpu()

    return hook


def _make_vision_attn_hook(layer_idx):
    """Hook for Qwen2_5OmniVisionAttention.forward().

    Vision attention applies rotary embeddings to Q and K before computing
    attention. Since RoPE is applied inside the forward, we capture the
    full attention computation by hooking at the module level and
    replicating the Q/K/RoPE/attention path.
    """
    def hook(module, args, kwargs, output):
        hidden_states = _hook_arg(args, kwargs, 0, "hidden_states")
        cu_seqlens = _hook_arg(args, kwargs, 1, "cu_seqlens")
        rotary_pos_emb = _hook_arg(args, kwargs, 2, "rotary_pos_emb")
        if hidden_states is None:
            return
        seq_len = hidden_states.shape[0]
        num_heads = module.num_heads
        head_dim  = hidden_states.shape[-1] // num_heads

        with torch.no_grad():
            q = module.q(hidden_states).reshape(seq_len, num_heads, head_dim)
            k = module.k(hidden_states).reshape(seq_len, num_heads, head_dim)

            if rotary_pos_emb is not None:
                q = _qwen_mod.apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
                k = _qwen_mod.apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

            q = q.permute(1, 0, 2)  # (heads, seq, d)
            k = k.permute(1, 0, 2)
            scale = 1.0 / math.sqrt(head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

            # Windowed masking via cu_seqlens.
            if cu_seqlens is not None:
                mask = torch.full((seq_len, seq_len), torch.finfo(attn_weights.dtype).min,
                                 device=attn_weights.device, dtype=attn_weights.dtype)
                for i in range(len(cu_seqlens) - 1):
                    s, e = cu_seqlens[i].item(), cu_seqlens[i+1].item()
                    mask[s:e, s:e] = 0.0
                attn_weights = attn_weights + mask.unsqueeze(0)

            attn_weights = torch.softmax(attn_weights, dim=-1)
            _captured_vision_attn[layer_idx] = attn_weights.mean(0).float().cpu()

    return hook


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

# ── Register hooks on encoder attention modules ──────────────────────────────
audio_hooks  = []
vision_hooks = []

# Audio encoder layers
audio_tower = model.thinker.audio_tower
for i, layer in enumerate(audio_tower.layers):
    h = layer.self_attn.register_forward_hook(_make_audio_attn_hook(i), with_kwargs=True)
    audio_hooks.append(h)
print(f"Registered {len(audio_hooks)} audio encoder attention hooks")

# Vision encoder blocks
visual = model.thinker.visual
for i, block in enumerate(visual.blocks):
    h = block.attn.register_forward_hook(_make_vision_attn_hook(i), with_kwargs=True)
    vision_hooks.append(h)
print(f"Registered {len(vision_hooks)} vision encoder attention hooks")

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

# ── Forward pass (hooks will capture encoder attention) ─────────────────────
print("Running forward pass (hooks active on encoders) …")
with torch.no_grad():
    output = model.thinker(
        **thinker_inputs,
        output_attentions=False,   # we don't need Thinker attention here
        use_audio_in_video=True,
        use_cache=False,
        return_dict=True,
    )

# Remove hooks
for h in audio_hooks + vision_hooks:
    h.remove()

n_audio_layers  = len(_captured_audio_attn)
n_vision_layers = len(_captured_vision_attn)
print(f"Captured audio encoder attention:  {n_audio_layers} layers")
print(f"Captured vision encoder attention: {n_vision_layers} layers")

if n_audio_layers == 0 and n_vision_layers == 0:
    print("ERROR: No encoder attention captured. Check hook registration paths.")
    sys.exit(1)


# ── Helper: compute attention entropy ────────────────────────────────────────
def attention_entropy(attn_matrix):
    """Mean entropy across all query positions. Higher = more uniform attention."""
    # attn_matrix: (seq, seq), each row sums to 1
    eps = 1e-10
    ent = -(attn_matrix * torch.log(attn_matrix + eps)).sum(dim=-1)  # (seq,)
    return ent.mean().item()


def attention_sparsity(attn_matrix):
    """Fraction of attention concentrated in top-10% of keys (mean over queries)."""
    seq_len = attn_matrix.shape[-1]
    k = max(1, seq_len // 10)
    topk_vals, _ = attn_matrix.topk(k, dim=-1)
    return topk_vals.sum(dim=-1).mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO ENCODER PLOTS
# ══════════════════════════════════════════════════════════════════════════════

if n_audio_layers > 0:
    audio_seq_len = _captured_audio_attn[0].shape[0]
    print(f"\nAudio encoder sequence length: {audio_seq_len}")

    # ── Plot 1: Entropy & sparsity across 32 audio layers ────────────────────
    audio_entropies  = []
    audio_sparsities = []
    for li in range(n_audio_layers):
        attn = _captured_audio_attn[li]
        audio_entropies.append(attention_entropy(attn))
        audio_sparsities.append(attention_sparsity(attn))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    x = np.arange(n_audio_layers)
    ax1.plot(x, audio_entropies, "o-", color="#e74c3c", markersize=4, linewidth=1.5)
    ax1.set_ylabel("Mean Attention Entropy", fontsize=10)
    ax1.set_title("Audio Encoder (32 layers) — Attention Entropy per Layer\n"
                  "(higher = more uniform / diffuse attention)", fontsize=11)
    ax1.grid(alpha=0.3)
    max_ent = math.log(audio_seq_len)
    ax1.axhline(max_ent, color="gray", linestyle="--", alpha=0.5, label=f"max entropy (log {audio_seq_len})")
    ax1.legend(fontsize=8)

    ax2.plot(x, audio_sparsities, "s-", color="#c0392b", markersize=4, linewidth=1.5)
    ax2.set_xlabel("Layer index", fontsize=10)
    ax2.set_ylabel("Top-10% Concentration", fontsize=10)
    ax2.set_title("Attention Sparsity (fraction of attention in top 10% keys)", fontsize=10)
    ax2.set_xticks(x)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, "1_audio_encoder_entropy.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out1}")

    # ── Plot 2: Audio encoder heatmaps at selected layers ────────────────────
    viz_layers = [l for l in AUDIO_VIZ_LAYERS if l < n_audio_layers]
    if viz_layers:
        fig, axes = plt.subplots(1, len(viz_layers),
                                 figsize=(5 * len(viz_layers), 4.5))
        if len(viz_layers) == 1:
            axes = [axes]

        for ax, li in zip(axes, viz_layers):
            attn = _captured_audio_attn[li].numpy()
            # Subsample if too large for visualization
            if attn.shape[0] > 512:
                step = attn.shape[0] // 512
                attn = attn[::step, ::step]
            im = ax.imshow(np.log10(attn + 1e-8), aspect="auto", cmap="inferno",
                           vmin=-5, vmax=0)
            ax.set_title(f"Audio Layer {li}", fontsize=11)
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log₁₀(attn)")

        fig.suptitle("Audio Encoder — Attention Heatmaps (mean over heads)\n"
                     "Block-diagonal = windowed attention pattern",
                     fontsize=11, y=1.02)
        plt.tight_layout()
        out2 = os.path.join(OUT_DIR, "2_audio_encoder_heatmaps.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out2}")


# ══════════════════════════════════════════════════════════════════════════════
# VISION ENCODER PLOTS
# ══════════════════════════════════════════════════════════════════════════════

if n_vision_layers > 0:
    vision_seq_len = _captured_vision_attn[0].shape[0]
    print(f"\nVision encoder sequence length: {vision_seq_len}")

    # ── Plot 3: Entropy & sparsity across 32 vision layers ───────────────────
    vision_entropies  = []
    vision_sparsities = []
    for li in range(n_vision_layers):
        attn = _captured_vision_attn[li]
        vision_entropies.append(attention_entropy(attn))
        vision_sparsities.append(attention_sparsity(attn))

    # Mark full-attention layers
    fullatt_idxs = set()
    if hasattr(visual, 'fullatt_block_indexes'):
        fullatt_idxs = set(visual.fullatt_block_indexes)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    x = np.arange(n_vision_layers)
    colors_ent = ["#8e44ad" if i in fullatt_idxs else "#7B68EE" for i in x]

    ax1.bar(x, vision_entropies, color=colors_ent, edgecolor="black", linewidth=0.3, width=0.7)
    ax1.set_ylabel("Mean Attention Entropy", fontsize=10)
    ax1.set_title("Vision Encoder (32 blocks) — Attention Entropy per Layer\n"
                  "Dark purple = full attention layers  |  Light purple = windowed attention",
                  fontsize=10)
    ax1.grid(alpha=0.3, axis="y")
    max_ent_v = math.log(vision_seq_len)
    ax1.axhline(max_ent_v, color="gray", linestyle="--", alpha=0.5,
                label=f"max entropy (log {vision_seq_len})")
    ax1.legend(fontsize=8)

    colors_sp = ["#6c3483" if i in fullatt_idxs else "#a569bd" for i in x]
    ax2.bar(x, vision_sparsities, color=colors_sp, edgecolor="black", linewidth=0.3, width=0.7)
    ax2.set_xlabel("Block index", fontsize=10)
    ax2.set_ylabel("Top-10% Concentration", fontsize=10)
    ax2.set_title("Attention Sparsity", fontsize=10)
    ax2.set_xticks(x)
    ax2.tick_params(axis="x", labelsize=7)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out3 = os.path.join(OUT_DIR, "3_vision_encoder_entropy.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out3}")

    # ── Plot 4: Vision encoder heatmaps at selected layers ───────────────────
    viz_layers = [l for l in VISION_VIZ_LAYERS if l < n_vision_layers]
    if viz_layers:
        fig, axes = plt.subplots(1, len(viz_layers),
                                 figsize=(5 * len(viz_layers), 4.5))
        if len(viz_layers) == 1:
            axes = [axes]

        for ax, li in zip(axes, viz_layers):
            attn = _captured_vision_attn[li].numpy()
            label = "FULL" if li in fullatt_idxs else "windowed"
            if attn.shape[0] > 512:
                step = attn.shape[0] // 512
                attn = attn[::step, ::step]
            im = ax.imshow(np.log10(attn + 1e-8), aspect="auto", cmap="magma",
                           vmin=-5, vmax=0)
            ax.set_title(f"Vision Block {li} ({label})", fontsize=11)
            ax.set_xlabel("Key position (patch index)")
            ax.set_ylabel("Query position (patch index)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log₁₀(attn)")

        fig.suptitle("Vision Encoder — Attention Heatmaps (mean over heads)\n"
                     "Full-attention layers show global patterns; windowed show block-diagonal",
                     fontsize=10, y=1.02)
        plt.tight_layout()
        out4 = os.path.join(OUT_DIR, "4_vision_encoder_heatmaps.png")
        plt.savefig(out4, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out4}")


# ── Plot 5: Side-by-side entropy comparison ──────────────────────────────────
if n_audio_layers > 0 and n_vision_layers > 0:
    fig, ax = plt.subplots(figsize=(12, 4))

    x_a = np.arange(n_audio_layers)
    x_v = np.arange(n_vision_layers)

    # Normalize entropies by max possible for fair comparison
    max_ent_a = math.log(audio_seq_len)
    max_ent_v = math.log(vision_seq_len)

    norm_audio  = np.array(audio_entropies)  / max_ent_a
    norm_vision = np.array(vision_entropies) / max_ent_v

    ax.plot(x_a, norm_audio,  "o-", color="#e74c3c", markersize=4, linewidth=1.5,
            label=f"Audio Encoder (seq={audio_seq_len})")
    ax.plot(x_v, norm_vision, "s-", color="#7B68EE", markersize=4, linewidth=1.5,
            label=f"Vision Encoder (seq={vision_seq_len})")

    ax.set_xlabel("Layer / Block index", fontsize=10)
    ax.set_ylabel("Normalized Entropy (entropy / max_entropy)", fontsize=10)
    ax.set_title("Encoder Attention Focus: Audio vs Vision across Depth\n"
                 "1.0 = perfectly uniform  |  0.0 = fully concentrated",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out5 = os.path.join(OUT_DIR, "5_encoder_entropy_comparison.png")
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out5}")


print(f"\nAll encoder visualizations saved to: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
