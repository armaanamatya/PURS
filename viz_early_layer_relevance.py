"""
Early Thinker layer text→audio/video relevance scorer.

Hooks Thinker layers 0 and 1 only. For each audio/video token, computes
the mean attention weight that text queries allocated to it — averaged over
all text tokens and all heads. This gives a per-token relevance curve:

    score[j] = mean over (heads h, text tokens t) of attn[h, t, j]

Plots:
  1. Text → audio relevance per token (layer 0 and layer 1)
  2. Text → video relevance per token (layer 0 and layer 1)

Run from: /data/armaan/purs/
  python viz_early_layer_relevance.py
  python viz_early_layer_relevance.py --video_path <path> --question "..."
"""

import argparse
import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OMNIZIP_DIR = os.path.join(os.path.dirname(__file__), "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None
from qwen_omni_utils import process_mm_info

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "videos", "worldsense",
                          "attribute_reasoning", "video.mp4")
QUESTION   = "What is the profession of the man with a beard wearing a suit in the video?"
MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
OUT_DIR    = os.path.join(os.path.dirname(__file__), "vizzing", "early_layer_relevance")

HOOK_LAYERS = [0, 1]  # only these two Thinker layers are instrumented

parser = argparse.ArgumentParser()
parser.add_argument("--video_path",  default=VIDEO_PATH)
parser.add_argument("--question",    default=QUESTION)
parser.add_argument("--model_path",  default=MODEL_PATH)
parser.add_argument("--out_dir",     default=OUT_DIR)
parser.add_argument("--fps",         type=float, default=None)
parser.add_argument("--max_pixels",  type=int,   default=None)
parser.add_argument("--max_frames",  type=int,   default=None)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Qwen2.5-Omni-7B (eager attention) …")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    attn_implementation="eager",
)
model.eval()
processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

# ── Prepare inputs ────────────────────────────────────────────────────────────
video_element = {"type": "video", "video": args.video_path}
if args.fps         is not None: video_element["fps"]        = args.fps
if args.max_pixels  is not None: video_element["max_pixels"] = args.max_pixels
if args.max_frames  is not None: video_element["max_frames"] = args.max_frames

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
        "content": [video_element, {"type": "text", "text": args.question}],
    },
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

num_frames = videos[0].shape[0]
model.thinker.nframes = num_frames
print(f"Video frames sampled: {num_frames}")

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

# ── Modality masks ────────────────────────────────────────────────────────────
input_ids      = inputs["input_ids"]
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id

ids       = input_ids[0].cpu()
audio_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]
video_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
text_pos  = ((ids != audio_token_id) & (ids != video_token_id)).nonzero(as_tuple=True)[0]

seq_len = ids.shape[0]
n_audio, n_video, n_text = audio_pos.numel(), video_pos.numel(), text_pos.numel()
print(f"Sequence: {seq_len} tokens  |  audio={n_audio}  video={n_video}  text={n_text}")

# GPU tensors for indexing inside hooks
audio_pos_gpu = audio_pos.to("cuda:0")
video_pos_gpu = video_pos.to("cuda:0")
text_pos_gpu  = text_pos.to("cuda:0")

# Storage: layer_idx → numpy array of shape (T_audio,) or (T_video,)
audio_scores = {}  # {0: arr, 1: arr}
video_scores = {}

# ── Hook factory ──────────────────────────────────────────────────────────────
def make_hook(layer_idx):
    def hook(module, args, kwargs, output):
        if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
            return output
        with torch.no_grad():
            # attn: (1, heads, seq, seq)  — eager mode only
            attn = output[1]

            # text→audio: for each audio token j, mean over heads and text queries
            # attn[0][:, text_pos_gpu, :] → (heads, T_text, seq)
            # [:, :, audio_pos_gpu]       → (heads, T_text, T_audio)
            # .mean(dim=(0,1))            → (T_audio,)
            if audio_pos_gpu.numel() > 0 and text_pos_gpu.numel() > 0:
                audio_scores[layer_idx] = (
                    attn[0][:, text_pos_gpu, :][:, :, audio_pos_gpu]
                    .float().mean(dim=(0, 1)).cpu().numpy()
                )

            if video_pos_gpu.numel() > 0 and text_pos_gpu.numel() > 0:
                video_scores[layer_idx] = (
                    attn[0][:, text_pos_gpu, :][:, :, video_pos_gpu]
                    .float().mean(dim=(0, 1)).cpu().numpy()
                )

        # Free the attention tensor from the forward graph immediately
        return (output[0], None) + output[2:]
    return hook

# ── Register hooks on layers 0 and 1 only ────────────────────────────────────
hooks = []
for li in HOOK_LAYERS:
    h = model.thinker.model.layers[li].self_attn.register_forward_hook(
        make_hook(li), with_kwargs=True
    )
    hooks.append(h)

print(f"Hooks registered on Thinker layers {HOOK_LAYERS}. Running forward pass …")

try:
    with torch.no_grad():
        model.thinker(
            **thinker_inputs,
            output_attentions=True,
            use_audio_in_video=True,
            use_cache=False,
            return_dict=True,
        )
finally:
    for h in hooks:
        h.remove()

print(f"Captured scores for layers: audio={list(audio_scores.keys())}  "
      f"video={list(video_scores.keys())}")

# ── Plot 1: Text → Audio per-token relevance ──────────────────────────────────
colors = {0: "#2ecc71", 1: "#27ae60"}
alphas = {0: 1.0,       1: 0.65}

fig, ax = plt.subplots(figsize=(14, 4))
x_audio = np.arange(n_audio)
for li in HOOK_LAYERS:
    if li not in audio_scores:
        continue
    scores = audio_scores[li]
    ax.plot(x_audio, scores, color=colors[li], alpha=alphas[li],
            linewidth=1.2, label=f"Layer {li}")
    # Shade top-20% tokens
    thresh = np.percentile(scores, 80)
    ax.fill_between(x_audio, 0, scores, where=scores >= thresh,
                    color=colors[li], alpha=0.2)

ax.set_xlabel("Audio token index (temporal order)", fontsize=11)
ax.set_ylabel("Mean text→audio attention weight", fontsize=11)
ax.set_title(
    "Text → Audio Relevance per Token  (Thinker layers 0 & 1)\n"
    f"Q: \"{args.question[:80]}\"",
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
out1 = os.path.join(args.out_dir, "1_text_to_audio_relevance.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out1}")

# ── Plot 2: Text → Video per-token relevance ──────────────────────────────────
vid_colors = {0: "#3498db", 1: "#2980b9"}

fig, ax = plt.subplots(figsize=(14, 4))
x_video = np.arange(n_video)
for li in HOOK_LAYERS:
    if li not in video_scores:
        continue
    scores = video_scores[li]
    ax.plot(x_video, scores, color=vid_colors[li], alpha=alphas[li],
            linewidth=1.2, label=f"Layer {li}")
    thresh = np.percentile(scores, 80)
    ax.fill_between(x_video, 0, scores, where=scores >= thresh,
                    color=vid_colors[li], alpha=0.2)

ax.set_xlabel("Video token index (patch order)", fontsize=11)
ax.set_ylabel("Mean text→video attention weight", fontsize=11)
ax.set_title(
    "Text → Video Relevance per Token  (Thinker layers 0 & 1)\n"
    f"Q: \"{args.question[:80]}\"",
    fontsize=11
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
out2 = os.path.join(args.out_dir, "2_text_to_video_relevance.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out2}")

# ── Plot 3: Side-by-side summary ──────────────────────────────────────────────
fig, (ax_a, ax_v) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

for li in HOOK_LAYERS:
    if li in audio_scores:
        sc = audio_scores[li]
        ax_a.plot(np.arange(n_audio), sc, color=colors[li],
                  alpha=alphas[li], linewidth=1.2, label=f"Layer {li}")
    if li in video_scores:
        sc = video_scores[li]
        ax_v.plot(np.arange(n_video), sc, color=vid_colors[li],
                  alpha=alphas[li], linewidth=1.2, label=f"Layer {li}")

ax_a.set_ylabel("text→audio score", fontsize=10)
ax_a.set_title("Text → Audio", fontsize=11)
ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

ax_v.set_ylabel("text→video score", fontsize=10)
ax_v.set_xlabel("Token index", fontsize=10)
ax_v.set_title("Text → Video", fontsize=11)
ax_v.legend(fontsize=8); ax_v.grid(alpha=0.3)

fig.suptitle(
    f"Early Thinker Relevance Scores  |  Q: \"{args.question[:70]}\"",
    fontsize=11, y=1.01
)
plt.tight_layout()
out3 = os.path.join(args.out_dir, "3_combined_relevance.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {out3}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n── Score summary ──")
for li in HOOK_LAYERS:
    if li in audio_scores:
        sc = audio_scores[li]
        top5 = np.argsort(sc)[-5:][::-1]
        print(f"Layer {li}  text→audio  min={sc.min():.4f}  max={sc.max():.4f}  "
              f"mean={sc.mean():.4f}  top-5 tokens={top5.tolist()}")
    if li in video_scores:
        sc = video_scores[li]
        top5 = np.argsort(sc)[-5:][::-1]
        print(f"Layer {li}  text→video  min={sc.min():.4f}  max={sc.max():.4f}  "
              f"mean={sc.mean():.4f}  top-5 tokens={top5.tolist()}")

print(f"\nAll outputs saved to: {args.out_dir}/")
