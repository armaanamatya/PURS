"""
mechinterp/depth_curve_batch.py
Batch depth-curve robustness check: audio-leaning vs video-leaning groups.

Reuses scaffolding from viz_layer_depth_experiment.py:
  - multi-GPU model load via device_map="auto" + max_memory
  - resolve_video_path (Windows-backslash + recursive-glob fallback)
  - per-clip JSONL writer + aggregate-at-end pattern

Reuses hook logic from viz_attention_depth_curve.py (single-clip):
  - eager attention + output_attentions=True with a streaming hook that
    captures compact stats per layer and immediately returns None for the
    attention tensor (so GPU memory stays bounded layer-by-layer)
  - same three metrics (layer_data, gini_data, last_token_data) for direct
    apples-to-apples comparison with the single-clip plots

Tests, in order of cost:
  1. Layer-4 collapse robustness across N clips per group
  2. Layer-0 question-conditional weighting (audio_share[L0] higher on audio group)
  3. L26-27 modality re-engagement specific to audio group?

Outputs (under --out_dir, default mechinterp/outputs/depth_curve_batch/):
  samples/<group>__<safename>__q<idx>.json    per-(clip,question) stats
  summary.json                                  aggregated mean/std arrays
  figures/group_last_token.png                  side-by-side stacked-area, per-group means
  figures/group_crossmodal.png                  audio→text & video→text overlay (mean ± 1σ)
  run_meta.json                                 run config

Run:
  cd /data/armaan/purs
  python mechinterp/depth_curve_batch.py --gpus 2,3,4,5 --n_audio 10 --n_video 10
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
OMNIZIP_DIR = os.path.join(REPO_ROOT, "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None
from qwen_omni_utils import process_mm_info


# --------------------------------------------------------------- args / paths --
parser = argparse.ArgumentParser()
parser.add_argument("--metadata",       default=os.path.join(REPO_ROOT, "videos", "metadata.json"))
parser.add_argument("--videos",         default=os.path.join(REPO_ROOT, "videos"))
parser.add_argument("--model_path",     default="/data/armaan/models/Qwen2.5-Omni-7B")
parser.add_argument("--out_dir",        default=os.path.join(SCRIPT_DIR, "outputs", "depth_curve_batch"))
parser.add_argument("--n_audio",        type=int, default=10)
parser.add_argument("--n_video",        type=int, default=10)
parser.add_argument("--audio_datasets", nargs="+", default=["daily-omni"])
parser.add_argument("--video_datasets", nargs="+", default=["video-mme"])
parser.add_argument("--gpus",           default="2,3", help="Comma-separated GPU IDs for model sharding")
parser.add_argument("--max_memory_per_gpu", default="14GiB",
                    help="HF accelerate max_memory per GPU; raise if model alone exceeds")
parser.add_argument("--fps",            type=float, default=2.0)
parser.add_argument("--max_frames",     type=int, default=32)
parser.add_argument("--max_pixels",     type=int, default=200704)  # 448*448
args = parser.parse_args()

GPU_IDS = [int(g) for g in args.gpus.split(",")]
N_LAYERS = 28
os.makedirs(args.out_dir, exist_ok=True)
SAMPLES_DIR = os.path.join(args.out_dir, "samples")
FIGS_DIR    = os.path.join(args.out_dir, "figures")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

with open(os.path.join(args.out_dir, "run_meta.json"), "w") as _f:
    json.dump(vars(args), _f, indent=2)


# ---------------------------------------------------------- helpers reused --

def resolve_video_path(file_field, videos_dir):
    """Lifted from viz_layer_depth_experiment.py."""
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
    filename = rel_norm.split("/")[-1]
    for match in glob.glob(os.path.join(videos_dir, "**", filename), recursive=True):
        if match.replace("\\", "/").endswith(rel_norm):
            return match
    matches = glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
    return matches[0] if len(matches) == 1 else None


def safe_dir_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


def select_clips(metadata_path, audio_datasets, video_datasets, n_audio, n_video):
    with open(metadata_path) as f:
        meta = json.load(f)
    if isinstance(meta, dict):
        meta = list(meta.values())

    def take(filter_fn, n):
        out, seen = [], set()
        # First pass: one question per unique clip for diversity.
        for e in meta:
            if not filter_fn(e):
                continue
            if e["file"] in seen:
                continue
            seen.add(e["file"])
            out.append((e, 0))
            if len(out) >= n:
                return out
        # Second pass: more questions per same clip if pool is shallow.
        for e in meta:
            if not filter_fn(e):
                continue
            for qi in range(1, len(e["questions"])):
                out.append((e, qi))
                if len(out) >= n:
                    return out
        return out

    return (take(lambda e: e["dataset"] in audio_datasets, n_audio),
            take(lambda e: e["dataset"] in video_datasets, n_video))


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


# --------------------------------- Streaming hook (from viz_attention_depth_curve.py) --

def _mean_block(head_mean, src_pos, dst_pos):
    if src_pos.numel() == 0 or dst_pos.numel() == 0:
        return 0.0
    return head_mean.index_select(0, src_pos).index_select(1, dst_pos).mean().item()


def _gini_for_queries(head_mean, query_pos):
    if query_pos.numel() == 0:
        return 0.0
    rows = head_mean.index_select(0, query_pos).float()
    sorted_rows = torch.sort(rows, dim=-1).values
    n = sorted_rows.shape[-1]
    idx = torch.arange(1, n + 1, device=sorted_rows.device, dtype=sorted_rows.dtype).view(1, -1)
    denom = (n * sorted_rows.sum(dim=-1)).clamp_min(1e-12)
    g = (2 * (idx * sorted_rows).sum(dim=-1) / denom) - (n + 1) / n
    return g.mean().item()


def setup_hooks(model, modalities_gpu, directions, last_q):
    """Register a streaming hook on every Thinker layer self_attn that captures
    compact stats and immediately returns None for attention (so GPU memory
    holds at most one layer's attention tensor at a time)."""
    layer_data = {d: [] for d in directions}
    gini_data = {m: [] for m in modalities_gpu}
    last_token_data = {m: [] for m in modalities_gpu}

    def make_hook(_layer_idx):
        def hook(module, args_h, kwargs, output):
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                return output
            with torch.no_grad():
                head_mean = output[1][0].mean(dim=0)
                # Move pos arrays to layer device (sharding may put each layer on a different GPU).
                hd = head_mean.device
                mods = {n: p.to(hd) for n, p in modalities_gpu.items()}
                for src, dst in directions:
                    layer_data[(src, dst)].append(_mean_block(head_mean, mods[src], mods[dst]))
                for mod, pos in mods.items():
                    gini_data[mod].append(_gini_for_queries(head_mean, pos))
                    if pos.numel() == 0:
                        last_token_data[mod].append(0.0)
                    else:
                        last_token_data[mod].append(
                            head_mean[last_q].index_select(0, pos).sum().item())
            return (output[0], None) + output[2:]
        return hook

    handles = []
    for i, layer in enumerate(model.thinker.model.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True))
    return handles, layer_data, gini_data, last_token_data


# ---------------------------------------------------------------- per-clip --

def score_clip_question(model, processor, video_path, question, input_device,
                        audio_token_id, video_token_id):
    system_text = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    video_element = {"type": "video", "video": video_path,
                     "fps": args.fps, "max_frames": args.max_frames,
                     "max_pixels": args.max_pixels}
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [video_element, {"type": "text", "text": question}]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    n_frames = videos[0].shape[0]
    model.thinker.nframes = n_frames

    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(input_device)
    for k, v in inputs.items():
        if v.is_floating_point():
            inputs[k] = v.to(torch.float16)
    thinker_inputs = {k: v for k, v in inputs.items() if k in THINKER_KEYS}

    ids = inputs["input_ids"][0].cpu()
    audio_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0].to(input_device)
    video_pos = (ids == video_token_id).nonzero(as_tuple=True)[0].to(input_device)
    text_pos = ((ids != audio_token_id) & (ids != video_token_id)).nonzero(as_tuple=True)[0].to(input_device)
    seq_len = ids.shape[0]
    last_q = seq_len - 1

    modalities_gpu = {"audio": audio_pos, "video": video_pos, "text": text_pos}
    directions = [(s, d) for s in modalities_gpu for d in modalities_gpu]

    handles, layer_data, gini_data, last_token_data = setup_hooks(
        model, modalities_gpu, directions, last_q)
    try:
        with torch.no_grad():
            _ = model.thinker(**thinker_inputs, output_attentions=True,
                              use_audio_in_video=True, use_cache=False, return_dict=True)
    finally:
        for h in handles:
            h.remove()
        torch.cuda.empty_cache()

    return {
        "seq_len": int(seq_len),
        "n_audio": int(audio_pos.numel()),
        "n_video": int(video_pos.numel()),
        "n_text": int(text_pos.numel()),
        "n_frames": int(n_frames),
        "layer_data": {f"{s}->{d}": list(map(float, v)) for (s, d), v in layer_data.items()},
        "gini_data": {m: list(map(float, v)) for m, v in gini_data.items()},
        "last_token_data": {m: list(map(float, v)) for m, v in last_token_data.items()},
    }


# ------------------------------------------------------------ aggregation --

def aggregate(samples):
    if not samples:
        return None
    out = {}
    for mod in ("audio", "video", "text"):
        arr = np.array([s["last_token_data"][mod] for s in samples], dtype=float)
        out[f"last_token_{mod}_mean"] = arr.mean(0)
        out[f"last_token_{mod}_std"] = arr.std(0)
        garr = np.array([s["gini_data"][mod] for s in samples], dtype=float)
        out[f"gini_{mod}_mean"] = garr.mean(0)
        out[f"gini_{mod}_std"] = garr.std(0)
    for s in ("audio", "video", "text"):
        for d in ("audio", "video", "text"):
            arr = np.array([sample["layer_data"][f"{s}->{d}"] for sample in samples], dtype=float)
            out[f"layer_{s}->{d}_mean"] = arr.mean(0)
            out[f"layer_{s}->{d}_std"] = arr.std(0)
    return out


# ---------------------------------------------------------------- plotting --

def plot_last_token(audio_agg, video_agg, n_layers, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    x = np.arange(n_layers)
    for ax, agg, title in [(axes[0], audio_agg, "Audio-leaning clips"),
                            (axes[1], video_agg, "Video-leaning clips")]:
        a = agg["last_token_audio_mean"]
        v = agg["last_token_video_mean"]
        t = agg["last_token_text_mean"]
        total = a + v + t + 1e-12
        ax.stackplot(x, a / total, v / total, t / total,
                     labels=["Audio", "Video", "Text"],
                     colors=["#e74c3c", "#3498db", "#2ecc71"], alpha=0.7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Decoder Layer")
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.0%}"))
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Attention Share")
    fig.suptitle("Last-Token Modality Attention Across Depth — group means", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_crossmodal(audio_agg, video_agg, n_layers, out_path):
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_layers)
    pairs = [("audio->text", "audio→text", "#e74c3c"),
             ("video->text", "video→text", "#3498db")]
    for key, label, color in pairs:
        am = audio_agg[f"layer_{key}_mean"]; asd = audio_agg[f"layer_{key}_std"]
        vm = video_agg[f"layer_{key}_mean"]; vsd = video_agg[f"layer_{key}_std"]
        ax.plot(x, am, "-", color=color, lw=2, marker="o", ms=4, label=f"{label} (audio-leaning)")
        ax.fill_between(x, am - asd, am + asd, color=color, alpha=0.15)
        ax.plot(x, vm, "--", color=color, lw=1.5, marker="s", ms=3, label=f"{label} (video-leaning)")
        ax.fill_between(x, vm - vsd, vm + vsd, color=color, alpha=0.08)
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Mean attention weight (query→key)")
    ax.set_title("Cross-modal flow: audio-leaning vs video-leaning groups (mean ± 1σ)")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# -------------------------------------------------------------------- main --

def main():
    audio_clips, video_clips = select_clips(
        args.metadata, set(args.audio_datasets), set(args.video_datasets),
        args.n_audio, args.n_video)
    print(f"Selected {len(audio_clips)} audio-leaning + {len(video_clips)} video-leaning samples")

    max_memory = {i: args.max_memory_per_gpu for i in GPU_IDS}
    print(f"Loading Qwen2.5-Omni-7B sharded across GPUs {GPU_IDS} "
          f"(eager attn, ≤{args.max_memory_per_gpu} each)…")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="eager",
    )
    model.eval()
    if hasattr(model, "token2wav"):
        model.token2wav = model.token2wav.to("cpu")
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    input_device = model.thinker.model.embed_tokens.weight.device
    audio_token_id = model.thinker.config.audio_token_id
    video_token_id = model.thinker.config.video_token_id
    print(f"Model ready. Input device: {input_device}\n")

    audio_samples, video_samples = [], []
    for group, clips, target in [("audio", audio_clips, audio_samples),
                                  ("video", video_clips, video_samples)]:
        for i, (entry, qi) in enumerate(clips):
            video_path = resolve_video_path(entry["file"], args.videos)
            if not video_path:
                print(f"  [{group} {i+1}/{len(clips)}] SKIP unresolved: {entry['file']}")
                continue
            question = entry["questions"][qi]["question"]
            t0 = time.time()
            try:
                stats = score_clip_question(model, processor, video_path, question,
                                            input_device, audio_token_id, video_token_id)
            except torch.cuda.OutOfMemoryError:
                print(f"  [{group} {i+1}/{len(clips)}] OOM — skipping: {entry['file']}")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  [{group} {i+1}/{len(clips)}] ERR {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue
            stats.update({
                "video_path": entry["file"],
                "question": question,
                "dataset": entry["dataset"],
                "task_type": entry.get("task_type", ""),
                "question_idx": qi,
            })
            target.append(stats)
            sample_name = f"{group}__{safe_dir_name(entry['file'])}__q{qi}.json"
            with open(os.path.join(SAMPLES_DIR, sample_name), "w") as f:
                json.dump(stats, f, indent=2)
            print(f"  [{group} {i+1}/{len(clips)}] {time.time()-t0:.1f}s "
                  f"T={stats['seq_len']} A={stats['n_audio']} V={stats['n_video']} "
                  f"frames={stats['n_frames']}")

    print(f"\nAggregating: audio_n={len(audio_samples)}, video_n={len(video_samples)}")
    audio_agg = aggregate(audio_samples)
    video_agg = aggregate(video_samples)
    if not audio_agg or not video_agg:
        print("ERROR: empty group — cannot aggregate")
        sys.exit(1)

    summary = {
        "n_audio_samples": len(audio_samples),
        "n_video_samples": len(video_samples),
        "n_layers": N_LAYERS,
        "audio_group": {k: list(map(float, v)) for k, v in audio_agg.items()},
        "video_group": {k: list(map(float, v)) for k, v in video_agg.items()},
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_last_token(audio_agg, video_agg, N_LAYERS,
                    os.path.join(FIGS_DIR, "group_last_token.png"))
    plot_crossmodal(audio_agg, video_agg, N_LAYERS,
                    os.path.join(FIGS_DIR, "group_crossmodal.png"))
    print(f"Done → {args.out_dir}")


if __name__ == "__main__":
    main()
