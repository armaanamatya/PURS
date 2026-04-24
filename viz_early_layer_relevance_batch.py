"""
Batch early-layer text→audio/video relevance scorer.

Iterates all entries in metadata.json, runs a hook-instrumented forward pass
on Thinker layers 0 and 1 for every (video, question) pair, and writes:

  Per-question plots:
    {out_dir}/{dataset}/{video_id}/{q_idx:03d}/text_to_audio.png
    {out_dir}/{dataset}/{video_id}/{q_idx:03d}/text_to_video.png

  Aggregate score log (one JSON line per question):
    {out_dir}/scores.jsonl

Run:
  python viz_early_layer_relevance_batch.py
  python viz_early_layer_relevance_batch.py --dataset worldsense --limit 20
  python viz_early_layer_relevance_batch.py --metadata metadata.json --videos videos/
"""

import argparse
import glob
import json
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(REPO_ROOT, "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qwen_mod
_qwen_mod.check_torch_load_is_safe = lambda: None
from qwen_omni_utils import process_mm_info

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--metadata",   default=os.path.join(REPO_ROOT, "metadata.json"))
parser.add_argument("--videos",     default=os.path.join(REPO_ROOT, "videos"))
parser.add_argument("--model_path", default="/data/armaan/models/Qwen2.5-Omni-7B")
parser.add_argument("--out_dir",    default=os.path.join(REPO_ROOT, "vizzing", "early_layer_relevance"))
parser.add_argument("--log",        default=None, help="JSONL log path (default: out_dir/scores.jsonl)")
parser.add_argument("--dataset",    default=None, help="Filter to one dataset name (e.g. worldsense)")
parser.add_argument("--limit",      type=int, default=None, help="Max video entries to process")
parser.add_argument("--gpus",       default="0,1", help="Comma-separated GPU IDs to spread model across (e.g. '0,1,2,3')")
parser.add_argument("--fps",        type=float, default=2.0,        help="Video sampling FPS (matches eval default)")
parser.add_argument("--max_frames", type=int,   default=128,         help="Max video frames (eval default for non-VideoMME)")
parser.add_argument("--max_pixels", type=int,   default=360*420,     help="Max pixels per frame (eval default)")
parser.add_argument("--skip_existing", action="store_true", help="Skip questions whose plot dir already exists")
args = parser.parse_args()

GPU_IDS     = [int(g) for g in args.gpus.split(",")]
HOOK_LAYERS = [0, 1]
LOG_PATH    = args.log or os.path.join(args.out_dir, "scores.jsonl")

os.makedirs(args.out_dir, exist_ok=True)

# ── Helpers (mirrors eval_qwen_omni.resolve_video_path) ───────────────────────
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
    if len(matches) == 1:
        return matches[0]
    return None


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

# ── Load model once ───────────────────────────────────────────────────────────
# Spread model evenly across specified GPUs so each GPU retains room for activations.
# With 8x RTX 6000 Ada (49 GB each), 2 GPUs gives ~24 GB/GPU for the 7B model,
# leaving ~25 GB free per GPU for the attention matrices of long videos.
_gpu_mem_per_device = "10GiB"
max_memory = {i: _gpu_mem_per_device for i in GPU_IDS}
print(f"Loading Qwen2.5-Omni-7B across GPUs {GPU_IDS} (≤{_gpu_mem_per_device} each) …")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,
    attn_implementation="flash_attention_2",
)
INPUT_DEVICE = model.thinker.model.embed_tokens.weight.device
model.eval()
# Offload TTS decoder to CPU — we never generate audio, only run the thinker forward.
# Mirrors the GPTQ device_map in eval_qwen_omni.py and saves ~5–8 GB on GPU.
if hasattr(model, "token2wav"):
    model.token2wav = model.token2wav.to("cpu")
processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id
print(f"Model ready. Input device: {INPUT_DEVICE}. max_frames={args.max_frames}\n")

# ── Load metadata ─────────────────────────────────────────────────────────────
with open(args.metadata) as f:
    raw = json.load(f)

entries = list(raw.values()) if isinstance(raw, dict) else raw
if args.dataset:
    canon = args.dataset.strip().lower().replace("_", "-").replace(" ", "-")
    entries = [e for e in entries if
               e.get("dataset", "").strip().lower().replace("_", "-") == canon]

entries = [e for e in entries if e.get("questions")]
if args.limit:
    entries = entries[:args.limit]

print(f"Processing {len(entries)} video entries …\n")

# ── Core: score one (video, question) pair ────────────────────────────────────
def score_question(video_path: str, question_text: str, dataset: str) -> dict | None:
    """Returns dict with per-token score arrays and metadata, or None on error."""

    system_text = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    video_element: dict = {"type": "video", "video": video_path,
                           "fps": args.fps,
                           "max_frames": args.max_frames,
                           "max_pixels": args.max_pixels}

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": [video_element, {"type": "text", "text": question_text}]},
    ]

    text   = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

    model.thinker.nframes = videos[0].shape[0]

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
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

    audio_scores: dict[int, np.ndarray] = {}
    video_scores: dict[int, np.ndarray] = {}

    def make_hook(li):
        # Post-hook: args[0] is the hidden states fed INTO self_attn (already layer-normed).
        # We compute Q[text] · K[audio/video]^T ourselves — only these small blocks,
        # never the full N×N attention matrix. Works with any attn_implementation.
        def hook(module, args, kwargs, output):
            with torch.no_grad():
                # hidden_states may arrive as positional arg or kwarg depending on caller
                hidden_states = args[0] if args else kwargs["hidden_states"]
                hs = hidden_states[0]  # (seq, D) — drop batch dim
                dev = hs.device

                t_pos = text_pos_gpu.to(dev)
                if t_pos.numel() == 0:
                    return output

                q_all = module.q_proj(hs[t_pos]).float()  # (T_text, H*d)
                num_heads    = module.num_heads
                num_kv_heads = module.num_key_value_heads
                head_dim     = module.head_dim
                T_text = t_pos.numel()

                # (num_heads, T_text, head_dim)
                q = q_all.view(T_text, num_heads, head_dim).permute(1, 0, 2)

                def _cross_scores(pos_gpu):
                    pos = pos_gpu.to(dev)
                    if pos.numel() == 0:
                        return None
                    k_all = module.k_proj(hs[pos]).float()  # (T_mod, Hkv*d)
                    T_mod = pos.numel()
                    k = k_all.view(T_mod, num_kv_heads, head_dim).permute(1, 0, 2)
                    if num_heads != num_kv_heads:
                        k = k.repeat_interleave(num_heads // num_kv_heads, dim=0)
                    # (num_heads, T_text, T_mod) → mean over heads & text → (T_mod,)
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                    return scores.mean(dim=(0, 1)).cpu().numpy()

                s = _cross_scores(audio_pos_gpu)
                if s is not None:
                    audio_scores[li] = s
                s = _cross_scores(video_pos_gpu)
                if s is not None:
                    video_scores[li] = s
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

    return {
        "seq_len": ids.shape[0],
        "n_audio": audio_pos.numel(),
        "n_video": video_pos.numel(),
        "n_text":  text_pos.numel(),
        "audio_scores": {str(li): audio_scores[li].tolist() for li in HOOK_LAYERS if li in audio_scores},
        "video_scores": {str(li): video_scores[li].tolist() for li in HOOK_LAYERS if li in video_scores},
        "audio_stats":  {
            str(li): {
                "min":     float(audio_scores[li].min()),
                "max":     float(audio_scores[li].max()),
                "mean":    float(audio_scores[li].mean()),
                "top5_idx": np.argsort(audio_scores[li])[-5:][::-1].tolist(),
            }
            for li in HOOK_LAYERS if li in audio_scores
        },
        "video_stats": {
            str(li): {
                "min":     float(video_scores[li].min()),
                "max":     float(video_scores[li].max()),
                "mean":    float(video_scores[li].mean()),
                "top5_idx": np.argsort(video_scores[li])[-5:][::-1].tolist(),
            }
            for li in HOOK_LAYERS if li in video_scores
        },
    }


def save_plots(result: dict, question_text: str, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)

    colors = {"0": "#2ecc71", "1": "#27ae60"}
    vid_colors = {"0": "#3498db", "1": "#2980b9"}

    # text→audio
    if result["audio_scores"]:
        fig, ax = plt.subplots(figsize=(14, 4))
        for li_str, scores in result["audio_scores"].items():
            sc = np.array(scores)
            ax.plot(sc, color=colors[li_str], linewidth=1.2, label=f"Layer {li_str}")
            thresh = np.percentile(sc, 80)
            ax.fill_between(np.arange(len(sc)), 0, sc, where=sc >= thresh,
                            color=colors[li_str], alpha=0.2)
        ax.set_xlabel("Audio token index"); ax.set_ylabel("Mean text→audio attention weight")
        ax.set_title(f"Text → Audio Relevance\nQ: \"{question_text[:80]}\"", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "text_to_audio.png"), dpi=120, bbox_inches="tight")
        plt.close()

    # text→video
    if result["video_scores"]:
        fig, ax = plt.subplots(figsize=(14, 4))
        for li_str, scores in result["video_scores"].items():
            sc = np.array(scores)
            ax.plot(sc, color=vid_colors[li_str], linewidth=1.2, label=f"Layer {li_str}")
            thresh = np.percentile(sc, 80)
            ax.fill_between(np.arange(len(sc)), 0, sc, where=sc >= thresh,
                            color=vid_colors[li_str], alpha=0.2)
        ax.set_xlabel("Video token index"); ax.set_ylabel("Mean text→video attention weight")
        ax.set_title(f"Text → Video Relevance\nQ: \"{question_text[:80]}\"", fontsize=10)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "text_to_video.png"), dpi=120, bbox_inches="tight")
        plt.close()


# ── Main loop ─────────────────────────────────────────────────────────────────
total_q = sum(len(e.get("questions", [])) for e in entries)
print(f"Total questions to score: {total_q}\n")

done = skipped = errors = 0

with open(LOG_PATH, "a") as log_f:
    for entry in entries:
        video_path = resolve_video_path(entry["file"], args.videos)
        dataset    = entry.get("dataset", "unknown")
        video_id   = safe_dir_name(Path(entry["file"]).stem)

        if video_path is None:
            print(f"  SKIP (no video): {entry['file']}")
            skipped += 1
            continue

        for q_idx, q in enumerate(entry["questions"]):
            question  = q["question"]
            task_type = q.get("task_type", entry.get("task_type", ""))

            plot_dir = os.path.join(args.out_dir, safe_dir_name(dataset),
                                    video_id, f"{q_idx:03d}")

            if args.skip_existing and os.path.exists(plot_dir):
                print(f"  SKIP (exists): {plot_dir}")
                skipped += 1
                continue

            label = f"{dataset}/{video_id}/q{q_idx}"
            print(f"[{done+1}/{total_q}] {label}  —  {question[:60]}")

            try:
                result = score_question(video_path, question, dataset)
            except RuntimeError as exc:
                print(f"  ERROR: {exc}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    print("  CUDA context corrupted — exiting. Re-run with --skip_existing to resume.")
                    sys.exit(1)
                errors += 1
                continue
            except Exception as exc:
                print(f"  ERROR: {exc}")
                errors += 1
                continue

            save_plots(result, question, plot_dir)

            record = {
                "dataset":    dataset,
                "file":       entry["file"],
                "video_id":   video_id,
                "task_type":  task_type,
                "question_idx": q_idx,
                "question":   question,
                "seq_len":    result["seq_len"],
                "n_audio":    result["n_audio"],
                "n_video":    result["n_video"],
                "n_text":     result["n_text"],
                "audio_stats":  result["audio_stats"],
                "video_stats":  result["video_stats"],
                "audio_scores": result["audio_scores"],
                "video_scores": result["video_scores"],
                "plot_dir":   plot_dir,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            done += 1

            # Print compact summary to terminal
            for li_str in result["audio_stats"]:
                s = result["audio_stats"][li_str]
                print(f"  audio L{li_str}: mean={s['mean']:.4f}  max={s['max']:.4f}  top5={s['top5_idx']}")
            for li_str in result["video_stats"]:
                s = result["video_stats"][li_str]
                print(f"  video L{li_str}: mean={s['mean']:.4f}  max={s['max']:.4f}  top5={s['top5_idx']}")

print(f"\nDone. scored={done}  skipped={skipped}  errors={errors}")
print(f"Plots → {args.out_dir}/")
print(f"Log   → {LOG_PATH}")
