"""
precompute_l6_saliency.py

Compute layer-6 Q·K cross-modal saliency scores for every video in the metadata
using the STOCK Qwen2.5-Omni model (no OmniZip). Results are written incrementally
so that a CUDA crash mid-run does not lose prior work.

On a CUDA error the script exits with code 1 so it can be restarted automatically:

    until CUDA_VISIBLE_DEVICES=6,7 python precompute_l6_saliency.py \\
        --existing_cache vizzing/layer_depth_experiment_v2/worldsense/layer_depth_stats.jsonl \\
        --output vizzing/layer_depth_all_full.jsonl \\
        --metadata /data/armaan/purs/videos/metadata.json \\
        --videos /data/armaan/purs/videos \\
        --model_path /data/armaan/models/Qwen2.5-Omni-7B \\
        --gpus 0,1; do echo "Restarting after CUDA error…"; sleep 3; done

Why stock model?
    OmniZip compresses audio tokens BEFORE the Thinker layers run. Hooking layer 6
    during an OmniZip forward pass would see only the surviving tokens, making the
    saliency scores unusable as a replacement for OmniZip's full keep mask. The stock
    model keeps all tokens intact.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

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
parser.add_argument("--metadata",        default="/workspace/metadata.json")
parser.add_argument("--videos",          default="/workspace/videos")
parser.add_argument("--model_path",      default="/data/armaan/models/Qwen2.5-Omni-7B")
parser.add_argument("--existing_cache",  default=None,
                    help="JSONL to seed the cache from (records copied verbatim, not recomputed).")
parser.add_argument("--output",          default="vizzing/layer_depth_all_full.jsonl",
                    help="Output JSONL. Records already in this file are also skipped.")
parser.add_argument("--dataset",         default=None)
parser.add_argument("--layer",           type=int, default=6)
parser.add_argument("--fps",             type=float, default=2.0)
parser.add_argument("--max_frames",      type=int,   default=64,
                    help="Max video frames for the saliency pass. Audio token count is "
                         "independent of this value so it does not affect cached audio scores.")
parser.add_argument("--max_pixels",      type=int,   default=100800,  # 360*280
                    help="Max pixels per frame. Keep below 105369 to avoid processor warnings.")
parser.add_argument("--query",           default="Describe this video.")
parser.add_argument("--gpus",            default="0,1")
args = parser.parse_args()

GPU_IDS = [int(g) for g in args.gpus.split(",")]
os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_path(p: str) -> str:
    return p.replace("\\", "/").strip().lower()


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


# ── Load already-covered file keys ───────────────────────────────────────────
def load_covered_keys(path: str | None) -> set[str]:
    keys: set[str] = set()
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        keys.add(normalize_path(rec["file"]))
                    except Exception:
                        pass
    return keys


existing_keys   = load_covered_keys(args.existing_cache)
output_keys     = load_covered_keys(args.output)
cached_file_keys = existing_keys | output_keys

print(f"Already covered: {len(cached_file_keys)} videos "
      f"({len(existing_keys)} from existing_cache, {len(output_keys)} from output)")


# ── Copy existing_cache records that are not yet in output ────────────────────
output_existing_keys = load_covered_keys(args.output)
if args.existing_cache and os.path.exists(args.existing_cache):
    with open(args.existing_cache, "r", encoding="utf-8") as fh, \
         open(args.output, "a", encoding="utf-8") as out_f:
        copied = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = normalize_path(rec["file"])
                if key not in output_existing_keys:
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    copied += 1
            except Exception:
                pass
    if copied:
        print(f"Copied {copied} records from existing_cache to output")


# ── Load metadata ─────────────────────────────────────────────────────────────
with open(args.metadata) as f:
    raw = json.load(f)
entries = list(raw.values()) if isinstance(raw, dict) else raw

if args.dataset:
    canon   = args.dataset.strip().lower().replace("_", "-").replace(" ", "-")
    entries = [e for e in entries
               if e.get("dataset", "").strip().lower().replace("_", "-") == canon]

to_compute = [e for e in entries
              if e.get("questions") and normalize_path(e["file"]) not in cached_file_keys]

print(f"Metadata: {len(entries)} entries | to compute: {len(to_compute)}\n")

if not to_compute:
    print("All videos already cached.")
    sys.exit(0)


# ── Load model ────────────────────────────────────────────────────────────────
max_memory = {i: "22GiB" for i in GPU_IDS}
print(f"Loading Qwen2.5-Omni-7B across GPUs {GPU_IDS} …")
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
        model.token2wav = model.token2wav.to("cpu")
    except NotImplementedError:
        pass
if hasattr(model, "disable_talker"):
    model.disable_talker()
processor      = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
audio_token_id = model.thinker.config.audio_token_id
video_token_id = model.thinker.config.video_token_id
print(f"Model ready. Input device: {INPUT_DEVICE}\n")


# ── Saliency extraction (identical hook to viz_layer_depth_experiment.py) ─────
def compute_saliency(video_path: str, query: str) -> dict | None:
    system_text = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": [
            {"type": "video", "video": video_path,
             "fps": args.fps, "max_frames": args.max_frames, "max_pixels": args.max_pixels},
            {"type": "text", "text": query},
        ]},
    ]
    text   = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    model.thinker.nframes  = videos[0].shape[0] if videos else 0

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

    result: dict = {}

    def hook_fn(module, args_h, kwargs, output):
        with torch.no_grad():
            hidden_states = args_h[0] if args_h else kwargs["hidden_states"]
            hs  = hidden_states[0]
            dev = hs.device

            t_pos = text_pos_gpu.to(dev)
            if t_pos.numel() == 0:
                result["audio"] = None
                result["video"] = None
                return output

            num_heads = module.num_heads
            num_kv    = module.num_key_value_heads
            head_dim  = module.head_dim
            T_text    = t_pos.numel()

            q_all = module.q_proj(hs[t_pos]).float()
            q     = q_all.view(T_text, num_heads, head_dim).permute(1, 0, 2)

            def _cross(pos_gpu):
                pos = pos_gpu.to(dev)
                if pos.numel() == 0:
                    return None
                k_all = module.k_proj(hs[pos]).float()
                T_mod = pos.numel()
                k = k_all.view(T_mod, num_kv, head_dim).permute(1, 0, 2)
                if num_heads != num_kv:
                    k = k.repeat_interleave(num_heads // num_kv, dim=0)
                s = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
                return s.mean(dim=(0, 1)).cpu().numpy()

            result["audio"] = _cross(audio_pos_gpu)
            result["video"] = _cross(video_pos_gpu)
        return output

    handle = model.thinker.model.layers[args.layer].self_attn.register_forward_hook(
        hook_fn, with_kwargs=True
    )
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
        handle.remove()
        torch.cuda.empty_cache()

    return result if result else None


# ── Main loop — writes incrementally, exits on CUDA error for restart ─────────
skipped = computed = 0

with open(args.output, "a", encoding="utf-8") as out_f:
    for v_idx, entry in enumerate(to_compute):
        file_field  = entry["file"]
        dataset     = entry.get("dataset", "unknown")
        video_id    = safe_dir_name(Path(file_field).stem)
        entry_label = f"{dataset}/{video_id}"

        video_path = resolve_video_path(file_field, args.videos)
        if video_path is None:
            print(f"[{v_idx+1}/{len(to_compute)}] SKIP {entry_label}: video not found")
            skipped += 1
            continue

        print(f"[{v_idx+1}/{len(to_compute)}] {entry_label}")
        try:
            scores = compute_saliency(video_path, args.query)
        except Exception as exc:
            is_cuda = "cuda" in str(exc).lower() or "cuda" in type(exc).__name__.lower()
            print(f"  ERROR: {exc}")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if is_cuda:
                print("  CUDA error — exiting for restart (already-written records are safe)")
                sys.exit(1)
            skipped += 1
            continue

        if scores is None or (scores.get("audio") is None and scores.get("video") is None):
            print(f"  WARN: no tokens found — skipping")
            skipped += 1
            continue

        audio_arr = scores.get("audio")
        video_arr = scores.get("video")
        n_audio   = int(audio_arr.shape[0]) if audio_arr is not None else 0
        n_video   = int(video_arr.shape[0]) if video_arr is not None else 0
        print(f"  audio tokens: {n_audio}  video tokens: {n_video}")

        record = {
            "video_id":         video_id,
            "dataset":          dataset,
            "file":             file_field,
            "n_questions":      1,
            "n_successful":     1,
            "precomputed":      True,
            "precompute_query": args.query,
            "raw_audio_scores": {str(args.layer): [audio_arr.tolist()] if audio_arr is not None else []},
            "raw_video_scores": {str(args.layer): [video_arr.tolist()] if video_arr is not None else []},
        }
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        computed += 1

print(f"\nDone. Computed: {computed}  Skipped: {skipped}  Output: {args.output}")
