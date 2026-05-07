"""eval_videozip.py

Benchmark Qwen2.5-Omni-7B + VideoZip on the same metadata used by the audio-side
L6 cache eval (`eval_qwen_omni_zip_cached.py`). VideoZip flips OmniZip's guidance
direction: video saliency drives audio compression. Saliency is read from the
EXISTING `vizzing/layer_depth_all_full.jsonl` cache (no new precompute needed).

Usage:
    CUDA_VISIBLE_DEVICES=0 python videozip/eval/eval_videozip.py \\
        --cache vizzing/layer_depth_all_full.jsonl \\
        --layer 6 \\
        --metadata videos/metadata.json \\
        --videos videos \\
        --output vizzing/results_videozip_l6.jsonl \\
        --log vizzing/results_videozip_l6.log \\
        --vram_log vizzing/vram_videozip_l6.jsonl \\
        --device cuda:0

Saliency source options (--video_saliency_source):
    l6_cached   read 1D Q*K saliency from the JSONL cache (default; free 1.6x prefill)
    attn        live Thinker attn_logits as the audio-side cache eval does
    sim_only    Aurelle's frozen cross-modal similarity baseline (Ablation A row)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Self-bootstrap so this works as both `python videozip/eval/eval_videozip.py ...`
# (path-based) and `python -m videozip.eval.eval_videozip ...` (module-based). When
# run path-based, only the script dir is on sys.path; we add the project root so
# `import videozip` resolves.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

import videozip  # bootstraps sys.path: project root, OmniZip-main/omnizip, qwen-omni-utils

import eval_qwen_omni_zip as base  # type: ignore[import-not-found]  # at project root
import omnizip.omnizip_units as omnizip_units  # type: ignore[import-not-found]
from omnizip.modeling_qwen2_5_omni import (  # type: ignore[import-not-found]
    Qwen2_5OmniForConditionalGeneration,
)
from transformers import Qwen2_5OmniProcessor

from videozip.cache.loader import load_video_scores
from videozip.src.videozip import omnizip_videozip


# ── Cached-video controller ──────────────────────────────────────────────────


class CachedVideoController:
    """Holds {video_key: 1D L6 score vector} and tracks the active video for the
    current generation call. Mirrors `CachedAudioController` from the audio eval."""

    def __init__(self, cache: dict[str, np.ndarray]):
        self.cache = cache
        self.current_key: str | None = None

    def set_context(self, file_key: str) -> None:
        self.current_key = file_key

    def get_current_scores(self) -> np.ndarray | None:
        if self.current_key is None:
            raise RuntimeError("CachedVideoController context not set before generation.")
        return self.cache.get(self.current_key)


# ── Monkey-patch installer ───────────────────────────────────────────────────


def install_videozip_patch(
    controller: CachedVideoController,
    *,
    video_saliency_source: str,
    audio_anchor_beta: float,
):
    """Replace `omnizip_units.omnizip` with a wrapper that dispatches to
    `omnizip_videozip` and feeds in cached video scores. Returns the original
    function so the caller can restore it."""
    original = omnizip_units.omnizip

    def wrapped(
        input_embeds,
        attn_logits,
        input_ids,
        audio_token_id,
        video_token_id,
        num_input_frames,
        merging_ratio_audio=0.5,
        merging_ratio_v=0.5,
        contextual_ratio=0.05,
        g=3,
    ):
        l6_scores = None
        if video_saliency_source == "l6_cached":
            arr = controller.get_current_scores()
            if arr is None:
                raise KeyError(
                    f"No cached video saliency for key={controller.current_key!r}"
                )
            l6_scores = torch.as_tensor(arr, device=input_embeds.device, dtype=torch.float32)
        return omnizip_videozip(
            input_embeds=input_embeds,
            attn_logits=attn_logits,
            input_ids=input_ids,
            audio_token_id=audio_token_id,
            video_token_id=video_token_id,
            num_input_frames=num_input_frames,
            merging_ratio_audio=merging_ratio_audio,
            merging_ratio_v=merging_ratio_v,
            contextual_ratio=contextual_ratio,
            g=g,
            l6_video_scores=l6_scores,
            video_saliency_source=video_saliency_source,
            audio_anchor_beta=audio_anchor_beta,
        )

    omnizip_units.omnizip = wrapped
    return original


# ── Helpers ──────────────────────────────────────────────────────────────────


def normalize_path_for_match(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def resolve_model_dtype(name: str):
    table = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in table:
        raise ValueError(f"Unsupported dtype {name!r}")
    return table[name]


def load_model(
    *,
    model_path: str,
    rho_audio: float,
    rho_video: float,
    g: int,
    contextual_ratio: float,
    dtype_name: str,
    device: str | None,
):
    dt = resolve_model_dtype(dtype_name)
    device_map = {"": device} if device else "auto"
    print(f"Loading {model_path} with VideoZip patch ...")
    print(
        f"  rho_audio={rho_audio} rho_video={rho_video} g={g} "
        f"contextual_ratio={contextual_ratio} dtype={dtype_name} device={device or 'auto'}"
    )
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dt,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model.thinker.omnizip_config = {
        "rho_audio": rho_audio,
        "rho_video": rho_video,
        "g": g,
        "contextual_ratio": contextual_ratio,
    }
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    base._record_model_loaded_vram()
    return model, processor


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="vizzing/layer_depth_all_full.jsonl")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument(
        "--video_saliency_source",
        default="l6_cached",
        choices=["l6_cached", "attn", "sim_only"],
        help="Where VideoZip reads its video saliency from. Default uses the existing cache.",
    )
    p.add_argument("--audio_anchor_beta", type=float, default=0.3,
                   help="Weight on audio-similarity term in audio-anchored ISTM (Ablation B).")
    p.add_argument("--device", default=None,
                   help="e.g. cuda:0. Recommended for OmniZip/VideoZip — multi-GPU sharding "
                        "has known rope_deltas issues.")
    p.add_argument("--model", default=None, help=f"Model path (or set {base.ENV_MODEL_PATH_KEY}).")
    p.add_argument("--metadata", default="videos/metadata.json")
    p.add_argument("--videos", default="videos")
    p.add_argument("--output", default="vizzing/results_videozip.jsonl")
    p.add_argument("--log", default="vizzing/eval_videozip.log")
    p.add_argument("--errors_log", default=None)
    p.add_argument("--category", default=None)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--max_pixels", type=int, default=360 * 420)
    p.add_argument("--max_frames_videomme", type=int, default=768)
    p.add_argument("--max_frames_other", type=int, default=128)
    p.add_argument("--no_audio", action="store_true")
    p.add_argument("--rho_audio", type=float, default=base.OMNIZIP_DEFAULT_RHO_AUDIO)
    p.add_argument("--rho_video", type=float, default=base.OMNIZIP_DEFAULT_RHO_VIDEO)
    p.add_argument("--g", type=int, default=base.OMNIZIP_DEFAULT_G)
    p.add_argument("--contextual_ratio", type=float, default=base.OMNIZIP_DEFAULT_CONTEXTUAL_RATIO)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--vram_log", default="vizzing/vram_videozip.jsonl")
    p.add_argument("--stderr_log", default=None)
    p.add_argument("--measure_prefill", action="store_true")
    args = p.parse_args()

    base.set_run_seed(args.seed)

    model_path = args.model or os.environ.get(base.ENV_MODEL_PATH_KEY) or base.DEFAULT_MODEL_PATH

    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not in PATH. MP4 audio decode needs ffmpeg (or use --no_audio).")

    # Cache load is needed only for l6_cached; warn but allow other sources without it.
    cached_video_scores: dict[str, np.ndarray] = {}
    if args.video_saliency_source == "l6_cached":
        cached_video_scores = load_video_scores(args.cache, layer=args.layer, key="file")
        if not cached_video_scores:
            raise RuntimeError(
                f"No cached VIDEO saliency in {args.cache} at layer {args.layer}. "
                f"Run videozip/precompute/validate_l6_cache.py to diagnose."
            )

    controller = CachedVideoController(cached_video_scores)
    original_omnizip = install_videozip_patch(
        controller,
        video_saliency_source=args.video_saliency_source,
        audio_anchor_beta=args.audio_anchor_beta,
    )

    errors_log_path = args.errors_log or os.path.join(
        os.path.dirname(args.log) or ".", "errors_videozip.log"
    )
    for path in (args.log, args.output, args.vram_log, errors_log_path,
                 args.stderr_log if args.stderr_log else None):
        if not path:
            continue
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if args.stderr_log:
        stderr_f = open(args.stderr_log, "a", encoding="utf-8")
        sys.stderr = base.StderrTee(stderr_f, sys.__stderr__)

    tee = base.Tee(args.log)
    sys.stdout = tee

    base.RUN_CONFIG = {
        "fps": args.fps,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_frames_videomme": args.max_frames_videomme,
        "max_frames_other": args.max_frames_other,
    }

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} entries from {args.metadata}")
    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")
    runnable = [e for e in meta if e.get("questions")]
    print(f"Running {len(runnable)} entries with VideoZip")
    print(
        f"Saliency source: {args.video_saliency_source}  cache={args.cache}  "
        f"layer={args.layer}  cached_videos={len(cached_video_scores)}"
    )
    print(
        f"rho_audio={args.rho_audio} rho_video={args.rho_video} g={args.g} "
        f"audio_anchor_beta={args.audio_anchor_beta}\n"
    )
    if not runnable:
        print("Nothing to run. Exiting.")
        return 0

    model, processor = load_model(
        model_path=model_path,
        rho_audio=args.rho_audio,
        rho_video=args.rho_video,
        g=args.g,
        contextual_ratio=args.contextual_ratio,
        dtype_name=args.dtype,
        device=args.device,
    )

    correct = total = skipped_no_video = skipped_no_cache = 0
    results: list[dict] = []

    try:
        with open(args.output, "w", encoding="utf-8") as out_f, \
             open(args.vram_log, "w", encoding="utf-8") as vram_f:
            for entry in runnable:
                video_path = base.resolve_video_path(entry["file"], args.videos)
                entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
                if video_path is None:
                    print(f"  SKIP {entry_label}: video not found")
                    skipped_no_video += 1
                    continue

                file_key = normalize_path_for_match(entry["file"])
                if (
                    args.video_saliency_source == "l6_cached"
                    and file_key not in cached_video_scores
                ):
                    print(f"  SKIP {entry_label}: no cached video saliency for {entry['file']}")
                    skipped_no_cache += len(entry["questions"])
                    continue

                use_audio = (not args.no_audio) and base.check_video_has_audio(video_path)
                if (not args.no_audio) and (not use_audio):
                    print(f"  INFO {entry_label}: no audio stream; video+text only.")

                controller.set_context(file_key)

                for q_idx, q in enumerate(entry["questions"]):
                    question = q["question"]
                    choices = q["choices"]
                    answer = q["answer"].strip().upper()
                    task_type = q.get("task_type", entry.get("task_type", ""))
                    dataset = entry.get("dataset", "")

                    before_alloc, before_reserved = base._capture_current_vram_gb()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                    try:
                        pred, reasoning, orig_nf, used_nf, timing = base.run_inference(
                            model, processor, video_path, dataset, question, choices,
                            use_audio, measure_prefill=args.measure_prefill,
                        )
                        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
                        curr_alloc = torch.cuda.memory_allocated() / 1024**3
                        curr_reserved = torch.cuda.memory_reserved() / 1024**3
                        vram_entry = {
                            "entry": entry_label,
                            "task_type": task_type,
                            "status": "ok",
                            "method": "videozip",
                            "video_saliency_source": args.video_saliency_source,
                            "cache_layer": args.layer,
                            "audio_anchor_beta": args.audio_anchor_beta,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "duration_s": entry.get("duration_s"),
                            "orig_frames": orig_nf,
                            "used_frames": used_nf,
                            "model_loaded_alloc_gb": round(base.MODEL_LOADED_ALLOC_GB or 0.0, 2),
                            "model_loaded_reserved_gb": round(base.MODEL_LOADED_RESERVED_GB or 0.0, 2),
                            "before_alloc_gb": round(before_alloc, 2),
                            "before_reserved_gb": round(before_reserved, 2),
                            "peak_alloc_gb": round(peak_alloc, 2),
                            "peak_reserved_gb": round(peak_reserved, 2),
                            "after_alloc_gb": round(curr_alloc, 2),
                            "after_reserved_gb": round(curr_reserved, 2),
                            **timing,
                        }
                        vram_f.write(json.dumps(vram_entry) + "\n")
                        vram_f.flush()
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
                        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
                        curr_alloc, curr_reserved = base._capture_current_vram_gb()
                        vram_entry = {
                            "entry": entry_label,
                            "task_type": task_type,
                            "status": "error",
                            "method": "videozip",
                            "video_saliency_source": args.video_saliency_source,
                            "cache_layer": args.layer,
                            "audio_anchor_beta": args.audio_anchor_beta,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "duration_s": entry.get("duration_s"),
                            "orig_frames": 0,
                            "used_frames": 0,
                            "model_loaded_alloc_gb": round(base.MODEL_LOADED_ALLOC_GB or 0.0, 2),
                            "model_loaded_reserved_gb": round(base.MODEL_LOADED_RESERVED_GB or 0.0, 2),
                            "before_alloc_gb": round(before_alloc, 2),
                            "before_reserved_gb": round(before_reserved, 2),
                            "peak_alloc_gb": round(peak_alloc, 2),
                            "peak_reserved_gb": round(peak_reserved, 2),
                            "after_alloc_gb": round(curr_alloc, 2),
                            "after_reserved_gb": round(curr_reserved, 2),
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                        vram_f.write(json.dumps(vram_entry) + "\n")
                        vram_f.flush()
                        with open(errors_log_path, "a", encoding="utf-8") as ef:
                            ef.write(f"\n--- {entry_label} ---\n{tb}\n")
                        pred, reasoning = "ERROR", str(e)
                        orig_nf = used_nf = 0
                        timing = {}

                    is_correct = pred.strip().upper() == answer
                    if is_correct:
                        correct += 1
                    total += 1

                    result = {
                        "model_variant": base.MODEL_VARIANT,
                        "method": "videozip",
                        "video_saliency_source": args.video_saliency_source,
                        "cache_layer": args.layer,
                        "audio_anchor_beta": args.audio_anchor_beta,
                        "saliency_cache_path": args.cache,
                        "dataset": entry.get("dataset"),
                        "task_type": task_type,
                        "duration_s": entry.get("duration_s"),
                        "question": question,
                        "choices": choices,
                        "answer": answer,
                        "prediction": pred,
                        "correct": is_correct,
                        "reasoning": reasoning,
                        "orig_frames": orig_nf,
                        "used_frames": used_nf,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        **timing,
                    }
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    results.append(result)

                    status = "OK" if is_correct else "X"
                    print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer}")
    finally:
        omnizip_units.omnizip = original_omnizip

    acc = correct / total if total else 0.0
    print("\n" + "=" * 50)
    print(f"Method:   videozip (saliency={args.video_saliency_source}, beta={args.audio_anchor_beta})")
    print(f"Cache:    {args.cache}  layer={args.layer}")
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Skipped (no cache): {skipped_no_cache}")
    print(f"Results: {args.output}")

    by_dataset: dict[str, dict[str, int]] = {}
    for r in results:
        ds = r["dataset"]
        slot = by_dataset.setdefault(ds, {"correct": 0, "total": 0})
        slot["total"] += 1
        if r["correct"]:
            slot["correct"] += 1
    print("\nPer-dataset:")
    for ds, s in sorted(by_dataset.items()):
        acc_ds = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"  {ds:<20} {s['correct']}/{s['total']} = {acc_ds:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
