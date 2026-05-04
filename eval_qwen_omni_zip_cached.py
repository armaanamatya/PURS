"""
eval_qwen_omni_zip_cached.py

Benchmark Qwen2.5-Omni-7B + OmniZip while replacing OmniZip's audio importance
score with a cached external saliency vector (for example Layer-6 Thinker
cross-modal Q·K scores saved in layer_depth_stats.jsonl).

Default behavior matches the intended "cache once per video" setting:
we average the saved per-question saliency vectors into one vector per video and
reuse that same vector for every question on the video.

Typical usage:
    CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_zip_cached.py \
      --scores vizzing/layer_depth_experiment_v2/worldsense/layer_depth_stats.jsonl \
      --layer 6 \
      --cache_reduce mean \
      --metadata videos/metadata.json \
      --videos videos \
      --category worldsense \
      --output vizzing/results_zip_cached_l6_worldsense.jsonl \
      --log vizzing/results_zip_cached_l6_worldsense.log \
      --device cuda:0
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

import eval_qwen_omni_zip as base
import omnizip.omnizip_units as omnizip_units
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor


def normalize_path_for_match(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def normalize_dataset_name(dataset: str | None) -> str:
    return (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")


def resolve_model_dtype(dtype_name: str):
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in table:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return table[dtype_name]


def load_cached_audio_scores(path: str, layer: int, dataset: str | None, cache_reduce: str) -> dict[str, dict]:
    wanted_dataset = normalize_dataset_name(dataset) if dataset else None
    cache: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if wanted_dataset and normalize_dataset_name(rec.get("dataset")) != wanted_dataset:
                continue
            raw = rec.get("raw_audio_scores", {}).get(str(layer))
            if not raw:
                continue
            per_question = [np.asarray(v, dtype=np.float32) for v in raw]
            if not per_question:
                continue

            key = normalize_path_for_match(rec["file"])
            item = {
                "dataset": rec.get("dataset"),
                "file": rec["file"],
                "video_id": rec.get("video_id"),
                "per_question": per_question,
            }
            if cache_reduce == "mean":
                item["reduced"] = np.mean(np.stack(per_question, axis=0), axis=0).astype(np.float32)
            elif cache_reduce == "median":
                item["reduced"] = np.median(np.stack(per_question, axis=0), axis=0).astype(np.float32)
            elif cache_reduce == "first":
                item["reduced"] = per_question[0]
            else:
                raise ValueError(f"Unsupported cache_reduce {cache_reduce!r}")
            cache[key] = item
    return cache


class CachedAudioController:
    def __init__(self, cache: dict[str, dict], strategy: str):
        self.cache = cache
        self.strategy = strategy
        self.current_key: str | None = None
        self.current_question_idx: int | None = None

    def set_context(self, file_key: str, question_idx: int) -> None:
        self.current_key = file_key
        self.current_question_idx = question_idx

    def get_current_scores(self) -> np.ndarray:
        if self.current_key is None or self.current_question_idx is None:
            raise RuntimeError("Cached audio controller context was not set before generation.")
        item = self.cache.get(self.current_key)
        if item is None:
            raise KeyError(f"No cached audio saliency found for {self.current_key}")

        if self.strategy == "question_idx":
            q_idx = self.current_question_idx
            if q_idx >= len(item["per_question"]):
                raise IndexError(
                    f"Question index {q_idx} out of range for cached saliency on {self.current_key} "
                    f"(have {len(item['per_question'])} cached questions)"
                )
            return item["per_question"][q_idx]
        return item["reduced"]


def install_cached_audio_patch(controller: CachedAudioController):
    original_omnizip = omnizip_units.omnizip

    def wrapped_omnizip(
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
        external_scores = controller.get_current_scores()
        replacement = torch.as_tensor(external_scores, device=attn_logits.device, dtype=torch.float32)
        return original_omnizip(
            input_embeds=input_embeds,
            attn_logits=replacement,
            input_ids=input_ids,
            audio_token_id=audio_token_id,
            video_token_id=video_token_id,
            num_input_frames=num_input_frames,
            merging_ratio_audio=merging_ratio_audio,
            merging_ratio_v=merging_ratio_v,
            contextual_ratio=contextual_ratio,
            g=g,
        )

    omnizip_units.omnizip = wrapped_omnizip
    return original_omnizip


def load_model(
    model_path: str,
    rho_audio: float,
    rho_video: float,
    g: int,
    contextual_ratio: float,
    dtype_name: str,
    quantization: str,
    device: str | None,
):
    if quantization != "none":
        raise NotImplementedError(
            "Quantized + cached-audio OmniZip is not wired into this script yet. "
            "Use non-quantized benchmark runs first."
        )

    dt = resolve_model_dtype(dtype_name)
    device_map = {"": device} if device else "auto"

    print(f"Loading model from {model_path} with cached-audio OmniZip ...")
    print(
        f"  rho_audio={rho_audio}  rho_video={rho_video}  g={g}  "
        f"contextual_ratio={contextual_ratio}  torch_dtype={dtype_name}  device={device or 'auto'}"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="layer_depth_stats.jsonl containing raw_audio_scores")
    parser.add_argument("--layer", type=int, default=6, choices=[6, 14], help="Which cached layer to use")
    parser.add_argument(
        "--cache_reduce",
        default="mean",
        choices=["mean", "median", "first", "question_idx"],
        help="How to reduce saved per-question saliency vectors into the cached vector used at eval time. "
             "'mean' is the intended cache-once-per-video setting.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force the entire model onto one device, e.g. cuda:0. Recommended for OmniZip due to "
             "known rope_deltas issues with multi-GPU sharding.",
    )
    parser.add_argument("--model", default=None, help=f"Model path (or set {base.ENV_MODEL_PATH_KEY}).")
    parser.add_argument("--metadata", default="/workspace/metadata.json")
    parser.add_argument("--videos", default="/workspace/videos")
    parser.add_argument("--output", default="/workspace/results_zip_cached.jsonl")
    parser.add_argument("--log", default="/workspace/eval_zip_cached.log")
    parser.add_argument("--errors_log", default=None)
    parser.add_argument("--category", default=None, help="Filter by dataset or task_type")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--max_frames_videomme", type=int, default=768)
    parser.add_argument("--max_frames_other", type=int, default=128)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--rho_audio", type=float, default=base.OMNIZIP_DEFAULT_RHO_AUDIO)
    parser.add_argument("--rho_video", type=float, default=base.OMNIZIP_DEFAULT_RHO_VIDEO)
    parser.add_argument("--g", type=int, default=base.OMNIZIP_DEFAULT_G)
    parser.add_argument("--contextual_ratio", type=float, default=base.OMNIZIP_DEFAULT_CONTEXTUAL_RATIO)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--quantization", default="none", choices=["none", "gptq", "awq"])
    parser.add_argument("--vram_log", default="/workspace/vram_log_cached.jsonl")
    parser.add_argument("--stderr_log", default=None)
    parser.add_argument("--measure_prefill", action="store_true")
    args = parser.parse_args()

    base.set_run_seed(args.seed)

    model_path = args.model or os.environ.get(base.ENV_MODEL_PATH_KEY) or base.DEFAULT_MODEL_PATH
    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print(
            "WARNING: ffmpeg not found in PATH. Decoding MP4 audio needs ffmpeg (or use --no_audio)."
        )

    cached_scores = load_cached_audio_scores(args.scores, args.layer, args.category, args.cache_reduce if args.cache_reduce != "question_idx" else "mean")
    if not cached_scores:
        raise RuntimeError(
            f"No cached audio saliency entries found in {args.scores} for layer={args.layer} "
            f"and category={args.category!r}"
        )

    controller = CachedAudioController(cached_scores, args.cache_reduce)
    original_omnizip = install_cached_audio_patch(controller)

    errors_log_path = args.errors_log or os.path.join(os.path.dirname(args.log) or ".", "errors_cached.log")
    paths_for_dirs = [args.log, args.output, args.vram_log, errors_log_path]
    if args.stderr_log is not None:
        paths_for_dirs.append(args.stderr_log)
    for path in paths_for_dirs:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if args.stderr_log is not None:
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
    print(f"Loaded {len(meta)} entries")

    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    runnable = [e for e in meta if e.get("questions")]
    print(f"Running {len(runnable)} entries with cached-audio OmniZip")
    print(
        f"Cache source: {args.scores}  layer={args.layer}  cache_reduce={args.cache_reduce}  "
        f"cached_videos={len(cached_scores)}"
    )
    print(
        f"Input mode: {'video+audio+text' if not args.no_audio else 'video+text (--no_audio)'}; "
        f"sample_fps={args.fps}; max_pixels={args.max_pixels}; max_new_tokens={args.max_new_tokens}; "
        f"temperature={args.temperature}; seed={args.seed}\n"
    )

    if not runnable:
        print("Nothing to run. Exiting.")
        return

    model, processor = load_model(
        model_path=model_path,
        rho_audio=args.rho_audio,
        rho_video=args.rho_video,
        g=args.g,
        contextual_ratio=args.contextual_ratio,
        dtype_name=args.dtype,
        quantization=args.quantization,
        device=args.device,
    )

    correct = total = skipped_no_video = skipped_no_cache = 0
    results = []

    try:
        with open(args.output, "w", encoding="utf-8") as out_f, open(args.vram_log, "w", encoding="utf-8") as vram_f:
            for entry in runnable:
                video_path = base.resolve_video_path(entry["file"], args.videos)
                entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
                if video_path is None:
                    print(f"  SKIP {entry_label}: video not found")
                    skipped_no_video += 1
                    continue

                file_key = normalize_path_for_match(entry["file"])
                if file_key not in cached_scores:
                    print(f"  SKIP {entry_label}: no cached saliency for {entry['file']}")
                    skipped_no_cache += len(entry["questions"])
                    continue

                use_audio = (not args.no_audio) and base.check_video_has_audio(video_path)
                if (not args.no_audio) and (not use_audio):
                    print(f"  INFO {entry_label}: no audio stream detected; using video+text input only.")

                for q_idx, q in enumerate(entry["questions"]):
                    question = q["question"]
                    choices = q["choices"]
                    answer = q["answer"].strip().upper()
                    task_type = q.get("task_type", entry.get("task_type", ""))
                    dataset = entry.get("dataset", "")

                    controller.set_context(file_key, q_idx)
                    before_alloc_gb, before_reserved_gb = base._capture_current_vram_gb()

                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                    try:
                        pred, reasoning, orig_nf, used_nf, timing = base.run_inference(
                            model,
                            processor,
                            video_path,
                            dataset,
                            question,
                            choices,
                            use_audio,
                            measure_prefill=args.measure_prefill,
                        )
                        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
                        curr_alloc_gb = torch.cuda.memory_allocated() / 1024**3
                        curr_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                        vram_entry = {
                            "entry": entry_label,
                            "task_type": task_type,
                            "status": "ok",
                            "method": "omnizip_cached_audio",
                            "cache_layer": args.layer,
                            "cache_reduce": args.cache_reduce,
                            "quantization": args.quantization,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "duration_s": entry.get("duration_s"),
                            "orig_frames": orig_nf,
                            "used_frames": used_nf,
                            "model_loaded_alloc_gb": round(base.MODEL_LOADED_ALLOC_GB or 0.0, 2),
                            "model_loaded_reserved_gb": round(base.MODEL_LOADED_RESERVED_GB or 0.0, 2),
                            "before_alloc_gb": round(before_alloc_gb, 2),
                            "before_reserved_gb": round(before_reserved_gb, 2),
                            "peak_alloc_gb": round(peak_alloc_gb, 2),
                            "peak_reserved_gb": round(peak_reserved_gb, 2),
                            "after_alloc_gb": round(curr_alloc_gb, 2),
                            "after_reserved_gb": round(curr_reserved_gb, 2),
                            **timing,
                        }
                        vram_f.write(json.dumps(vram_entry) + "\n")
                        vram_f.flush()
                    except Exception as e:
                        import traceback

                        tb = traceback.format_exc()
                        print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
                        curr_alloc_gb, curr_reserved_gb = base._capture_current_vram_gb()
                        vram_entry = {
                            "entry": entry_label,
                            "task_type": task_type,
                            "status": "error",
                            "method": "omnizip_cached_audio",
                            "cache_layer": args.layer,
                            "cache_reduce": args.cache_reduce,
                            "quantization": args.quantization,
                            "temperature": args.temperature,
                            "seed": args.seed,
                            "duration_s": entry.get("duration_s"),
                            "orig_frames": 0,
                            "used_frames": 0,
                            "model_loaded_alloc_gb": round(base.MODEL_LOADED_ALLOC_GB or 0.0, 2),
                            "model_loaded_reserved_gb": round(base.MODEL_LOADED_RESERVED_GB or 0.0, 2),
                            "before_alloc_gb": round(before_alloc_gb, 2),
                            "before_reserved_gb": round(before_reserved_gb, 2),
                            "peak_alloc_gb": round(peak_alloc_gb, 2),
                            "peak_reserved_gb": round(peak_reserved_gb, 2),
                            "after_alloc_gb": round(curr_alloc_gb, 2),
                            "after_reserved_gb": round(curr_reserved_gb, 2),
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
                        "quantization": args.quantization,
                        "dataset": entry.get("dataset"),
                        "task_type": task_type,
                        "duration_s": entry.get("duration_s"),
                        "question": question,
                        "choices": choices,
                        "answer": answer,
                        "prediction": pred,
                        "correct": is_correct,
                        "reasoning": reasoning,
                        "method": "omnizip_cached_audio",
                        "cache_layer": args.layer,
                        "cache_reduce": args.cache_reduce,
                        "saliency_scores_path": args.scores,
                        "orig_frames": orig_nf,
                        "used_frames": used_nf,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        **timing,
                    }
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                    results.append(result)

                    status = "✓" if is_correct else "✗"
                    print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer}")
    finally:
        omnizip_units.omnizip = original_omnizip

    acc = correct / total if total else 0.0
    print(f"\n{'=' * 50}")
    print(f"Model variant: {base.MODEL_VARIANT} + omnizip_cached_audio")
    print(f"Quantization:  {args.quantization}")
    print(f"Cache layer:   {args.layer} ({args.cache_reduce})")
    print(f"Accuracy:      {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Skipped (no cache): {skipped_no_cache}")
    print(f"Results: {args.output}")

    datasets: dict[str, dict[str, int]] = {}
    for r in results:
        ds = r["dataset"]
        datasets.setdefault(ds, {"correct": 0, "total": 0})
        datasets[ds]["total"] += 1
        if r["correct"]:
            datasets[ds]["correct"] += 1
    print("\nPer-dataset:")
    for ds, stats in sorted(datasets.items()):
        print(f"  {ds:<20} {stats['correct']}/{stats['total']} = {stats['correct'] / stats['total']:.2%}")


if __name__ == "__main__":
    main()
