"""Evaluate Qwen2.5-Omni-7B with experimental OmniDrop decoder pruning.

This runner reuses the existing OmniZip Qwen model copy because it already
contains the local Qwen2.5-Omni implementation. Unlike OmniZip, compression is
not applied before the LLM. Instead, ``modeling_qwen2_5_omni.py`` reads
``model.thinker.omnidrop_config`` and prunes audiovisual hidden states between
decoder layers during the prefill forward pass.

Status: experimental integration for paper-faithful layer-wise behavior and
retention diagnostics. Use the full-token and OmniZip runners as controls.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import Qwen2_5OmniProcessor

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIDROP_DIR = os.path.join(_REPO_ROOT, "omnidrop")
if OMNIDROP_DIR not in sys.path:
    sys.path.insert(0, OMNIDROP_DIR)

import eval_qwen_omni_zip as base
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration


MODEL_VARIANT = "qwen2.5-omni+omnidrop"


def load_model(model_path: str, cfg: dict, dtype_name: str):
    dt = base.resolve_model_dtype(dtype_name)
    print(f"Loading model from {model_path} with OmniDrop ...")
    print(f"  config={json.dumps(cfg, sort_keys=True)}")
    print(f"  torch_dtype={dtype_name}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dt,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.thinker.omnizip_config = None
    model.thinker.omnidrop_config = cfg

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    base._record_model_loaded_vram()
    return model, processor


def _trace_summary(trace: list[dict]) -> dict:
    if not trace:
        return {}
    pruned = sum(int(x.get("pruned", 0)) for x in trace)
    first = trace[0]
    last = trace[-1]
    av_start = int(first.get("av_before") or 0)
    av_end = int(last.get("av_after") or 0)
    av_before_ratios = [
        (float(x.get("av_before", 0)) / av_start)
        for x in trace
        if av_start > 0 and x.get("av_before") is not None
    ]
    return {
        "omnidrop_layers": len(trace),
        "omnidrop_pruned_total": pruned,
        "omnidrop_seq_start": first.get("seq_before"),
        "omnidrop_seq_end": last.get("seq_after"),
        "omnidrop_av_start": av_start,
        "omnidrop_av_end": av_end,
        "omnidrop_mean_av_retained": round(sum(av_before_ratios) / len(av_before_ratios), 6)
        if av_before_ratios
        else None,
        "omnidrop_final_av_retained": round(av_end / av_start, 6) if av_start > 0 else None,
        "omnidrop_trace": trace,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help=f"Model path (or set {base.ENV_MODEL_PATH_KEY}).")
    parser.add_argument("--metadata", default="/workspace/metadata.json")
    parser.add_argument("--videos", default="/workspace/videos")
    parser.add_argument("--output", default="/workspace/results_omnidrop.jsonl")
    parser.add_argument("--log", default="/workspace/eval_omnidrop.log")
    parser.add_argument("--errors_log", default=None)
    parser.add_argument("--category", default=None, help="Filter by dataset or task_type")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--max_frames_videomme", type=int, default=768)
    parser.add_argument("--max_frames_other", type=int, default=128)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vram_log", default="/workspace/vram_log_omnidrop.jsonl")
    parser.add_argument("--stderr_log", default=None)
    parser.add_argument("--measure_prefill", action="store_true")

    parser.add_argument("--p_init", type=float, default=0.0)
    parser.add_argument("--p_final", type=float, default=0.2)
    parser.add_argument("--t_mid", type=float, default=0.5)
    parser.add_argument("--schedule_beta", type=float, default=20.0)
    parser.add_argument("--lambda_div", type=float, default=0.2)
    parser.add_argument("--tds_start_layer", type=int, default=14)
    parser.add_argument("--no_pre_prune", action="store_true", help="Disable OmniDrop Sec. 3.4 pre-LLM pruning.")
    parser.add_argument("--audio_keep_ratio", type=float, default=0.7)
    parser.add_argument("--video_prune_rate", type=float, default=0.8)
    parser.add_argument("--video_group_size", type=int, default=4)
    parser.add_argument("--audio_tokens_per_chunk", type=int, default=50)
    parser.add_argument("--video_tokens_per_chunk", type=int, default=288)
    args = parser.parse_args()
    base.set_run_seed(args.seed)

    model_path = args.model or os.environ.get(base.ENV_MODEL_PATH_KEY) or base.DEFAULT_MODEL_PATH
    if not os.path.exists(model_path) and os.path.exists(base.FALLBACK_MODEL_PATH):
        model_path = base.FALLBACK_MODEL_PATH

    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found in PATH. Use --no_audio or install ffmpeg if MP4 audio decode fails.\n")

    errors_log_path = args.errors_log or os.path.join(os.path.dirname(args.log) or ".", "errors.log")
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
    print(f"Running {len(runnable)} entries with OmniDrop")
    print(
        "Each line in --output is written after one question finishes. "
        "OmniDrop trace summaries are written to both results and VRAM logs.\n"
    )
    if not runnable:
        print("Nothing to run. Exiting.")
        return

    omnidrop_cfg = {
        "enabled": True,
        "p_init": args.p_init,
        "p_final": args.p_final,
        "t_mid": args.t_mid,
        "beta": args.schedule_beta,
        "lambda_div": args.lambda_div,
        "tds_start_layer": args.tds_start_layer,
        "pre_prune": not args.no_pre_prune,
        "audio_keep_ratio": args.audio_keep_ratio,
        "video_prune_rate": args.video_prune_rate,
        "video_group_size": args.video_group_size,
        "audio_tokens_per_chunk": args.audio_tokens_per_chunk,
        "video_tokens_per_chunk": args.video_tokens_per_chunk,
    }
    model, processor = load_model(model_path, omnidrop_cfg, args.dtype)

    correct = total = skipped_no_video = 0
    results = []
    with open(args.output, "w", encoding="utf-8") as out_f, open(args.vram_log, "w", encoding="utf-8") as vram_f:
        for entry in runnable:
            video_path = base.resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video not found")
                skipped_no_video += 1
                continue

            use_audio = (not args.no_audio) and base.check_video_has_audio(video_path)
            if (not args.no_audio) and (not use_audio):
                print(f"  INFO {entry_label}: no audio stream detected; using video+text input only.")

            for q in entry["questions"]:
                question = q["question"]
                choices = q["choices"]
                answer = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset = entry.get("dataset", "")
                before_alloc_gb, before_reserved_gb = base._capture_current_vram_gb()

                try:
                    torch.cuda.reset_peak_memory_stats()
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
                    trace = list(getattr(model.thinker.model, "omnidrop_last_trace", []) or [])
                    trace_info = _trace_summary(trace)
                    pre_info = dict(getattr(model.thinker, "omnidrop_pre_prune_stats", {}) or {})
                    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
                    curr_alloc_gb = torch.cuda.memory_allocated() / 1024**3
                    curr_reserved_gb = torch.cuda.memory_reserved() / 1024**3
                    vram_entry = {
                        "entry": entry_label,
                        "task_type": task_type,
                        "status": "ok",
                        "method": "omnidrop",
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
                        **pre_info,
                        **trace_info,
                    }
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
                        "method": "omnidrop",
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
                    with open(errors_log_path, "a", encoding="utf-8") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{tb}\n")
                    pred, reasoning = "ERROR", str(e)
                    orig_nf = used_nf = 0
                    timing = {}
                    trace_info = {}
                    pre_info = {}

                vram_f.write(json.dumps(vram_entry) + "\n")
                vram_f.flush()

                is_correct = pred.strip().upper() == answer
                correct += int(is_correct)
                total += 1
                result = {
                    "model_variant": MODEL_VARIANT,
                    "dataset": entry.get("dataset"),
                    "task_type": task_type,
                    "duration_s": entry.get("duration_s"),
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "prediction": pred,
                    "correct": is_correct,
                    "reasoning": reasoning,
                    "method": "omnidrop",
                    "orig_frames": orig_nf,
                    "used_frames": used_nf,
                    "temperature": args.temperature,
                    "seed": args.seed,
                    **timing,
                    **pre_info,
                    **trace_info,
                }
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                results.append(result)

                status = "OK" if is_correct else "NO"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer} trace={trace_info}")

    acc = correct / total if total else 0
    print(f"\n{'=' * 50}")
    print(f"Model variant: {MODEL_VARIANT}")
    print(f"OmniDrop Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Results: {args.output}")

    datasets: dict = {}
    for r in results:
        d = r["dataset"]
        datasets.setdefault(d, {"correct": 0, "total": 0})
        datasets[d]["total"] += 1
        if r["correct"]:
            datasets[d]["correct"] += 1
    print("\nPer-dataset:")
    for ds, s in sorted(datasets.items()):
        print(f"  {ds:<20} {s['correct']}/{s['total']} = {s['correct'] / s['total']:.2%}")


if __name__ == "__main__":
    main()
