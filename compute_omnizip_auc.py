"""
Compute ROC AUC between saved layer-depth audio saliency scores and OmniZip's
actual audio keep mask.

This script:
1. Loads the layer-depth v2 JSONL with raw_audio_scores saved per question.
2. Re-runs the OmniZip thinker forward pass for each (video, question).
3. Captures OmniZip's global keep mask at runtime by wrapping omnizip().
4. Extracts the audio-token keep mask and compares it to saved layer scores.

Outputs:
  {out_dir}/omnizip_auc_questions.jsonl  - one record per (video, question)
  {out_dir}/omnizip_auc_summary.json     - aggregate summary
  {out_dir}/omnizip_auc_report.md        - readable report
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(REPO_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if OMNIZIP_DIR not in sys.path:
    sys.path.insert(0, OMNIZIP_DIR)
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
import omnizip.omnizip_units as omnizip_units
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor

from eval_qwen_omni_zip import (
    _batch_for_omni_generate,
    build_user_prompt_for_dataset,
    resolve_video_path,
)


ENV_MODEL_PATH_KEY = "QWEN_OMNI_MODEL_PATH"
DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"

THINKER_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lengths",
    "video_second_per_grid",
    "inputs_embeds",
    "position_ids",
    "past_key_values",
    "rope_deltas",
    "use_cache",
    "cache_position",
    "labels",
}


def safe_dir_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)[:120]


def normalize_path_for_match(path: str) -> str:
    return path.replace("\\", "/").strip().lower()


def normalize_dataset_name(dataset: str | None) -> str:
    return (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")


def roc_auc_score_from_ranks(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC using Mann-Whitney U with average ranks for ties."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(ranks[pos].sum())
    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_pos / (n_pos * n_neg))


def load_layer_scores(path: str, layer: int, dataset: str | None) -> list[dict]:
    wanted_dataset = normalize_dataset_name(dataset) if dataset else None
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if wanted_dataset and normalize_dataset_name(rec.get("dataset")) != wanted_dataset:
                continue
            raw_audio = rec.get("raw_audio_scores", {})
            if str(layer) not in raw_audio:
                continue
            rows.append(rec)
    return rows


def load_metadata_entries(path: str, dataset: str | None) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    entries = list(raw.values()) if isinstance(raw, dict) else raw
    wanted_dataset = normalize_dataset_name(dataset) if dataset else None
    out = {}
    for entry in entries:
        if wanted_dataset and normalize_dataset_name(entry.get("dataset")) != wanted_dataset:
            continue
        key = normalize_path_for_match(entry["file"])
        out[key] = entry
    return out


def resolve_model_dtype(dtype_name: str):
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in table:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return table[dtype_name]


def install_omnizip_mask_capture():
    capture = {}
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
        out_embeds, global_mask = original_omnizip(
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
        )
        capture["global_mask"] = global_mask.detach().cpu()
        capture["input_ids"] = input_ids.detach().cpu()
        capture["audio_token_id"] = int(audio_token_id)
        capture["video_token_id"] = int(video_token_id)
        return out_embeds, global_mask

    omnizip_units.omnizip = wrapped_omnizip
    return capture, original_omnizip


def prepare_generate_inputs(model, processor, video_path: str, dataset: str, question: str, choices: list, fps: float, max_pixels: int, max_frames: int):
    system_text = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )
    user_prompt = build_user_prompt_for_dataset(dataset, question, choices)
    video_element = {
        "type": "video",
        "video": video_path,
        "fps": fps,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
    }
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [video_element, {"type": "text", "text": user_prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    if not videos or videos[0] is None or getattr(videos[0], "shape", None) is None or videos[0].shape[0] <= 0:
        raise ValueError(f"Decoded 0 video frames for {video_path}")

    model.thinker.nframes = int(videos[0].shape[0])
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            inputs[key] = value.to(model.dtype)
    return inputs


def run_omnizip_mask(model, processor, capture: dict, video_path: str, dataset: str, question: str, choices: list, fps: float, max_pixels: int, max_frames: int) -> tuple[np.ndarray, int]:
    capture.clear()
    inputs = prepare_generate_inputs(
        model=model,
        processor=processor,
        video_path=video_path,
        dataset=dataset,
        question=question,
        choices=choices,
        fps=fps,
        max_pixels=max_pixels,
        max_frames=max_frames,
    )
    tokenizer = processor.tokenizer
    gen_in = _batch_for_omni_generate(inputs)
    with torch.no_grad():
        model.generate(
            **gen_in,
            use_audio_in_video=True,
            return_audio=False,
            thinker_max_new_tokens=1,
            thinker_do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    if "global_mask" not in capture or "input_ids" not in capture:
        raise RuntimeError("OmniZip mask capture failed; global_mask was not produced.")

    global_mask = capture["global_mask"].reshape(-1).numpy().astype(bool)
    input_ids = capture["input_ids"].reshape(-1).numpy()
    audio_token_id = capture["audio_token_id"]
    audio_positions = np.flatnonzero(input_ids == audio_token_id)
    audio_mask = global_mask[audio_positions]
    return audio_mask, int(audio_positions.size)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {"n": int(arr.size), "mean": float(arr.mean()), "std": float(arr.std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="layer_depth_stats.jsonl with raw_audio_scores")
    parser.add_argument("--metadata", default=os.path.join(REPO_ROOT, "videos", "metadata.json"))
    parser.add_argument("--videos", default=os.path.join(REPO_ROOT, "videos"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "vizzing", "omnizip_auc"))
    parser.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. worldsense")
    parser.add_argument("--layer", type=int, default=6, choices=[6, 14], help="Which saved saliency layer to compare")
    parser.add_argument("--model_path", default=os.environ.get(ENV_MODEL_PATH_KEY) or DEFAULT_MODEL_PATH)
    parser.add_argument("--rho_audio", type=float, default=0.3)
    parser.add_argument("--rho_video", type=float, default=0.6)
    parser.add_argument("--g", type=int, default=3)
    parser.add_argument("--contextual_ratio", type=float, default=0.05)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--device",
        default=None,
        help="Force the full model onto one device, e.g. cuda:0. "
             "Useful to avoid OmniZip rope_deltas bugs with multi-GPU device_map=auto.",
    )
    parser.add_argument("--limit_videos", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    layer_rows = load_layer_scores(args.scores, args.layer, args.dataset)
    if args.limit_videos is not None:
        layer_rows = layer_rows[: args.limit_videos]
    metadata_by_file = load_metadata_entries(args.metadata, args.dataset)

    print(f"Loading model from {args.model_path}")
    device_map = {"": args.device} if args.device else "auto"
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=resolve_model_dtype(args.dtype),
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.thinker.omnizip_config = {
        "rho_audio": args.rho_audio,
        "rho_video": args.rho_video,
        "g": args.g,
        "contextual_ratio": args.contextual_ratio,
    }
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    capture, original_omnizip = install_omnizip_mask_capture()
    question_records = []
    skipped = []

    try:
        for row_idx, row in enumerate(layer_rows, start=1):
            file_key = normalize_path_for_match(row["file"])
            meta_entry = metadata_by_file.get(file_key)
            if meta_entry is None:
                skipped.append({"file": row["file"], "reason": "metadata_not_found"})
                continue

            raw_scores_per_question = row.get("raw_audio_scores", {}).get(str(args.layer), [])
            questions = meta_entry.get("questions", [])
            if len(raw_scores_per_question) != len(questions):
                skipped.append(
                    {
                        "file": row["file"],
                        "reason": "question_count_mismatch",
                        "saved_questions": len(raw_scores_per_question),
                        "metadata_questions": len(questions),
                    }
                )
                continue

            video_path = resolve_video_path(row["file"], args.videos)
            if video_path is None:
                skipped.append({"file": row["file"], "reason": "video_not_found"})
                continue

            print(f"[{row_idx}/{len(layer_rows)}] {row.get('video_id', Path(row['file']).stem)}")

            for q_idx, (q_meta, raw_scores) in enumerate(zip(questions, raw_scores_per_question)):
                audio_keep_mask, n_audio = run_omnizip_mask(
                    model=model,
                    processor=processor,
                    capture=capture,
                    video_path=video_path,
                    dataset=meta_entry.get("dataset", row.get("dataset", "")),
                    question=q_meta["question"],
                    choices=q_meta.get("choices", []),
                    fps=args.fps,
                    max_pixels=args.max_pixels,
                    max_frames=args.max_frames,
                )

                scores = np.asarray(raw_scores, dtype=np.float64)
                if scores.shape[0] != audio_keep_mask.shape[0]:
                    skipped.append(
                        {
                            "file": row["file"],
                            "question_idx": q_idx,
                            "reason": "audio_length_mismatch",
                            "saved_len": int(scores.shape[0]),
                            "mask_len": int(audio_keep_mask.shape[0]),
                        }
                    )
                    continue

                y_keep = audio_keep_mask.astype(np.int64)
                auc_keep = roc_auc_score_from_ranks(y_keep, scores)
                auc_drop = roc_auc_score_from_ranks(1 - y_keep, scores)

                question_records.append(
                    {
                        "video_id": row.get("video_id", safe_dir_name(Path(row["file"]).stem)),
                        "dataset": row.get("dataset"),
                        "file": row["file"],
                        "question_idx": q_idx,
                        "question": q_meta["question"],
                        "layer": args.layer,
                        "n_audio_tokens": int(n_audio),
                        "omnizip_keep_rate": float(y_keep.mean()),
                        "auc_keep": float(auc_keep),
                        "auc_drop": float(auc_drop),
                    }
                )
                print(
                    f"  Q{q_idx}: auc_keep={auc_keep:.4f}  "
                    f"auc_drop={auc_drop:.4f}  keep_rate={y_keep.mean():.4f}"
                )
    finally:
        omnizip_units.omnizip = original_omnizip

    q_out = os.path.join(args.out_dir, "omnizip_auc_questions.jsonl")
    with open(q_out, "w", encoding="utf-8") as fh:
        for rec in question_records:
            fh.write(json.dumps(rec) + "\n")

    by_dataset: dict[str, list[float]] = {}
    for rec in question_records:
        by_dataset.setdefault(rec["dataset"] or "unknown", []).append(rec["auc_keep"])

    summary = {
        "layer": args.layer,
        "rho_audio": args.rho_audio,
        "rho_video": args.rho_video,
        "g": args.g,
        "contextual_ratio": args.contextual_ratio,
        "n_question_records": len(question_records),
        "overall_auc_keep": summarize([rec["auc_keep"] for rec in question_records]),
        "overall_auc_drop": summarize([rec["auc_drop"] for rec in question_records]),
        "by_dataset_auc_keep": {ds: summarize(vals) for ds, vals in sorted(by_dataset.items())},
        "skipped": skipped,
    }

    summary_path = os.path.join(args.out_dir, "omnizip_auc_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    report_path = os.path.join(args.out_dir, "omnizip_auc_report.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"# OmniZip AUC Report (Layer {args.layer})\n\n")
        fh.write(f"- Question records: {len(question_records)}\n")
        fh.write(
            f"- Overall AUC(keep): {summary['overall_auc_keep']['mean']:.4f} "
            f"+/- {summary['overall_auc_keep']['std']:.4f}\n"
        )
        fh.write(
            f"- Overall AUC(drop): {summary['overall_auc_drop']['mean']:.4f} "
            f"+/- {summary['overall_auc_drop']['std']:.4f}\n\n"
        )
        fh.write("## By Dataset\n\n")
        for ds, ds_summary in sorted(summary["by_dataset_auc_keep"].items()):
            fh.write(
                f"- {ds}: {ds_summary['mean']:.4f} +/- {ds_summary['std']:.4f} "
                f"(n={ds_summary['n']})\n"
            )
        if skipped:
            fh.write("\n## Skipped\n\n")
            for item in skipped:
                fh.write(f"- {json.dumps(item)}\n")

    print("\nSaved:")
    print(f"  Questions: {q_out}")
    print(f"  Summary:   {summary_path}")
    print(f"  Report:    {report_path}")
    print(
        "\nOverall: "
        f"AUC(keep)={summary['overall_auc_keep']['mean']:.4f} +/- {summary['overall_auc_keep']['std']:.4f}  "
        f"AUC(drop)={summary['overall_auc_drop']['mean']:.4f} +/- {summary['overall_auc_drop']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
