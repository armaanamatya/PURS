"""
eval_qwen_omni.py
Runs Qwen2.5-Omni-7B on all videos in metadata.json and answers MCQ questions.
Reads Q&A from metadata.json — no HuggingFace needed on the server.

Usage:
    python eval_qwen_omni.py --metadata metadata.json --videos /workspace/videos --output results.jsonl
    python eval_qwen_omni.py --metadata metadata.json --videos /workspace/videos --output results.jsonl --category lecture
"""

import argparse
import json
import os
import glob
import sys
import torch
from datetime import datetime
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

OMNIZIP_DIR = os.path.join(os.path.dirname(__file__), "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)

from qwen_omni_utils import process_mm_info


class Tee:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*60}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.log.close()

ENV_MODEL_PATH_KEY = "QWEN_OMNI_MODEL_PATH"
DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
FALLBACK_MODEL_PATH = "/workspace/model"

MODEL_PATH = os.environ.get(ENV_MODEL_PATH_KEY) or DEFAULT_MODEL_PATH

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_variant: str):
    print(f"Loading {model_variant} model from {MODEL_PATH} ...")
    if model_variant == "qwen2.5-omni":
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    elif model_variant == "qwen3-omni":
        try:
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as e:
            raise ImportError(
                "Qwen3 Omni classes not found in your installed transformers. "
                "Install a transformers version that includes Qwen3-Omni support, "
                "or run with --model_variant qwen2.5-omni."
            ) from e

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    else:
        raise ValueError(f"Unsupported model variant: {model_variant}")
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    processor,
    video_path: str,
    question: str,
    choices: list,
    fps: float,
    max_pixels: int,
    max_new_tokens: int,
) -> tuple[str, str]:
    choice_text = "\n".join(choices) if choices[0].startswith("A") else "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    prompt = f"{question}\n\n{choice_text}\n\nThink step by step, then end your response with 'Answer: X' where X is A, B, C, or D."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, max_new_tokens=max_new_tokens)

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    # Strip the prompt portion that gets echoed back
    if "assistant" in decoded.lower():
        decoded = decoded[decoded.lower().rfind("assistant") + len("assistant"):].strip()

    # Parse answer letter — prefer "Answer: X" pattern, fall back to last ABCD char
    import re as _re
    m = _re.search(r"Answer:\s*([A-D])", decoded, _re.IGNORECASE)
    letter = m.group(1).upper() if m else next((c for c in reversed(decoded.strip()) if c in "ABCD"), decoded.strip())
    return letter, decoded

# ── Video lookup ──────────────────────────────────────────────────────────────

def resolve_video_path(file_field: str, videos_dir: str) -> str | None:
    """Try the stored path, then search videos_dir by filename."""
    # Try stored path directly (works if running locally)
    if os.path.exists(file_field):
        return file_field

    # Normalize Windows backslashes → forward slashes, then get filename
    normalized = file_field.replace("\\", "/")
    filename = normalized.split("/")[-1]          # e.g. soccer_matches_1.mp4
    stem = filename.rsplit(".", 1)[0]             # e.g. soccer_matches_1

    # Try flat in videos_dir
    candidate = os.path.join(videos_dir, filename)
    if os.path.exists(candidate):
        return candidate

    # Recursive search in videos_dir
    for ext in ("mp4", "mkv", "webm", "avi"):
        matches = glob.glob(os.path.join(videos_dir, "**", f"{stem}.{ext}"), recursive=True)
        if matches:
            return matches[0]

    return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=None, help=f"Model path (or set {ENV_MODEL_PATH_KEY}).")
    parser.add_argument("--metadata", default="metadata.json", help="Path to enriched metadata.json")
    parser.add_argument("--videos",   default="/workspace/videos", help="Directory containing video files")
    parser.add_argument("--output",   default="/workspace/results.jsonl", help="Output JSONL file")
    parser.add_argument("--log",      default="/workspace/eval.log", help="Log file (appended each run)")
    parser.add_argument("--category", default=None, help="Only run this category (e.g. lecture)")
    parser.add_argument("--fps",      type=float, default=0.5, help="Video sampling fps (lower = less VRAM)")
    parser.add_argument("--max_pixels", type=int, default=256*256, help="Max pixels per frame (lower = less VRAM)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Generation length cap (lower = less VRAM)")
    parser.add_argument(
        "--model_variant",
        default="qwen2.5-omni",
        choices=["qwen2.5-omni", "qwen3-omni"],
        help="Which Omni model variant to run",
    )
    args = parser.parse_args()

    global MODEL_PATH
    if args.model:
        MODEL_PATH = args.model
    elif not os.path.exists(MODEL_PATH) and os.path.exists(FALLBACK_MODEL_PATH):
        MODEL_PATH = FALLBACK_MODEL_PATH

    tee = Tee(args.log)
    sys.stdout = tee

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} entries")

    # Filter by dataset
    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    # Only entries that have questions
    runnable = [e for e in meta if e.get("questions")]
    skipped_no_qa = len(meta) - len(runnable)
    if skipped_no_qa:
        print(f"Skipping {skipped_no_qa} entries with no Q&A")
    print(f"Running {len(runnable)} entries\n")

    if not runnable:
        print("Nothing to run. Exiting.")
        return

    model, processor = load_model(args.model_variant)

    correct = total = skipped_no_video = 0
    results = []

    with open(args.output, "w") as out_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video file not found")
                skipped_no_video += 1
                continue

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))

                try:
                    pred, reasoning = run_inference(
                        model,
                        processor,
                        video_path,
                        question,
                        choices,
                        args.fps,
                        args.max_pixels,
                        args.max_new_tokens,
                    )
                except Exception as e:
                    print(f"  ERROR {entry_label}: {e}")
                    pred, reasoning = "ERROR", ""

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "model_variant": args.model_variant,
                    "dataset":    entry.get("dataset"),
                    "task_type":  task_type,
                    "duration_s": entry.get("duration_s"),
                    "question":   question,
                    "choices":    choices,
                    "answer":     answer,
                    "prediction": pred,
                    "correct":    is_correct,
                    "reasoning":  reasoning,
                }
                out_f.write(json.dumps(result) + "\n")
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer}")

    # Final summary
    acc = correct / total if total else 0
    print(f"\n{'='*50}")
    print(f"Model variant: {args.model_variant}")
    print(f"Accuracy:      {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Results saved: {args.output}")

    # Per-dataset breakdown
    datasets: dict = {}
    for r in results:
        d = r["dataset"]
        datasets.setdefault(d, {"correct": 0, "total": 0})
        datasets[d]["total"] += 1
        if r["correct"]:
            datasets[d]["correct"] += 1
    print(f"\nPer-dataset:")
    for ds, s in sorted(datasets.items()):
        print(f"  {ds:<20} {s['correct']}/{s['total']} = {s['correct']/s['total']:.2%}")

if __name__ == "__main__":
    main()
