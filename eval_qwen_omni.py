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

MODEL_PATH = "/workspace/model"

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model from {MODEL_PATH} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, processor, video_path: str, question: str, choices: list) -> str:
    choice_text = "\n".join(choices) if choices[0].startswith("A") else "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    prompt = f"{question}\n\n{choice_text}\n\nAnswer with only the letter (A, B, C, or D)."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant. Answer multiple-choice questions about videos concisely."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 2.0},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output = model.generate(**inputs, use_audio_in_video=False, return_audio=False, max_new_tokens=8)

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    for char in reversed(decoded.strip()):
        if char in "ABCD":
            return char
    return decoded.strip()

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
    parser.add_argument("--metadata", default="metadata.json", help="Path to enriched metadata.json")
    parser.add_argument("--videos",   default="/workspace/videos", help="Directory containing video files")
    parser.add_argument("--output",   default="/workspace/results.jsonl", help="Output JSONL file")
    parser.add_argument("--log",      default="/workspace/eval.log", help="Log file (appended each run)")
    parser.add_argument("--category", default=None, help="Only run this category (e.g. lecture)")
    args = parser.parse_args()

    tee = Tee(args.log)
    sys.stdout = tee

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} entries")

    # Filter by category
    if args.category:
        meta = [e for e in meta if e["category"] == args.category]
        print(f"Filtered to {len(meta)} entries for category '{args.category}'")

    # Only entries that have questions
    runnable = [e for e in meta if e.get("questions")]
    skipped_no_qa = len(meta) - len(runnable)
    if skipped_no_qa:
        print(f"Skipping {skipped_no_qa} entries with no Q&A (run enrich_metadata.py first)")
    print(f"Running {len(runnable)} entries\n")

    if not runnable:
        print("Nothing to run. Exiting.")
        return

    model, processor = load_model()

    correct = total = skipped_no_video = 0
    results = []

    with open(args.output, "w") as out_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            if video_path is None:
                print(f"  SKIP {entry['category']}_{entry['index']}: video file not found")
                skipped_no_video += 1
                continue

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", "")

                try:
                    pred = run_inference(model, processor, video_path, question, choices)
                except Exception as e:
                    print(f"  ERROR {entry['category']}_{entry['index']}: {e}")
                    pred = "ERROR"

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "category":      entry["category"],
                    "index":         entry["index"],
                    "dataset_source":entry["dataset_source"],
                    "duration_s":    entry.get("actual_duration_s"),
                    "task_type":     task_type,
                    "question":      question,
                    "choices":       choices,
                    "answer":        answer,
                    "prediction":    pred,
                    "correct":       is_correct,
                }
                out_f.write(json.dumps(result) + "\n")
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {entry['category']}_{entry['index']} [{task_type}] pred={pred} ans={answer}")

    # Final summary
    acc = correct / total if total else 0
    print(f"\n{'='*50}")
    print(f"Accuracy:      {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Results saved: {args.output}")

    # Per-category breakdown
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, {"correct": 0, "total": 0})
        cats[c]["total"] += 1
        if r["correct"]:
            cats[c]["correct"] += 1
    print(f"\nPer-category:")
    for cat, s in sorted(cats.items()):
        print(f"  {cat:<20} {s['correct']}/{s['total']} = {s['correct']/s['total']:.2%}")

if __name__ == "__main__":
    main()
