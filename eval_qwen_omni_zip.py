"""
eval_qwen_omni_zip.py
Runs Qwen2.5-Omni-7B + OmniZip token compression on all videos in metadata.json.

OmniZip compresses audio-visual tokens dynamically, giving ~3.4x speedup and
~1.4x memory reduction with minimal accuracy loss — so we can afford fps=2.0
and higher resolution vs the plain eval.

Usage:
    python eval_qwen_omni_zip.py --metadata /workspace/metadata.json \\
        --videos /workspace/videos --output /workspace/results_zip.jsonl

OmniZip params (tune to trade speed vs quality):
    --rho_audio       fraction of audio tokens to KEEP  (default 0.4)
    --rho_video       fraction of video tokens to KEEP  (default 0.7)
    --g               temporal group size               (default 3)
    --contextual_ratio fraction of contextual tokens    (default 0.05)
"""

import argparse
import json
import os
import glob
import re
import sys
import torch
from datetime import datetime
from pathlib import Path

# OmniZip replaces the standard model class — must be importable from OmniZip-main/
# (same layout as viz_attention_omnizip.py).
OMNIZIP_DIR = os.path.join(os.path.dirname(__file__), "OmniZip-main")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src"))

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

MODEL_PATH = "/workspace/model"

# ── Tee logger ────────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"RUN (OmniZip): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(rho_audio: float, rho_video: float, g: int, contextual_ratio: float):
    print(f"Loading model from {MODEL_PATH} with OmniZip ...")
    print(f"  rho_audio={rho_audio}  rho_video={rho_video}  g={g}  contextual_ratio={contextual_ratio}")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    omnizip_config = {
        "rho_audio":        rho_audio,
        "rho_video":        rho_video,
        "g":                g,
        "contextual_ratio": contextual_ratio,
    }
    model.thinker.omnizip_config = omnizip_config

    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, processor, video_path: str, question: str, choices: list) -> tuple[str, str]:
    choice_text = "\n".join(choices) if choices and choices[0].startswith("A") \
                  else "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    prompt = (f"{question}\n\n{choice_text}\n\n"
              "Think step by step, then end your response with 'Answer: X' where X is A, B, C, or D.")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
        )}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": 1.0, "max_pixels": 360*420},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

    # OmniZip requires nframes to be set on thinker before each forward pass
    num_input_frames = videos[0].shape[0] if videos else 1
    model.thinker.nframes = num_input_frames

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False,
            max_new_tokens=512,
            thinker_do_sample=False,
        )

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    if "assistant" in decoded.lower():
        decoded = decoded[decoded.lower().rfind("assistant") + len("assistant"):].strip()

    m = re.search(r"Answer:\s*([A-D])", decoded, re.IGNORECASE)
    letter = m.group(1).upper() if m else next(
        (c for c in reversed(decoded.strip()) if c in "ABCD"), decoded.strip()
    )
    return letter, decoded

# ── Video lookup ──────────────────────────────────────────────────────────────

def resolve_video_path(file_field: str, videos_dir: str) -> str | None:
    if os.path.exists(file_field):
        return file_field
    normalized = file_field.replace("\\", "/")
    # Try path relative to videos_dir: strip leading "videos/" prefix
    rel = normalized
    for prefix in ("videos/", "videos\\"):
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):]
            break
    candidate = os.path.join(videos_dir, rel)
    if os.path.exists(candidate):
        return candidate
    # Fallback: search by stem
    stem = normalized.split("/")[-1].rsplit(".", 1)[0]
    for ext in ("mp4", "mkv", "webm", "avi"):
        matches = glob.glob(os.path.join(videos_dir, "**", f"{stem}.{ext}"), recursive=True)
        if matches:
            return matches[0]
    return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata",         default="/workspace/metadata.json")
    parser.add_argument("--videos",           default="/workspace/videos")
    parser.add_argument("--output",           default="/workspace/results_zip.jsonl")
    parser.add_argument("--log",              default="/workspace/eval_zip.log")
    parser.add_argument("--category",         default=None, help="Filter by dataset or task_type")
    parser.add_argument("--rho_audio",        type=float, default=0.4)
    parser.add_argument("--rho_video",        type=float, default=0.7)
    parser.add_argument("--g",                type=int,   default=3)
    parser.add_argument("--contextual_ratio", type=float, default=0.05)
    parser.add_argument("--vram_log",         default="/workspace/vram_log.jsonl")
    args = parser.parse_args()

    tee = Tee(args.log)
    sys.stdout = tee

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} entries")

    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    runnable = [e for e in meta if e.get("questions")]
    print(f"Running {len(runnable)} entries with OmniZip\n")

    if not runnable:
        print("Nothing to run. Exiting.")
        return

    model, processor = load_model(args.rho_audio, args.rho_video, args.g, args.contextual_ratio)

    correct = total = skipped_no_video = 0
    results = []

    with open(args.output, "w") as out_f, open(args.vram_log, "w") as vram_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video not found")
                skipped_no_video += 1
                continue

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))

                try:
                    torch.cuda.reset_peak_memory_stats()
                    pred, reasoning = run_inference(model, processor, video_path, question, choices)
                    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
                    curr_vram = torch.cuda.memory_allocated() / 1024**3
                    vram_entry = {
                        "entry": entry_label, "task_type": task_type,
                        "duration_s": entry.get("duration_s"),
                        "peak_vram_gb": round(peak_vram, 2),
                        "after_vram_gb": round(curr_vram, 2),
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"  ERROR {entry_label}: {e}")
                    with open("/workspace/errors.log", "a") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{tb}\n")
                    pred, reasoning = "ERROR", str(e)

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "dataset":    entry.get("dataset"),
                    "task_type":  task_type,
                    "duration_s": entry.get("duration_s"),
                    "question":   question,
                    "choices":    choices,
                    "answer":     answer,
                    "prediction": pred,
                    "correct":    is_correct,
                    "reasoning":  reasoning,
                    "method":     "omnizip",
                }
                out_f.write(json.dumps(result) + "\n")
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer}")

    acc = correct / total if total else 0
    print(f"\n{'='*50}")
    print(f"OmniZip Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Results: {args.output}")

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
