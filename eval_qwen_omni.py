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
import re
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(_REPO_ROOT, "OmniZip-main")
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

# ── Prompts (same sources as Qwen2.5-Omni web_demo + OmniZip lmms-eval task utils) ─────────

SYSTEM_PROMPT_DEFAULT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

_WORLD_SENSE_SYS = (
    "Carefully watch this video and pay attention to every detail. "
    "Based on your observations, select the best option that accurately addresses the question."
)

_WORLD_SENSE_FRAMES_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

_VIDEO_MME_OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)

_VIDEO_MME_POST_PROMPT = "The best answer is:"


def _format_choice_lines(choices: list) -> str:
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def _build_user_prompt_video_mme(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    question_block = question + "\n" + option_lines
    return _VIDEO_MME_OPTION_PROMPT + "\n" + question_block + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_worldsense(question: str, choices: list) -> str:
    parts: list[str] = [_WORLD_SENSE_SYS, _WORLD_SENSE_FRAMES_AUDIO, question + "\n"]
    for op in choices:
        parts.append(op + "\n")
    return "".join(parts)


def _build_user_prompt_daily_omni(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    head = (
        "Listen and watch the video carefully. "
        "Select the best answer to the following multiple-choice question. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    return head + "\n" + question + "\n" + option_lines + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_default(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        + question
        + "\n"
        + option_lines
        + "\n"
        + _VIDEO_MME_POST_PROMPT
    )


def _canonicalize_dataset_name(dataset: str | None) -> str:
    return (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")


def build_user_prompt_for_dataset(dataset: str, question: str, choices: list) -> str:
    dataset_name = _canonicalize_dataset_name(dataset)
    if dataset_name in {"video-mme", "videomme"}:
        return _build_user_prompt_video_mme(question, choices)
    if dataset_name == "worldsense":
        return _build_user_prompt_worldsense(question, choices)
    if dataset_name in {"daily-omni", "dailyomni"}:
        return _build_user_prompt_daily_omni(question, choices)
    return _build_user_prompt_default(question, choices)


def resolve_model_dtype(dtype_name: str):
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in table:
        keys = ", ".join(sorted(table.keys()))
        raise ValueError(f"Unknown dtype_name {dtype_name!r}; expected one of: {keys}")
    return table[dtype_name]


# ── Audio: lmms-eval Qwen2_5_Omni sets use_audio_in_video per video via _check_if_video_has_audio ──

def check_video_has_audio(video_path: str) -> bool:
    """Return True if the video file contains an audio stream."""
    try:
        import av
        container = av.open(video_path)
        has = len(container.streams.audio) > 0
        container.close()
        return has
    except Exception:
        return False


# ── Answer parsing (matching official evals) ─────────────────────────────────

def parse_answer(response: str, choices: list | None = None) -> str:
    """Parse model response to extract answer letter.
    Uses official parsing: check if response starts with A-D,
    then search for first [ABCD], then fallback to 'A'.
    """
    resp = response.strip()
    # Strip common answer prefixes
    for prefix in ["The best answer is", "The correct answer is", "The answer is",
                   "The answer", "The best option is", "The correct option is",
                   "Best answer:", "Best option:"]:
        if resp.lower().startswith(prefix.lower()):
            resp = resp[len(prefix):].strip()

    # Check if response starts with a letter (official OmniZip eval approach)
    for opt in ["A", "B", "C", "D"]:
        if resp.upper().startswith(opt):
            return opt

    # Search for first ABCD match (official VideoMME approach)
    m = re.search(r"[ABCD]", resp)
    if m:
        return m[0]

    # Fallback
    return "A"

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_variant: str, dtype_name: str):
    dt = resolve_model_dtype(dtype_name)
    print(f"Loading {model_variant} model from {MODEL_PATH} (torch_dtype={dtype_name}) ...")
    if model_variant == "qwen2.5-omni":
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=dt,
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
            torch_dtype=dt,
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
    dataset: str,
    question: str,
    choices: list,
    fps: float,
    max_pixels: int,
    max_new_tokens: int,
    use_audio: bool,
) -> tuple[str, str]:
    prompt = build_user_prompt_for_dataset(dataset, question, choices)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DEFAULT}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    effective_use_audio = use_audio
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            audios, images, videos = process_mm_info(messages, use_audio_in_video=effective_use_audio)
            last_err = None
            break
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    if last_err is not None:
        if effective_use_audio:
            msg = str(last_err)
            if ("NoBackendError" in msg) or ("can't start new thread" in msg) or ("Format not recognised" in msg):
                effective_use_audio = False
                audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            else:
                raise last_err
        else:
            raise last_err

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=effective_use_audio,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            use_audio_in_video=effective_use_audio,
            return_audio=False,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output)]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    letter = parse_answer(decoded, choices)
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
    parser.add_argument("--errors_log", default=None, help="Where to append tracebacks (default: alongside --log).")
    parser.add_argument("--category", default=None, help="Only run this category (e.g. lecture)")
    parser.add_argument("--fps",      type=float, default=2.0, help="Video sampling fps")
    parser.add_argument("--max_pixels", type=int, default=360*420, help="Max pixels per frame")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Generation cap (lmms-eval qwen2_5_omni default)")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weights dtype (use explicit values; torch_dtype='auto' often fails on Windows/some GPUs)",
    )
    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Never pass audio from video (global). Default: per-video, enable audio only if the file has an audio stream (like lmms-eval).",
    )
    parser.add_argument("--vram_log", default="/workspace/vram_log.jsonl", help="Per-question VRAM JSONL (allocated/reserved)")
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

    errors_log_path = args.errors_log or os.path.join(os.path.dirname(args.log) or ".", "errors.log")
    errors_log_dir = os.path.dirname(errors_log_path)
    if errors_log_dir:
        os.makedirs(errors_log_dir, exist_ok=True)

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

    model, processor = load_model(args.model_variant, args.dtype)

    correct = total = skipped_no_video = 0
    results = []

    with open(args.output, "w") as out_f, open(args.vram_log, "w") as vram_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video file not found")
                skipped_no_video += 1
                continue

            use_audio = (not args.no_audio) and check_video_has_audio(video_path)

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset   = entry.get("dataset", "")

                try:
                    torch.cuda.reset_peak_memory_stats()
                    pred, reasoning = run_inference(
                        model,
                        processor,
                        video_path,
                        dataset,
                        question,
                        choices,
                        args.fps,
                        args.max_pixels,
                        args.max_new_tokens,
                        use_audio,
                    )
                    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                    peak_resv_gb = torch.cuda.max_memory_reserved() / 1024**3
                    curr_alloc_gb = torch.cuda.memory_allocated() / 1024**3
                    curr_resv_gb = torch.cuda.memory_reserved() / 1024**3
                    vram_entry = {
                        "entry": entry_label,
                        "task_type": task_type,
                        "duration_s": entry.get("duration_s"),
                        "peak_alloc_gb": round(peak_alloc_gb, 2),
                        "peak_reserved_gb": round(peak_resv_gb, 2),
                        "after_alloc_gb": round(curr_alloc_gb, 2),
                        "after_reserved_gb": round(curr_resv_gb, 2),
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"  ERROR {entry_label}: {e}")
                    with open(errors_log_path, "a") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{tb}\n")
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
