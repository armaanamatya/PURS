"""
eval_qwen3_omni.py
Runs Qwen3-Omni (MoE) on all videos in metadata.json and answers MCQ questions.
Reads Q&A from metadata.json — no HuggingFace needed on the server.

Requires transformers with Qwen3-Omni classes (Qwen3OmniMoeForConditionalGeneration).

Generation is always text-only (disable_talker + return_audio=False). Default input is
video+audio+text when the video has an audio stream; --no_audio only disables input audio.

Usage:
    python eval_qwen3_omni.py --metadata metadata.json --videos /workspace/videos --output results.jsonl
    python eval_qwen3_omni.py --metadata metadata.json --videos /workspace/videos --output results.jsonl --category lecture
"""

import argparse
import json
import os
import glob
import shutil
import sys
import time
import torch
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(_REPO_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)

from qwen_omni_utils import process_mm_info

from mcq_answer_parse import parse_answer


def cuda_time_ms(fn):
    """Run fn() with CUDA event timing. Returns (elapsed_ms, result)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


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


class StderrTee:
    """Duplicate stderr to a file (use instead of shell `| tee` when the directory may not exist yet)."""

    def __init__(self, log_file, terminal):
        self.log = log_file
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


ENV_MODEL_PATH_KEY = "QWEN3_OMNI_MODEL_PATH"
ENV_MODEL_PATH_FALLBACK_KEY = "QWEN_OMNI_MODEL_PATH"
DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen3-Omni-30B-A3B-Instruct"
FALLBACK_MODEL_PATH = "/workspace/model"

MODEL_PATH = (
    os.environ.get(ENV_MODEL_PATH_KEY)
    or os.environ.get(ENV_MODEL_PATH_FALLBACK_KEY)
    or DEFAULT_MODEL_PATH
)

# ── Prompts (same sources as Qwen2.5-Omni web_demo + OmniZip lmms-eval task utils) ─────────

SYSTEM_PROMPT_DEFAULT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

SYSTEM_MCQ_SUFFIX = (
    "For multiple-choice questions, reply with only one letter: A, B, C, or D. "
    "Do not explain, do not ask follow-up questions, and do not add text after the letter."
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


# ── Answer parsing: see mcq_answer_parse.py ───────────────────────────────────

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(dtype_name: str):
    dt = resolve_model_dtype(dtype_name)
    try:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    except ImportError as e:
        raise ImportError(
            "Qwen3 Omni classes not found in your installed transformers. "
            "Install a transformers version that includes Qwen3-Omni support "
            "(e.g. pip install git+https://github.com/huggingface/transformers)."
        ) from e

    print(f"Loading qwen3-omni from {MODEL_PATH} (torch_dtype={dtype_name}) ...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dt,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor


def _prepare_omni_inputs(model: torch.nn.Module, inputs: object) -> object:
    """Move processor outputs to the model device; cast only floating tensors to model dtype (never input_ids)."""
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            inputs[key] = value.to(model.dtype)
    return inputs


def _generation_output_token_ids(gen_out: object) -> torch.Tensor:
    """HF generate() may return a LongTensor or GenerateDecoderOnlyOutput; iterating the latter iterates dict keys."""
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    return gen_out


# Keys Qwen3-OmniProcessor may put on the batch but model.generate() does not accept (HF warnings).
_OMNI_GENERATE_BATCH_DROP = frozenset({"images", "return_tensors", "text"})


def _batch_for_omni_generate(batch: object) -> dict:
    return {k: v for k, v in batch.items() if k not in _OMNI_GENERATE_BATCH_DROP}


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
    measure_prefill: bool = False,
) -> tuple[str, str, dict]:
    prompt = build_user_prompt_for_dataset(dataset, question, choices)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    effective_use_audio = use_audio
    try:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=effective_use_audio)
    except Exception as first_err:
        # qwen_omni_utils loads video audio via librosa on the .mp4 path; soundfile often fails on mp4 and
        # the chained exception may not match substring checks — retry without separate audio extraction.
        if not effective_use_audio:
            raise first_err
        try:
            audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
            effective_use_audio = False
        except Exception as second_err:
            raise second_err from first_err

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=effective_use_audio,
    )
    inputs = _prepare_omni_inputs(model, inputs)

    tokenizer = processor.tokenizer
    # Text-only: use return_audio=False on the top-level Omni wrapper. Do not pass generation_mode here —
    # many transformers builds forward unknown kwargs to thinker.generate(), which then errors on
    # unused model_kwargs (generation_mode is only valid on some wrapper versions).
    gen_kw: dict = {
        "use_audio_in_video": effective_use_audio,
        "return_audio": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if hasattr(model, "thinker"):
        gen_kw["thinker_max_new_tokens"] = max_new_tokens
        gen_kw["thinker_do_sample"] = False
    else:
        gen_kw["max_new_tokens"] = max_new_tokens
        gen_kw["temperature"] = 0.0
        gen_kw["do_sample"] = False

    gen_in = _batch_for_omni_generate(inputs)

    timing = {}
    if measure_prefill:
        prefill_kw = dict(gen_kw)
        if "thinker_max_new_tokens" in prefill_kw:
            prefill_kw["thinker_max_new_tokens"] = 1
        else:
            prefill_kw["max_new_tokens"] = 1
        with torch.no_grad():
            prefill_ms, _ = cuda_time_ms(lambda: model.generate(**gen_in, **prefill_kw))
        timing["prefill_ms"] = round(prefill_ms, 2)

    with torch.no_grad():
        if measure_prefill:
            e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))
            timing["e2e_ms"] = round(e2e_ms, 2)
        else:
            raw_out = model.generate(**gen_in, **gen_kw)

    seq_ids = _generation_output_token_ids(raw_out)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, timing

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
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model path (or set {ENV_MODEL_PATH_KEY} or {ENV_MODEL_PATH_FALLBACK_KEY}).",
    )
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
        help="Input: do not load audio from the video. Generation stays text-only either way.",
    )
    parser.add_argument(
        "--measure_prefill",
        action="store_true",
        help="Measure prefill time (TTFT) via generate(max_new_tokens=1) with CUDA events. "
             "Adds a second generate() call per question — use for benchmarking only.",
    )
    parser.add_argument("--vram_log", default="/workspace/vram_log.jsonl", help="Per-question VRAM JSONL (allocated/reserved)")
    parser.add_argument(
        "--stderr_log",
        default=None,
        help="Append stderr to this file (parent dirs created automatically). Prefer this over `| tee` when run2/... may not exist yet.",
    )
    args = parser.parse_args()

    global MODEL_PATH
    if args.model:
        MODEL_PATH = args.model
    elif not os.path.exists(MODEL_PATH) and os.path.exists(FALLBACK_MODEL_PATH):
        MODEL_PATH = FALLBACK_MODEL_PATH

    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print(
            "WARNING: ffmpeg not found in PATH. Decoding MP4 audio needs ffmpeg (or use --no_audio, or "
            "QWEN_OMNI_AUDIO_WAV_ROOT + pre-extracted .wav). Without sudo: "
            "`conda install -c conda-forge ffmpeg`, or put a static ffmpeg in ~/bin and export PATH, "
            "or ask an admin for system ffmpeg.\n"
        )

    errors_log_path = args.errors_log or os.path.join(os.path.dirname(args.log) or ".", "errors.log")
    paths_for_dirs = [args.log, args.output, args.vram_log, errors_log_path]
    if args.stderr_log is not None:
        paths_for_dirs.append(args.stderr_log)
    for path in paths_for_dirs:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if args.stderr_log is not None:
        _stderr_f = open(args.stderr_log, "a", encoding="utf-8")
        sys.stderr = StderrTee(_stderr_f, sys.__stderr__)

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
    print(f"Running {len(runnable)} entries")
    print(
        "Each line in --output is written after one question finishes; the first can take many minutes "
        "(video decode + optional MP4 audio via librosa/audioread + generation). Use --no_audio to skip separate audio loading.\n"
    )

    if not runnable:
        print("Nothing to run. Exiting.")
        return

    model, processor = load_model(args.dtype)

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
                    pred, reasoning, timing = run_inference(
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
                        measure_prefill=args.measure_prefill,
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
                        **timing,
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                    with open(errors_log_path, "a") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{tb}\n")
                    pred, reasoning = "ERROR", str(e)
                    timing = {}

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "model_variant": "qwen3-omni",
                    "dataset":    entry.get("dataset"),
                    "task_type":  task_type,
                    "duration_s": entry.get("duration_s"),
                    "question":   question,
                    "choices":    choices,
                    "answer":     answer,
                    "prediction": pred,
                    "correct":    is_correct,
                    "reasoning":  reasoning,
                    **timing,
                }
                out_f.write(json.dumps(result) + "\n")
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer}")

    # Final summary
    acc = correct / total if total else 0
    print(f"\n{'='*50}")
    print("Model variant: qwen3-omni")
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
