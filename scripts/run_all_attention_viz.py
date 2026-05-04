"""
Batch runner for Qwen2.5-Omni attention visualization scripts.

Runs viz_attention_encoders.py and/or viz_attention_depth_curve.py over every
video/question in videos/metadata.json and writes per-video/per-question folders.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED = {
    "encoder": [
        "1_audio_encoder_entropy.png",
        "2_audio_encoder_heatmaps.png",
        "3_vision_encoder_entropy.png",
        "4_vision_encoder_heatmaps.png",
        "5_encoder_entropy_comparison.png",
    ],
    "depth_curve": [
        "1_crossmodal_attention_curves.png",
        "2_self_vs_cross_attention.png",
        "3_attention_gini_depth.png",
        "4_crossmodal_heatmap.png",
        "5_last_token_attention_depth.png",
    ],
}

SCRIPT_BY_KIND = {
    "encoder": "viz_attention_encoders.py",
    "depth_curve": "viz_attention_depth_curve.py",
}


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
    if str(choices[0]).startswith("A"):
        return "\n".join(str(choice) for choice in choices)
    return "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))


def _build_user_prompt_video_mme(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    return _VIDEO_MME_OPTION_PROMPT + "\n" + question + "\n" + option_lines + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_worldsense(question: str, choices: list) -> str:
    parts: list[str] = [_WORLD_SENSE_SYS, _WORLD_SENSE_FRAMES_AUDIO, question + "\n"]
    for op in choices:
        parts.append(str(op) + "\n")
    return "".join(parts)


def _build_user_prompt_daily_omni(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    head = (
        "Listen and watch the video carefully. "
        "Select the best answer to the following multiple-choice question. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    return head + "\n" + question + "\n" + option_lines + "\n" + _VIDEO_MME_POST_PROMPT


def build_user_prompt_for_dataset(dataset: str, question: str, choices: list) -> str:
    dataset_name = (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")
    if dataset_name in {"video-mme", "videomme"}:
        return _build_user_prompt_video_mme(question, choices)
    if dataset_name == "worldsense":
        return _build_user_prompt_worldsense(question, choices)
    if dataset_name in {"daily-omni", "dailyomni"}:
        return _build_user_prompt_daily_omni(question, choices)
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


def resolve_video_path(file_field: str, videos_dir: str) -> str | None:
    """Resolve metadata paths robustly across Windows/Linux separators."""
    if os.path.exists(file_field):
        return file_field

    normalized = file_field.replace("\\", "/")
    if os.path.exists(normalized):
        return normalized

    rel = normalized
    if rel.startswith("videos/"):
        rel = rel[len("videos/") :]

    candidate = os.path.normpath(os.path.join(videos_dir, rel))
    if os.path.exists(candidate):
        return candidate

    filename = rel.split("/")[-1]
    suffix_matches = []
    for match in glob.glob(os.path.join(videos_dir, "**", filename), recursive=True):
        if match.replace("\\", "/").endswith(rel):
            suffix_matches.append(match)
    if suffix_matches:
        return suffix_matches[0]

    basename_matches = glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
    if len(basename_matches) == 1:
        return basename_matches[0]

    return None


def slugify(value: str, max_len: int = 80) -> str:
    value = value.strip().replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._/-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-/")
    return (value[:max_len].strip("._-/") or "item")


def video_rel_dir(entry: dict) -> Path:
    rel = str(entry.get("file") or entry.get("video") or "video").replace("\\", "/")
    if rel.startswith("videos/"):
        rel = rel[len("videos/") :]
    if rel.endswith("/video.mp4"):
        rel = rel[: -len("/video.mp4")]
    else:
        rel = os.path.splitext(rel)[0]
    parts = [slugify(part) for part in rel.split("/") if part and part != "videos"]
    return Path(*parts) if parts else Path("video")


def outputs_complete(out_dir: Path, kind: str) -> bool:
    return all((out_dir / name).exists() for name in EXPECTED[kind])


def write_question_files(out_dir: Path, entry: dict, question_obj: dict, prompt: str, q_index: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "question.txt").write_text(prompt, encoding="utf-8")
    payload = {
        "dataset": entry.get("dataset"),
        "task_type": question_obj.get("task_type", entry.get("task_type")),
        "duration_s": entry.get("duration_s"),
        "file": entry.get("file"),
        "question_index": q_index,
        "question": question_obj.get("question"),
        "choices": question_obj.get("choices", []),
        "answer": question_obj.get("answer"),
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_tasks(args: argparse.Namespace) -> list[dict]:
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    kinds = ["encoder", "depth_curve"] if args.scripts == "both" else [args.scripts]
    tasks: list[dict] = []
    question_count = 0

    for entry_i, entry in enumerate(metadata):
        questions = entry.get("questions") or []
        if not questions:
            continue

        video_path = resolve_video_path(str(entry.get("file", "")), args.videos)
        if not video_path:
            print(f"SKIP missing video for entry {entry_i}: {entry.get('file')}")
            continue

        rel_video_dir = video_rel_dir(entry)
        dataset = str(entry.get("dataset", ""))

        for q_i, question_obj in enumerate(questions, start=1):
            question_count += 1
            if args.limit_questions and question_count > args.limit_questions:
                return tasks

            prompt = build_user_prompt_for_dataset(
                dataset,
                str(question_obj.get("question", "")),
                list(question_obj.get("choices") or []),
            )
            q_dir_name = f"q{q_i:03d}"

            for kind in kinds:
                out_dir = out_root / kind / rel_video_dir / q_dir_name
                write_question_files(out_dir, entry, question_obj, prompt, q_i)

                if not args.overwrite and outputs_complete(out_dir, kind):
                    print(f"SKIP complete: {kind} {rel_video_dir}/{q_dir_name}")
                    continue

                tasks.append(
                    {
                        "kind": kind,
                        "script": str(REPO_ROOT / SCRIPT_BY_KIND[kind]),
                        "video_path": video_path,
                        "question": prompt,
                        "model_path": args.model_path,
                        "out_dir": str(out_dir),
                        "fps": args.fps,
                        "max_pixels": args.max_pixels,
                        "max_frames": args.max_frames,
                        "label": f"{kind} {rel_video_dir}/{q_dir_name}",
                    }
                )

    return tasks


def run_tasks(tasks: list[dict], gpus: list[str], poll_seconds: float, fail_fast: bool) -> int:
    pending = list(tasks)
    running: dict[str, tuple[subprocess.Popen, object, dict]] = {}
    failures = 0
    completed = 0

    while pending or running:
        free_gpus = [gpu for gpu in gpus if gpu not in running]
        for gpu in free_gpus:
            if not pending:
                break
            task = pending.pop(0)
            out_dir = Path(task["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(out_dir / "run.log", "w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            cmd = [
                sys.executable,
                task["script"],
                "--video_path",
                task["video_path"],
                "--question",
                task["question"],
                "--model_path",
                task["model_path"],
                "--out_dir",
                task["out_dir"],
            ]
            if task["fps"] is not None:
                cmd.extend(["--fps", str(task["fps"])])
            if task["max_pixels"] is not None:
                cmd.extend(["--max_pixels", str(task["max_pixels"])])
            if task["max_frames"] is not None:
                cmd.extend(["--max_frames", str(task["max_frames"])])
            print(f"START gpu={gpu}: {task['label']}")
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            running[gpu] = (proc, log_file, task)

        time.sleep(poll_seconds)

        for gpu, (proc, log_file, task) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue

            log_file.close()
            del running[gpu]
            completed += 1
            status_path = Path(task["out_dir"]) / "status.json"
            status = {
                "returncode": rc,
                "kind": task["kind"],
                "script": Path(task["script"]).name,
                "video_path": task["video_path"],
                "out_dir": task["out_dir"],
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

            if rc == 0:
                print(f"DONE  gpu={gpu}: {task['label']} ({completed}/{len(tasks)})")
            else:
                failures += 1
                print(f"FAIL  gpu={gpu}: {task['label']} rc={rc} (see {task['out_dir']}/run.log)")
                if fail_fast:
                    for _, (p, lf, _) in running.items():
                        p.terminate()
                        lf.close()
                    return failures

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run attention visualizations for all metadata questions.")
    parser.add_argument("--metadata", default="videos/metadata.json")
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--out_root", default="vizzing/all")
    parser.add_argument("--model_path", default="/data/armaan/models/Qwen2.5-Omni-7B")
    parser.add_argument("--fps", type=float, default=0.5, help="Video sampling FPS for visualization jobs.")
    parser.add_argument("--max_pixels", type=int, default=360 * 420, help="Max pixels per sampled frame.")
    parser.add_argument("--max_frames", type=int, default=48, help="Max sampled frames; use 0 to disable.")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids. One process is run per listed GPU.")
    parser.add_argument("--scripts", choices=["both", "encoder", "depth_curve"], default="both")
    parser.add_argument("--overwrite", action="store_true", help="Re-run tasks even when expected PNGs already exist.")
    parser.add_argument("--limit_questions", type=int, default=0, help="Debug limit across metadata questions; 0 means all.")
    parser.add_argument("--poll_seconds", type=float, default=5.0)
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        args.fps = None
    if args.max_pixels <= 0:
        args.max_pixels = None
    if args.max_frames <= 0:
        args.max_frames = None

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id")

    args.metadata = str((REPO_ROOT / args.metadata).resolve() if not Path(args.metadata).is_absolute() else Path(args.metadata))
    args.videos = str((REPO_ROOT / args.videos).resolve() if not Path(args.videos).is_absolute() else Path(args.videos))
    args.out_root = str((REPO_ROOT / args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root))

    tasks = make_tasks(args)
    print(f"Prepared {len(tasks)} tasks across GPUs {', '.join(gpus)}")
    print(f"Output root: {Path(args.out_root).resolve()}")
    if args.dry_run:
        for task in tasks[:20]:
            print(f"DRY {task['label']} -> {task['out_dir']}")
        if len(tasks) > 20:
            print(f"... {len(tasks) - 20} more")
        return 0

    failures = run_tasks(tasks, gpus, args.poll_seconds, args.fail_fast)
    print(f"Finished: {len(tasks) - failures}/{len(tasks)} succeeded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
