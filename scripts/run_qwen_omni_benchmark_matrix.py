#!/usr/bin/env python3
"""
Run the Qwen2.5-Omni benchmark matrix across methods, temperatures, and repeats.

This script launches the existing eval_*.py entrypoints, keeps one eval process per
selected GPU, samples nvidia-smi for process-level VRAM, and summarizes each
method/temperature distribution across repeated full-dataset runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import queue
import shlex
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


METHODS = ("baseline", "gptq", "awq", "omnizip", "divprune", "mixkv", "rediprune")
DEFAULT_TEMPERATURES = (0.1, 0.9)
STAT_FIELDS = (
    "accuracy",
    "prefill_ms_mean",
    "prefill_ms_max",
    "e2e_ms_mean",
    "e2e_ms_max",
    "orig_frames_mean",
    "orig_frames_max",
    "used_frames_mean",
    "used_frames_max",
    "frame_keep_ratio_mean",
    "model_loaded_alloc_gb",
    "model_loaded_reserved_gb",
    "peak_alloc_gb",
    "peak_reserved_gb",
    "inference_extra_peak_alloc_gb",
    "after_alloc_gb",
    "gpu_process_peak_gb",
    "gpu_total_peak_gb",
    "run_wall_s",
)


@dataclass(frozen=True)
class Job:
    method: str
    temperature: float
    temp_label: str
    repeat: int
    seed: int
    out_dir: Path
    command: list[str]
    result_path: Path
    vram_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_methods(raw: str) -> list[str]:
    values = parse_csv_values(raw)
    unknown = [value for value in values if value not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}. Choices: {', '.join(METHODS)}")
    return values


def parse_temperatures(raw: str) -> list[float]:
    values = [float(item) for item in parse_csv_values(raw)]
    if not values:
        raise SystemExit("At least one temperature is required.")
    return values


def temp_label(temp: float) -> str:
    text = f"{temp:g}"
    return text.replace("-", "m").replace(".", "p")


def run_label(temp: float, repeat: int) -> str:
    return f"temp_{temp_label(temp)}-run_{repeat:02d}"


def legacy_run_dir(output_root: Path, method: str, temp: float, repeat: int) -> Path:
    return output_root / method / f"temp_{temp_label(temp)}" / f"run_{repeat:02d}"


def json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"_parse_error": str(exc), "_line_no": line_no})
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def summarize_values(values: list[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "stddev": None, "variance": None, "min": None, "max": None}
    if len(clean) == 1:
        return {
            "n": 1,
            "mean": clean[0],
            "stddev": 0.0,
            "variance": 0.0,
            "min": clean[0],
            "max": clean[0],
        }
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "stddev": statistics.stdev(clean),
        "variance": statistics.variance(clean),
        "min": min(clean),
        "max": max(clean),
    }


def run_text_command(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()


def query_gpu_rows() -> list[dict[str, int]]:
    output = run_text_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, int]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "used_mb": int(parts[1]),
                "free_mb": int(parts[2]),
                "total_mb": int(parts[3]),
            }
        )
    return rows


def query_compute_app_memory() -> dict[int, int]:
    try:
        output = run_text_command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    rows: dict[int, int] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return rows


def query_single_gpu_memory(gpu: str) -> dict[str, int] | None:
    try:
        output = run_text_command(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    line = output.splitlines()[0] if output.splitlines() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    return {"used_mb": int(parts[0]), "free_mb": int(parts[1]), "total_mb": int(parts[2])}


def descendants_of(pid: int) -> set[int]:
    seen = {pid}
    stack = [pid]
    while stack:
        current = stack.pop()
        child_file = Path(f"/proc/{current}/task/{current}/children")
        if not child_file.exists():
            continue
        try:
            child_ids = [int(raw) for raw in child_file.read_text().split()]
        except (OSError, ValueError):
            child_ids = []
        for child in child_ids:
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


class GpuSampler:
    def __init__(self, pid: int, gpu: str, path: Path, interval_s: float):
        self.pid = pid
        self.gpu = gpu
        self.path = path
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self.interval_s <= 0:
            return
        self._thread.start()

    def stop(self) -> None:
        if self.interval_s <= 0:
            return
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_s * 2))

    def _run(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                pid_set = descendants_of(self.pid)
                app_memory = query_compute_app_memory()
                process_used_mb = sum(memory for pid, memory in app_memory.items() if pid in pid_set)
                gpu_memory = query_single_gpu_memory(self.gpu) or {}
                record = {
                    "time_s": time.time(),
                    "root_pid": self.pid,
                    "tracked_pids": sorted(pid_set),
                    "process_used_mb": process_used_mb,
                    "gpu_index": self.gpu,
                    "gpu_used_mb": gpu_memory.get("used_mb"),
                    "gpu_free_mb": gpu_memory.get("free_mb"),
                    "gpu_total_mb": gpu_memory.get("total_mb"),
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                self._stop.wait(self.interval_s)


def select_gpus(spec: str, min_free_gb: float, dry_run: bool) -> list[str]:
    spec = spec.strip().lower()
    explicit = spec not in {"auto", "available", "any", "all"}
    if explicit:
        return [str(int(item)) for item in parse_csv_values(spec)]

    try:
        rows = query_gpu_rows()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if dry_run:
            return ["0"]
        raise SystemExit("nvidia-smi is required for --gpus auto/any/all. Pass --gpus 0 or another explicit list.")

    if spec == "all":
        selected = [row["index"] for row in rows]
    else:
        min_free_mb = int(min_free_gb * 1024)
        available = [row for row in rows if row["free_mb"] >= min_free_mb]
        if not available:
            detail = ", ".join(f"{row['index']}:{row['free_mb']/1024:.1f}GB free" for row in rows)
            raise SystemExit(f"No GPU has at least {min_free_gb:g} GB free. Current free memory: {detail}")
        if spec == "any":
            selected = [max(available, key=lambda row: row["free_mb"])["index"]]
        else:
            selected = [row["index"] for row in available]
    return [str(index) for index in selected]


def script_for_method(method: str) -> str:
    if method in {"baseline", "gptq", "awq"}:
        return "eval_qwen_omni.py"
    if method == "omnizip":
        return "eval_qwen_omni_zip.py"
    if method == "divprune":
        return "eval_qwen_omni_divprune.py"
    if method == "mixkv":
        return "eval_qwen_omni_mixkv.py"
    if method == "rediprune":
        return "eval_qwen_omni_rediprune.py"
    raise ValueError(method)


def method_specific_args(method: str, args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if method == "baseline":
        out += ["--model_variant", "qwen2.5-omni"]
        if args.qwen_model:
            out += ["--model", args.qwen_model]
    elif method == "gptq":
        out += ["--model_variant", "qwen2.5-omni", "--quantization", "gptq"]
        if args.gptq_model:
            out += ["--model", args.gptq_model]
    elif method == "awq":
        out += ["--model_variant", "qwen2.5-omni", "--quantization", "awq"]
        if args.awq_model:
            out += ["--model", args.awq_model]
    elif method == "omnizip":
        out += [
            "--rho_audio",
            str(args.rho_audio),
            "--rho_video",
            str(args.rho_video),
            "--g",
            str(args.omnizip_g),
            "--contextual_ratio",
            str(args.contextual_ratio),
        ]
        if args.qwen_model:
            out += ["--model", args.qwen_model]
    elif method == "divprune":
        out += ["--subset_ratio", str(args.divprune_subset_ratio), "--prune_mode", args.divprune_prune_mode]
        if args.qwen_model:
            out += ["--model", args.qwen_model]
    elif method == "mixkv":
        out += [
            "--budget",
            str(args.mixkv_budget),
            "--window_size",
            str(args.mixkv_window_size),
            "--select_method",
            args.mixkv_select_method,
        ]
        if args.mixkv_head_score_path:
            out += ["--head_score_path", args.mixkv_head_score_path]
        if args.qwen_model:
            out += ["--model", args.qwen_model]
    elif method == "rediprune":
        out += [
            "--subset_ratio",
            str(args.rediprune_subset_ratio),
            "--prune_mode",
            args.rediprune_prune_mode,
            "--alpha",
            str(args.rediprune_alpha),
            "--tau",
            str(args.rediprune_tau),
        ]
        if args.qwen_model:
            out += ["--model", args.qwen_model]
    return out


def build_jobs(args: argparse.Namespace) -> list[Job]:
    root = repo_root()
    methods = parse_methods(args.methods)
    temperatures = parse_temperatures(args.temperatures)
    jobs: list[Job] = []
    for method_i, method in enumerate(methods):
        for temp_i, temp in enumerate(temperatures):
            label = temp_label(temp)
            for repeat in range(1, args.runs + 1):
                seed = args.seed_base + method_i * 100000 + temp_i * 1000 + repeat
                out_dir = args.output_root / method / run_label(temp, repeat)
                result_path = out_dir / "results.jsonl"
                vram_path = out_dir / "vram_log.jsonl"
                command = [
                    args.python,
                    str(root / script_for_method(method)),
                    "--metadata",
                    args.metadata,
                    "--videos",
                    args.videos,
                    "--output",
                    str(result_path),
                    "--log",
                    str(out_dir / "console.log"),
                    "--vram_log",
                    str(vram_path),
                    "--errors_log",
                    str(out_dir / "errors.log"),
                    "--stderr_log",
                    str(out_dir / "stderr.log"),
                    "--fps",
                    str(args.fps),
                    "--max_pixels",
                    str(args.max_pixels),
                    "--max_frames_videomme",
                    str(args.max_frames_videomme),
                    "--max_frames_other",
                    str(args.max_frames_other),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--dtype",
                    args.dtype,
                    "--temperature",
                    str(temp),
                    "--seed",
                    str(seed),
                ]
                if args.category:
                    command += ["--category", args.category]
                if args.no_audio:
                    command.append("--no_audio")
                if args.measure_prefill:
                    command.append("--measure_prefill")
                command += method_specific_args(method, args)
                command += args.extra_eval_args
                jobs.append(
                    Job(
                        method=method,
                        temperature=temp,
                        temp_label=label,
                        repeat=repeat,
                        seed=seed,
                        out_dir=out_dir,
                        command=command,
                        result_path=result_path,
                        vram_path=vram_path,
                    )
                )
    return jobs


def summarize_gpu_samples(path: Path) -> dict[str, float | None]:
    rows = read_jsonl(path)
    process_values = numeric_values(rows, "process_used_mb")
    gpu_values = numeric_values(rows, "gpu_used_mb")
    return {
        "gpu_process_peak_gb": (max(process_values) / 1024) if process_values else None,
        "gpu_total_peak_gb": (max(gpu_values) / 1024) if gpu_values else None,
    }


def summarize_run(job: Job, return_code: int, start_s: float, end_s: float, gpu: str) -> dict[str, Any]:
    results = read_jsonl(job.result_path)
    vram_rows = read_jsonl(job.vram_path)
    samples_path = job.out_dir / "gpu_samples.jsonl"

    total = len([row for row in results if "_parse_error" not in row])
    correct = sum(1 for row in results if row.get("correct") is True)
    accuracy = correct / total if total else None

    prefill_values = numeric_values(results, "prefill_ms") or numeric_values(vram_rows, "prefill_ms")
    e2e_values = numeric_values(results, "e2e_ms") or numeric_values(vram_rows, "e2e_ms")
    model_loaded_alloc = safe_max(numeric_values(vram_rows, "model_loaded_alloc_gb"))
    model_loaded_reserved = safe_max(numeric_values(vram_rows, "model_loaded_reserved_gb"))
    peak_alloc = safe_max(numeric_values(vram_rows, "peak_alloc_gb"))
    peak_reserved = safe_max(numeric_values(vram_rows, "peak_reserved_gb"))
    after_alloc = safe_max(numeric_values(vram_rows, "after_alloc_gb"))
    orig_values = numeric_values(vram_rows, "orig_frames") or numeric_values(results, "orig_frames")
    used_values = numeric_values(vram_rows, "used_frames")
    if not used_values:
        used_values = numeric_values(vram_rows, "pruned_frames")
    if not used_values:
        used_values = numeric_values(results, "used_frames")
    if not used_values:
        used_values = numeric_values(results, "pruned_frames")

    frame_ratios: list[float] = []
    if vram_rows:
        for row in vram_rows:
            raw_orig = row.get("orig_frames")
            raw_used = row.get("used_frames", row.get("pruned_frames"))
            try:
                orig_value = float(raw_orig)
                used_value = float(raw_used)
            except (TypeError, ValueError):
                continue
            if orig_value > 0:
                frame_ratios.append(used_value / orig_value)
    elif results:
        for row in results:
            raw_orig = row.get("orig_frames")
            raw_used = row.get("used_frames", row.get("pruned_frames"))
            try:
                orig_value = float(raw_orig)
                used_value = float(raw_used)
            except (TypeError, ValueError):
                continue
            if orig_value > 0:
                frame_ratios.append(used_value / orig_value)

    extra_values: list[float] = []
    for row in vram_rows:
        try:
            peak = float(row["peak_alloc_gb"])
        except (KeyError, TypeError, ValueError):
            continue
        baseline = row.get("before_alloc_gb", model_loaded_alloc)
        try:
            baseline_value = float(baseline)
        except (TypeError, ValueError):
            continue
        extra_values.append(max(0.0, peak - baseline_value))

    sample_summary = summarize_gpu_samples(samples_path)
    metrics: dict[str, Any] = {
        "method": job.method,
        "temperature": job.temperature,
        "repeat": job.repeat,
        "seed": job.seed,
        "gpu": gpu,
        "return_code": return_code,
        "ok": return_code == 0 and total > 0,
        "out_dir": str(job.out_dir),
        "result_path": str(job.result_path),
        "vram_path": str(job.vram_path),
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
        "prefill_ms_mean": safe_mean(prefill_values),
        "prefill_ms_max": safe_max(prefill_values),
        "e2e_ms_mean": safe_mean(e2e_values),
        "e2e_ms_max": safe_max(e2e_values),
        "orig_frames_mean": safe_mean(orig_values),
        "orig_frames_max": safe_max(orig_values),
        "used_frames_mean": safe_mean(used_values),
        "used_frames_max": safe_max(used_values),
        "frame_keep_ratio_mean": safe_mean(frame_ratios),
        "model_loaded_alloc_gb": model_loaded_alloc,
        "model_loaded_reserved_gb": model_loaded_reserved,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "inference_extra_peak_alloc_gb": safe_max(extra_values),
        "after_alloc_gb": after_alloc,
        "run_wall_s": end_s - start_s,
    }
    metrics.update(sample_summary)
    return metrics


def existing_success(job: Job) -> dict[str, Any] | None:
    for summary_path in (
        job.out_dir / "run_summary.json",
        legacy_run_dir(job.out_dir.parent.parent, job.method, job.temperature, job.repeat) / "run_summary.json",
    ):
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if summary.get("ok") is True:
            summary["resumed"] = True
            summary["resumed_from"] = str(summary_path.parent)
            return summary
    return None


def run_job(job: Job, gpu: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.resume:
        resumed = existing_success(job)
        if resumed is not None:
            print(f"[resume] {job.method} temp={job.temperature:g} run={job.repeat:02d}", flush=True)
            return resumed

    job.out_dir.mkdir(parents=True, exist_ok=True)
    (job.out_dir / "command.txt").write_text(shlex.join(job.command) + "\n", encoding="utf-8")
    (job.out_dir / "job.json").write_text(json.dumps(job, default=json_default, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[start] gpu={gpu} {job.method} temp={job.temperature:g} run={job.repeat:02d}", flush=True)
    start_s = time.time()
    with (job.out_dir / "driver_stdout.log").open("w", encoding="utf-8") as stdout_f:
        process = subprocess.Popen(
            job.command,
            cwd=str(repo_root()),
            env=env,
            stdout=stdout_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        sampler = GpuSampler(process.pid, gpu, job.out_dir / "gpu_samples.jsonl", args.gpu_sample_interval)
        sampler.start()
        return_code = process.wait()
        sampler.stop()
    end_s = time.time()

    metrics = summarize_run(job, return_code, start_s, end_s, gpu)
    (job.out_dir / "run_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    status = "ok" if metrics["ok"] else f"failed rc={return_code}"
    print(f"[done]  gpu={gpu} {job.method} temp={job.temperature:g} run={job.repeat:02d} {status}", flush=True)
    return metrics


def worker_loop(gpu: str, jobs_queue: queue.Queue[Job], args: argparse.Namespace, sink: list[dict[str, Any]], lock: threading.Lock) -> None:
    while True:
        try:
            job = jobs_queue.get_nowait()
        except queue.Empty:
            return
        try:
            metrics = run_job(job, gpu, args)
        except Exception as exc:
            metrics = {
                "method": job.method,
                "temperature": job.temperature,
                "repeat": job.repeat,
                "seed": job.seed,
                "gpu": gpu,
                "return_code": None,
                "ok": False,
                "out_dir": str(job.out_dir),
                "error": repr(exc),
            }
            job.out_dir.mkdir(parents=True, exist_ok=True)
            (job.out_dir / "run_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            print(f"[error] gpu={gpu} {job.method} temp={job.temperature:g} run={job.repeat:02d}: {exc!r}", flush=True)
        with lock:
            sink.append(metrics)
        jobs_queue.task_done()


def write_run_metrics(output_root: Path, metrics: list[dict[str, Any]]) -> None:
    path = output_root / "run_metrics.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(metrics, key=lambda x: (x["method"], x["temperature"], x["repeat"])):
            handle.write(json.dumps(row) + "\n")


def aggregate(metrics: list[dict[str, Any]], requested_runs: int) -> dict[str, Any]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in metrics:
        grouped.setdefault((row["method"], float(row["temperature"])), []).append(row)

    summary: dict[str, Any] = {}
    for (method, temp), rows in sorted(grouped.items()):
        successful = [row for row in rows if row.get("ok") is True]
        key = f"{method}/temp_{temp_label(temp)}"
        metric_stats = {}
        for field in STAT_FIELDS:
            values: list[float] = []
            for row in successful:
                value = row.get(field)
                if value is None or isinstance(value, bool):
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            metric_stats[field] = summarize_values(values)
        summary[key] = {
            "method": method,
            "temperature": temp,
            "runs_requested": requested_runs,
            "runs_observed": len(rows),
            "runs_successful": len(successful),
            "runs_failed": len(rows) - len(successful),
            "stats": metric_stats,
        }
    return summary


def write_summary_csv(output_root: Path, summary: dict[str, Any]) -> None:
    path = output_root / "summary_by_config.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "temperature",
                "runs_requested",
                "runs_successful",
                "runs_failed",
                "metric",
                "n",
                "mean",
                "stddev",
                "variance",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        for entry in summary.values():
            for metric, stats in entry["stats"].items():
                writer.writerow(
                    {
                        "method": entry["method"],
                        "temperature": entry["temperature"],
                        "runs_requested": entry["runs_requested"],
                        "runs_successful": entry["runs_successful"],
                        "runs_failed": entry["runs_failed"],
                        "metric": metric,
                        **stats,
                    }
                )


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(output_root: Path, summary: dict[str, Any], jobs: list[Job], gpus: list[str], args: argparse.Namespace) -> None:
    path = output_root / "SUMMARY.md"
    methods = parse_methods(args.methods)
    temps = parse_temperatures(args.temperatures)
    lines = [
        "# Qwen2.5-Omni Benchmark Matrix",
        "",
        f"Started output root: `{output_root}`",
        f"Methods: {', '.join(methods)}",
        f"Temperatures: {', '.join(f'{temp:g}' for temp in temps)}",
        f"Repeats per method-temperature: {args.runs}",
        f"Total eval executions: {len(methods)} methods x {len(temps)} temperatures x {args.runs} repeats = {len(jobs)}",
        f"GPUs used: {', '.join(gpus)}",
        "",
        "## Memory Fields",
        "",
        "- `model_loaded_alloc_gb`: `torch.cuda.memory_allocated()` right after model load. This is closest to model weights plus persistent buffers, but it excludes CUDA context and non-PyTorch allocations.",
        "- `peak_alloc_gb`: per-question PyTorch allocated peak after resetting peak stats just before inference. This includes loaded model memory plus inference-time allocations such as activations and KV cache.",
        "- `inference_extra_peak_alloc_gb`: `peak_alloc_gb - before_alloc_gb`, an approximate per-question incremental inference overhead.",
        "- `peak_reserved_gb`: PyTorch caching allocator reservation, which can be higher than allocated memory.",
        "- `gpu_process_peak_gb`: peak process VRAM from `nvidia-smi`, sampled by the harness. This is the best top-line VRAM number because it includes CUDA context, allocator reservation, and non-PyTorch allocations.",
        "- `gpu_total_peak_gb`: total memory used on that GPU during the run. Use this carefully if other processes share the GPU.",
        "",
        "## Frame Fields",
        "",
        "- `orig_frames_*`: mean/max decoded frame count per question after the normal dataset caps (`--fps`, `--max_frames_videomme`, `--max_frames_other`) are applied.",
        "- `used_frames_*`: mean/max frame count actually kept for inference. For baseline, GPTQ, AWQ, OmniZip, and SnapKV MixKV this equals `orig_frames`; for DivPrune/ReDiPrune it reflects the pruned frame count.",
        "- `frame_keep_ratio_mean`: average `used_frames / orig_frames` across questions in a run.",
        "",
        "## Summary",
        "",
        "| Method | Temp | Successful Runs | Accuracy Mean | Accuracy Std | Prefill Mean ms | Frames Mean | Keep Ratio | Peak Alloc GB | Process Peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in summary.values():
        stats = entry["stats"]
        lines.append(
            "| "
            + " | ".join(
                [
                    entry["method"],
                    fmt(entry["temperature"]),
                    f"{entry['runs_successful']}/{entry['runs_requested']}",
                    fmt(stats["accuracy"]["mean"]),
                    fmt(stats["accuracy"]["stddev"]),
                    fmt(stats["prefill_ms_mean"]["mean"]),
                    fmt(stats["used_frames_mean"]["mean"]),
                    fmt(stats["frame_keep_ratio_mean"]["mean"]),
                    fmt(stats["peak_alloc_gb"]["mean"]),
                    fmt(stats["gpu_process_peak_gb"]["mean"]),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(jobs: list[Job], gpus: list[str], args: argparse.Namespace) -> None:
    method_count = len(parse_methods(args.methods))
    temp_count = len(parse_temperatures(args.temperatures))
    print(f"Would run {len(jobs)} eval executions on GPU(s): {', '.join(gpus)}")
    print(f"Combination count: {method_count} methods x {temp_count} temperatures = {method_count * temp_count} method-temperature configs")
    print(f"Execution count with repeats: {len(jobs)}")
    for job in jobs[: min(10, len(jobs))]:
        print(f"\n# {job.method} temp={job.temperature:g} run={job.repeat:02d} seed={job.seed}")
        print(shlex.join(job.command))
    if len(jobs) > 10:
        print(f"\n... {len(jobs) - 10} more commands omitted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5-Omni benchmark matrix and summarize repeated runs.")
    parser.add_argument("--metadata", default="videos/metadata.json")
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--methods", default=",".join(METHODS), help=f"Comma-separated subset of: {', '.join(METHODS)}")
    parser.add_argument("--temperatures", default=",".join(str(temp) for temp in DEFAULT_TEMPERATURES))
    parser.add_argument("--runs", type=int, default=10, help="Repeats per method-temperature config.")
    parser.add_argument("--seed_base", type=int, default=20260415)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", default="auto", help="'auto', 'any', 'all', or comma-separated ids like 5,7.")
    parser.add_argument("--min_free_gb", type=float, default=30.0, help="Minimum free VRAM for --gpus auto/any.")
    parser.add_argument("--parallel", type=int, default=0, help="Concurrent evals. 0 means one worker per selected GPU.")
    parser.add_argument("--gpu_sample_interval", type=float, default=1.0, help="Seconds between nvidia-smi samples. 0 disables sampling.")
    parser.add_argument("--resume", action="store_true", help="Skip run dirs with a successful run_summary.json.")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=100352)
    parser.add_argument("--max_frames_videomme", type=int, default=768)
    parser.add_argument("--max_frames_other", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--category", default=None)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--measure_prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra_eval_args", nargs=argparse.REMAINDER, default=[])

    parser.add_argument("--qwen_model", default=None, help="Optional path for baseline/OmniZip/pruning/MixKV full Qwen checkpoint.")
    parser.add_argument("--gptq_model", default=None, help="Optional path for Qwen2.5-Omni GPTQ checkpoint.")
    parser.add_argument("--awq_model", default=None, help="Optional path for Qwen2.5-Omni AWQ checkpoint.")

    parser.add_argument("--rho_audio", type=float, default=0.3)
    parser.add_argument("--rho_video", type=float, default=0.6)
    parser.add_argument("--omnizip_g", type=int, default=3)
    parser.add_argument("--contextual_ratio", type=float, default=0.05)

    parser.add_argument("--divprune_subset_ratio", type=float, default=0.5)
    parser.add_argument("--divprune_prune_mode", default="frame", choices=["frame", "token"])

    parser.add_argument("--mixkv_budget", type=int, default=256)
    parser.add_argument("--mixkv_window_size", type=int, default=32)
    parser.add_argument("--mixkv_select_method", default="snapkv", choices=["snapkv", "attn", "vnorm", "headwisemixkv"])
    parser.add_argument("--mixkv_head_score_path", default=None)

    parser.add_argument("--rediprune_subset_ratio", type=float, default=0.5)
    parser.add_argument("--rediprune_prune_mode", default="frame", choices=["frame", "token"])
    parser.add_argument("--rediprune_alpha", type=float, default=0.5)
    parser.add_argument("--rediprune_tau", type=float, default=0.0)

    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.output_root is None:
        args.output_root = repo_root() / "runs" / f"qwen25_matrix_{timestamp()}"
    else:
        args.output_root = args.output_root.resolve()
    args.extra_eval_args = args.extra_eval_args or []
    return args


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args)
    gpus = select_gpus(args.gpus, args.min_free_gb, args.dry_run)
    if not gpus:
        raise SystemExit("No GPUs selected.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(args.output_root),
        "gpus": gpus,
        "methods": parse_methods(args.methods),
        "temperatures": parse_temperatures(args.temperatures),
        "runs": args.runs,
        "total_jobs": len(jobs),
        "args": vars(args),
    }
    (args.output_root / "plan.json").write_text(json.dumps(plan, default=json_default, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        dry_run(jobs, gpus, args)
        return 0

    max_workers = args.parallel if args.parallel > 0 else len(gpus)
    max_workers = max(1, min(max_workers, len(gpus)))
    selected_gpus = gpus[:max_workers]

    jobs_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        jobs_queue.put(job)

    metrics: list[dict[str, Any]] = []
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_loop, gpu, jobs_queue, args, metrics, lock) for gpu in selected_gpus]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    write_run_metrics(args.output_root, metrics)
    summary = aggregate(metrics, args.runs)
    (args.output_root / "summary_by_config.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(args.output_root, summary)
    write_markdown(args.output_root, summary, jobs, selected_gpus, args)

    failed = sum(1 for row in metrics if row.get("ok") is not True)
    print(f"\nWrote benchmark outputs to {args.output_root}", flush=True)
    print(f"Completed {len(metrics)} runs; failed or empty runs: {failed}", flush=True)
    print(f"Summary: {args.output_root / 'SUMMARY.md'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
