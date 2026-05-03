from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "resultswprefill"
OUT_DIR = ROOT / "figures"

METHODS = [
    ("Baseline", "baseline"),
    ("OmniZip", "omnizip"),
    ("GPTQ-\nInt4", "gptq"),
    ("AWQ", "awq"),
    ("DivPrune", "divprune"),
    ("MixKV", "mixkv"),
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_metrics() -> tuple[list[str], list[float], list[float], list[float]]:
    labels: list[str] = []
    scores: list[float] = []
    memories: list[float] = []
    prefill_times: list[float] = []

    for label, dirname in METHODS:
        rows = load_jsonl(RESULTS_DIR / dirname / "results.jsonl")
        vram_rows = load_jsonl(RESULTS_DIR / dirname / "vram_log.jsonl")

        labels.append(label)
        scores.append(100 * sum(bool(row["correct"]) for row in rows) / len(rows))
        memories.append(max(float(row["peak_alloc_gb"]) for row in vram_rows))
        prefill_times.append(mean(float(row["prefill_ms"]) for row in rows))

    return labels, scores, memories, prefill_times


def annotate_bars(ax, bars, color, dy=0.4):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            f"{value:.1f}" if value < 100 else f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
        )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    labels, scores, memories, prefill_times = collect_metrics()

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=220)
    fig.patch.set_alpha(0)

    panel = FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.94,
        transform=fig.transFigure,
        boxstyle="round,pad=0.018,rounding_size=0.07",
        linewidth=0,
        facecolor="#f2efee",
        zorder=-10,
    )
    fig.add_artist(panel)
    ax.set_facecolor("#f2efee")

    score_color = "#f59b53"
    memory_color = "#b8767b"

    score_bars = ax.bar(x - width / 2, scores, width, label="Avg. Score", color=score_color, edgecolor="none")
    memory_bars = ax.bar(x + width / 2, memories, width, label="Memory (GB)", color=memory_color, edgecolor="none")

    ax2 = ax.twinx()
    line = ax2.plot(
        x,
        prefill_times,
        color="black",
        marker="o",
        markersize=4.2,
        linewidth=1.4,
        label="Prefilling Time (ms)",
        zorder=5,
    )[0]

    annotate_bars(ax, score_bars, score_color)
    annotate_bars(ax, memory_bars, memory_color)

    for idx, value in enumerate(prefill_times):
        ax2.annotate(
            f"{value:,.0f}",
            (x[idx], value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(0, max(max(scores), max(memories)) * 1.28)
    ax2.set_ylim(0, max(prefill_times) * 1.22)

    ax.set_ylabel("Avg. Score and Memory", fontsize=9)
    ax2.set_ylabel("Prefilling Time (ms)", fontsize=9)

    ax.tick_params(axis="y", labelsize=8, length=0)
    ax2.tick_params(axis="y", labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)

    for spine in ax.spines.values():
        spine.set_color("#6d6663")
        spine.set_linewidth(0.7)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.grid(axis="y", color="#d8d2d0", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)

    handles = [score_bars, memory_bars, line]
    legend = ax.legend(
        handles,
        ["Avg. Score", "Memory (GB)", "Prefilling Time (ms)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
        fontsize=9,
        handlelength=0.9,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    for text in legend.get_texts():
        text.set_color("#222")

    ax.text(
        0.5,
        -0.17,
        "(c)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#7f5b4d",
    )

    fig.tight_layout(pad=1.6)
    outputs = [
        (OUT_DIR / "qwen25_prefill_score_memory_time.png", OUT_DIR / "qwen25_prefill_score_memory_time.pdf"),
        (
            OUT_DIR / "qwen25_prefill_avgscore_memory_time.png",
            OUT_DIR / "qwen25_prefill_avgscore_memory_time.pdf",
        ),
    ]
    for png_path, pdf_path in outputs:
        fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
