from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import xlsxwriter


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "10x" / "qwen25_matrix_gpu7_all7_snapkv"
OUT_XLSX = ROOT / "10x" / "qwen25_prefill_time_line_graph.xlsx"
OUT_PNG = ROOT / "10x" / "qwen25_prefill_time_line_graph.png"
OUT_SVG = ROOT / "10x" / "qwen25_prefill_time_line_graph.svg"

METHODS = [
    ("Baseline", "baseline"),
    ("OmniZip", "omnizip"),
    ("DivPrune", "divprune"),
    ("ReDiPrune", "rediprune"),
    ("MixKV", "mixkv"),
]


def load_run_summaries() -> dict[str, dict[float, list[dict]]]:
    grouped: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for _, dirname in METHODS:
        method_dir = RESULTS_ROOT / dirname
        if not method_dir.exists():
            raise FileNotFoundError(f"Missing method directory: {method_dir}")

        for summary_path in sorted(method_dir.glob("temp_*-run_*/run_summary.json")):
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)

            if not summary.get("ok"):
                continue

            grouped[dirname][float(summary["temperature"])].append(summary)

    return grouped


def summarize(grouped: dict[str, dict[float, list[dict]]]) -> tuple[list[dict], list[dict]]:
    per_temp_rows: list[dict] = []
    overall_rows: list[dict] = []

    for label, dirname in METHODS:
        method_values: list[float] = []

        for temp in sorted(grouped[dirname]):
            rows = grouped[dirname][temp]
            values = [float(row["prefill_ms_mean"]) for row in rows]
            if not values:
                continue

            method_values.extend(values)
            per_temp_rows.append(
                {
                    "method": label,
                    "temperature": temp,
                    "runs": len(values),
                    "prefill_mean_ms": mean(values),
                    "prefill_sd_ms": stdev(values) if len(values) > 1 else 0.0,
                    "prefill_min_ms": min(values),
                    "prefill_max_ms": max(values),
                }
            )

        if not method_values:
            raise ValueError(f"No successful prefill runs found for {label}")

        overall_rows.append(
            {
                "method": label,
                "runs": len(method_values),
                "prefill_mean_ms": mean(method_values),
                "prefill_sd_ms": stdev(method_values) if len(method_values) > 1 else 0.0,
                "prefill_min_ms": min(method_values),
                "prefill_max_ms": max(method_values),
            }
        )

    return overall_rows, per_temp_rows


def write_png(overall_rows: list[dict]) -> None:
    plt.rcParams["svg.fonttype"] = "none"

    labels = [row["method"] for row in overall_rows]
    values = [row["prefill_mean_ms"] for row in overall_rows]
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=220)
    fig.patch.set_facecolor("#f2efee")
    ax.set_facecolor("#f2efee")

    ax.plot(
        x,
        values,
        color="black",
        marker="o",
        markersize=4.5,
        linewidth=1.7,
        zorder=3,
    )

    for idx, value in enumerate(values):
        ax.annotate(
            f"{value:.0f}",
            (idx, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="black",
        )

    ax.set_title("Prefill Time", fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("Prefill Time (ms)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_xlim(-0.35, len(labels) - 0.65)
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", color="#d8d2d0", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#6d6663")
        spine.set_linewidth(0.8)

    fig.tight_layout(pad=1.3)
    fig.savefig(OUT_PNG, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(OUT_SVG, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def write_xlsx(overall_rows: list[dict], per_temp_rows: list[dict]) -> None:
    workbook = xlsxwriter.Workbook(OUT_XLSX)
    chart_sheet = workbook.add_worksheet("Prefill Line")
    detail_sheet = workbook.add_worksheet("Per Temperature")

    title_fmt = workbook.add_format({"bold": True, "font_size": 16})
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#F2EFEE", "border": 1})
    text_fmt = workbook.add_format({"border": 1})
    number_fmt = workbook.add_format({"border": 1, "num_format": "0.0"})
    integer_fmt = workbook.add_format({"border": 1, "num_format": "0"})

    chart_sheet.set_tab_color("#000000")
    chart_sheet.set_column("A:A", 16)
    chart_sheet.set_column("B:B", 12)
    chart_sheet.set_column("C:F", 15)
    chart_sheet.write("A1", "Prefill Time Line Graph", title_fmt)

    overall_headers = [
        "Method",
        "Runs",
        "Prefill Mean (ms)",
        "Prefill SD (ms)",
        "Prefill Min (ms)",
        "Prefill Max (ms)",
    ]
    for col, header in enumerate(overall_headers):
        chart_sheet.write(22, col, header, header_fmt)

    for row_idx, row in enumerate(overall_rows, start=23):
        chart_sheet.write(row_idx, 0, row["method"], text_fmt)
        chart_sheet.write_number(row_idx, 1, row["runs"], integer_fmt)
        chart_sheet.write_number(row_idx, 2, row["prefill_mean_ms"], number_fmt)
        chart_sheet.write_number(row_idx, 3, row["prefill_sd_ms"], number_fmt)
        chart_sheet.write_number(row_idx, 4, row["prefill_min_ms"], number_fmt)
        chart_sheet.write_number(row_idx, 5, row["prefill_max_ms"], number_fmt)

    line_chart = workbook.add_chart({"type": "line"})
    first_data_row = 23
    last_data_row = first_data_row + len(overall_rows) - 1
    line_chart.add_series(
        {
            "name": "Prefill Time (ms)",
            "categories": ["Prefill Line", first_data_row, 0, last_data_row, 0],
            "values": ["Prefill Line", first_data_row, 2, last_data_row, 2],
            "line": {"color": "#000000", "width": 1.75},
            "marker": {
                "type": "circle",
                "size": 5,
                "border": {"color": "#000000"},
                "fill": {"color": "#000000"},
            },
            "data_labels": {
                "value": True,
                "num_format": "0",
                "font": {"bold": True, "color": "#000000", "size": 10},
                "position": "above",
            },
        }
    )
    line_chart.set_title({"name": "Prefill Time", "name_font": {"bold": True, "size": 15}})
    line_chart.set_x_axis({"name": "", "num_font": {"bold": True, "size": 10}})
    line_chart.set_y_axis(
        {
            "name": "Prefill Time (ms)",
            "major_gridlines": {"visible": True, "line": {"color": "#D8D2D0"}},
            "min": 0,
            "max": max(row["prefill_mean_ms"] for row in overall_rows) * 1.2,
            "num_format": "0",
        }
    )
    line_chart.set_legend({"none": True})
    line_chart.set_chartarea({"fill": {"color": "#F2EFEE"}, "border": {"color": "#6D6663"}})
    line_chart.set_plotarea({"fill": {"color": "#F2EFEE"}, "border": {"color": "#6D6663"}})
    line_chart.set_size({"width": 680, "height": 390})
    chart_sheet.insert_chart("A3", line_chart)

    detail_sheet.set_column("A:A", 16)
    detail_sheet.set_column("B:C", 12)
    detail_sheet.set_column("D:G", 18)
    detail_headers = [
        "Method",
        "Temperature",
        "Runs",
        "Prefill Mean (ms)",
        "Prefill SD (ms)",
        "Prefill Min (ms)",
        "Prefill Max (ms)",
    ]
    for col, header in enumerate(detail_headers):
        detail_sheet.write(0, col, header, header_fmt)

    for row_idx, row in enumerate(per_temp_rows, start=1):
        detail_sheet.write(row_idx, 0, row["method"], text_fmt)
        detail_sheet.write_number(row_idx, 1, row["temperature"], number_fmt)
        detail_sheet.write_number(row_idx, 2, row["runs"], integer_fmt)
        detail_sheet.write_number(row_idx, 3, row["prefill_mean_ms"], number_fmt)
        detail_sheet.write_number(row_idx, 4, row["prefill_sd_ms"], number_fmt)
        detail_sheet.write_number(row_idx, 5, row["prefill_min_ms"], number_fmt)
        detail_sheet.write_number(row_idx, 6, row["prefill_max_ms"], number_fmt)

    chart_sheet.freeze_panes(23, 0)
    detail_sheet.freeze_panes(1, 0)
    workbook.close()


def main() -> None:
    grouped = load_run_summaries()
    overall_rows, per_temp_rows = summarize(grouped)
    write_png(overall_rows)
    write_xlsx(overall_rows, per_temp_rows)
    print(f"Wrote {OUT_XLSX}")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_SVG}")


if __name__ == "__main__":
    main()
