"""
A3 — TSP-layer sensitivity sweep.

Runs the KL-vs-baseline harness across the 12 candidate TSP layers
{0, 3, 5, 7, 10, 12, 15, 18, 20, 21, 25, 27} for the 28-layer Qwen2.5-Omni
Thinker, holding tsp_length and other hyperparameters fixed. Loads the
model once and reuses it; baseline logits are computed once per prompt
and cached across sweep points.

Outputs:
  fastkv_omni/outputs/a3_tsp_sweep/sweep.jsonl   — one record per (tsp_idx, prompt)
  fastkv_omni/outputs/a3_tsp_sweep/sweep.csv     — flattened (tsp_idx, kl_mean, kl_max, ...)
  fastkv_omni/outputs/a3_tsp_sweep/sweep.png     — KL vs tsp_idx (one line per prompt)

Sanity expectations:
  - tsp_idx = 27 (last layer): KL ≈ 0   (no downstream layers to be affected)
  - tsp_idx = 0  (first layer): KL is largest (all 27 downstream layers see pruned set)
  - if KL at tsp_idx=27 is NOT ~0, the patch has a bug — investigate before
    interpreting any other point.

Run:
    python a3_tsp_sweep.py --model_path /data/armaan/models/Qwen2.5-Omni-7B \\
        --tsp_length 2048
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from kl_vs_baseline import _load_model, load_prompts, run_one_config  # noqa: E402


DEFAULT_TSP_IDXS = [0, 3, 5, 7, 10, 12, 15, 18, 20, 21, 25, 27]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data/armaan/models/Qwen2.5-Omni-7B")
    parser.add_argument("--tsp_length", type=int, default=2048)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--pooling", default="avgpool", choices=["avgpool", "maxpool"])
    parser.add_argument("--prompts_json", default=None)
    parser.add_argument("--tsp_idxs", default=",".join(map(str, DEFAULT_TSP_IDXS)),
                        help="Comma-separated list of TSP layer indices to sweep.")
    parser.add_argument("--out_dir", default=str(HERE.parent / "outputs" / "a3_tsp_sweep"))
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip matplotlib plotting (useful on headless boxes without matplotlib).")
    args = parser.parse_args()

    tsp_idxs = [int(x) for x in args.tsp_idxs.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "sweep.jsonl")
    out_csv = os.path.join(args.out_dir, "sweep.csv")
    out_png = os.path.join(args.out_dir, "sweep.png")

    model, processor = _load_model(args.model_path)
    prompts = load_prompts(args.prompts_json)

    all_records = []
    baseline_cache: dict = {}
    for k, tsp_idx in enumerate(tsp_idxs):
        print(f"\n[sweep] ({k+1}/{len(tsp_idxs)}) tsp_idx={tsp_idx}")
        recs, baseline_cache = run_one_config(
            model, processor, prompts,
            tsp_idx=tsp_idx, tsp_length=args.tsp_length,
            window_size=args.window_size, kernel_size=args.kernel_size, pooling=args.pooling,
            baseline_cache=baseline_cache,
        )
        all_records.extend(recs)

    # JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        for r in all_records:
            writer.writerow(r)

    print(f"\n[sweep] wrote {len(all_records)} records → {out_jsonl}")
    print(f"[sweep] wrote csv → {out_csv}")

    # Plot
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[sweep] matplotlib not available; skipping plot.")
            return

        # Group by prompt_id
        by_prompt: dict = {}
        for r in all_records:
            by_prompt.setdefault(r["prompt_id"], []).append(r)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for pid, recs in by_prompt.items():
            recs_sorted = sorted(recs, key=lambda x: x["tsp_idx"])
            xs = [r["tsp_idx"] for r in recs_sorted]
            ys_mean = [r["kl_mean"] for r in recs_sorted]
            ys_max = [r["kl_max"] for r in recs_sorted]
            ax1.plot(xs, ys_mean, marker="o", label=pid)
            ax2.plot(xs, ys_max, marker="o", label=pid)
        for ax, title in ((ax1, "Mean KL(baseline ‖ patched)"),
                          (ax2, "Max KL(baseline ‖ patched)")):
            ax.set_xlabel("TSP layer index")
            ax.set_ylabel("KL (nats)")
            ax.set_title(title)
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"A3 — TSP layer sweep  (tsp_length={args.tsp_length}, "
                     f"window={args.window_size}, pooling={args.pooling})")
        fig.tight_layout()
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        print(f"[sweep] wrote plot → {out_png}")


if __name__ == "__main__":
    main()
