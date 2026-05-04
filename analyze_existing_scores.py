"""
Compute richer metrics from existing early_layer_relevance/scores.jsonl.

Uses only the raw per-token score arrays already stored at layers 0 and 1:

  1. Top-k Jaccard between questions on same video
       k_audio = round(0.60 * T_audio)   (OmniZip default: drop 40%)
       k_video = round(0.30 * T_video)   (OmniZip default: drop 70%)
     Answers: would two different questions make the same pruning decision?

  2. Separation score at pruning threshold
       (mean_score[kept] − mean_score[dropped]) / std(all scores)
     Answers: how cleanly does the signal separate kept vs dropped tokens?

  3. Temporal autocorrelation of the score vector
       Pearson(scores[t], scores[t+1])
     Answers: is the saliency temporally smooth (speech blocks) or scattered?

Outputs:
  {out_dir}/scores_analysis.jsonl  — one record per video
  {out_dir}/scores_analysis.png    — 3-panel summary plot
  {out_dir}/scores_analysis.md     — human-readable report

Run:
  python analyze_existing_scores.py
  python analyze_existing_scores.py --scores vizzing/early_layer_relevance/scores.jsonl
"""

import argparse
import json
import math
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--scores",   default=os.path.join(REPO_ROOT, "vizzing", "early_layer_relevance", "scores.jsonl"))
parser.add_argument("--out_dir",  default=os.path.join(REPO_ROOT, "vizzing", "analysis"))
parser.add_argument("--rho_audio_drop", type=float, default=0.40, help="OmniZip audio drop rate")
parser.add_argument("--rho_video_drop", type=float, default=0.70, help="OmniZip video drop rate")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

AUDIO_KEEP = 1.0 - args.rho_audio_drop   # 0.60
VIDEO_KEEP = 1.0 - args.rho_video_drop   # 0.30
LAYERS = ["0", "1"]

# ── Pure-python helpers (no numpy dependency) ─────────────────────────────────

def _mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")

def _std(lst):
    if len(lst) < 2: return 0.0
    m = _mean(lst)
    return math.sqrt(sum((x - m) ** 2 for x in lst) / len(lst))

def pearson(x, y):
    n = len(x)
    if n < 2: return float("nan")
    mx, my = _mean(x), _mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx  = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    dy  = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return (num / (dx * dy)) if dx > 0 and dy > 0 else float("nan")

def top_k_set(arr, k):
    return set(i for i, _ in sorted(enumerate(arr), key=lambda t: -t[1])[:k])

def jaccard(s1, s2):
    union = len(s1 | s2)
    return len(s1 & s2) / union if union > 0 else 1.0

def separation(arr, keep_frac):
    """(mean_kept − mean_dropped) / std_all at the given keep fraction."""
    n = len(arr)
    if n < 2: return float("nan")
    k = max(1, round(keep_frac * n))
    if k >= n: return float("nan")
    sorted_desc = sorted(arr, reverse=True)
    kept    = sorted_desc[:k]
    dropped = sorted_desc[k:]
    s = _std(arr)
    if s == 0: return 0.0
    return (_mean(kept) - _mean(dropped)) / s

def temporal_autocorr(arr):
    if len(arr) < 2: return float("nan")
    return pearson(arr[:-1], arr[1:])

# ── Load and group all records by (dataset, video_id) ────────────────────────
print(f"Loading {args.scores} …")
groups = defaultdict(list)   # key: (dataset, video_id) → list of records
n_records = 0
with open(args.scores) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        key = (r.get("dataset", "unknown"), r["video_id"])
        groups[key].append(r)
        n_records += 1

print(f"Loaded {n_records} records across {len(groups)} videos.")

multi_q_groups = {k: v for k, v in groups.items() if len(v) >= 2}
print(f"Videos with ≥2 questions: {len(multi_q_groups)}\n")

# ── Per-dataset accumulators ──────────────────────────────────────────────────
# Structure: acc[metric][layer][modality] = list of values (one per video)
METRICS = ["jaccard", "separation", "autocorr"]
MODS    = ["audio", "video"]

acc = {
    m: {li: {mod: [] for mod in MODS} for li in LAYERS}
    for m in METRICS
}
# Also track per-dataset
ds_acc = defaultdict(lambda: {
    metric: {layer: {mod: [] for mod in MODS} for layer in LAYERS}
    for metric in METRICS
})

out_records = []
log_path = os.path.join(args.out_dir, "scores_analysis.jsonl")

with open(log_path, "w") as log_f:
    for (dataset, video_id), records in sorted(multi_q_groups.items()):
        records_sorted = sorted(records, key=lambda r: r.get("question_idx", 0))
        n_q = len(records_sorted)

        video_result = {
            "dataset":   dataset,
            "video_id":  video_id,
            "n_questions": n_q,
        }

        for li in LAYERS:
            # ── collect per-question arrays for this layer ──────────────────
            a_vecs = []
            v_vecs = []
            for rec in records_sorted:
                a = rec.get("audio_scores", {}).get(li)
                v = rec.get("video_scores", {}).get(li)
                if a: a_vecs.append(a)
                if v: v_vecs.append(v)

            # ── Jaccard ──────────────────────────────────────────────────────
            def pairwise_jaccard(vecs, keep_frac):
                if len(vecs) < 2: return float("nan")
                jvals = []
                for v1, v2 in combinations(vecs, 2):
                    k = max(1, round(keep_frac * len(v1)))
                    jvals.append(jaccard(top_k_set(v1, k), top_k_set(v2, k)))
                return _mean(jvals)

            a_jac = pairwise_jaccard(a_vecs, AUDIO_KEEP)
            v_jac = pairwise_jaccard(v_vecs, VIDEO_KEEP)

            # ── Separation score (mean across questions) ─────────────────────
            a_sep = _mean([separation(v, AUDIO_KEEP) for v in a_vecs]) if a_vecs else float("nan")
            v_sep = _mean([separation(v, VIDEO_KEEP) for v in v_vecs]) if v_vecs else float("nan")

            # ── Temporal autocorrelation (mean across questions) ─────────────
            a_auto = _mean([temporal_autocorr(v) for v in a_vecs]) if a_vecs else float("nan")
            v_auto = _mean([temporal_autocorr(v) for v in v_vecs]) if v_vecs else float("nan")

            video_result[f"L{li}"] = {
                "audio_jaccard":   a_jac,   "video_jaccard":   v_jac,
                "audio_sep":       a_sep,   "video_sep":       v_sep,
                "audio_autocorr":  a_auto,  "video_autocorr":  v_auto,
            }

            # Accumulate
            for val, mod, metric in [
                (a_jac,  "audio", "jaccard"),   (v_jac,  "video", "jaccard"),
                (a_sep,  "audio", "separation"), (v_sep,  "video", "separation"),
                (a_auto, "audio", "autocorr"),  (v_auto, "video", "autocorr"),
            ]:
                if not math.isnan(val):
                    acc[metric][li][mod].append(val)
                    ds_acc[dataset][metric][li][mod].append(val)

        log_f.write(json.dumps(video_result) + "\n")
        out_records.append(video_result)

        print(f"[{dataset}] {video_id[:55]}  nQ={n_q}")
        for li in LAYERS:
            d = video_result[f"L{li}"]
            print(f"  L{li}  jaccard=a{d['audio_jaccard']:.3f}/v{d['video_jaccard']:.3f}"
                  f"  sep=a{d['audio_sep']:.3f}/v{d['video_sep']:.3f}"
                  f"  autocorr=a{d['audio_autocorr']:.3f}/v{d['video_autocorr']:.3f}")

# ── Global aggregate table ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("GLOBAL AGGREGATE (mean ± std across all videos)")
print("=" * 72)
print(f"\n{'Metric':>15}  {'Layer':>6}  {'Audio':>14}  {'Video':>14}  N")
for metric in METRICS:
    for li in LAYERS:
        a_lst = acc[metric][li]["audio"]
        v_lst = acc[metric][li]["video"]
        n = len(a_lst)
        print(f"  {metric:>13}  L{li:>1}     "
              f"{_mean(a_lst):.4f}±{_std(a_lst):.4f}  "
              f"{_mean(v_lst):.4f}±{_std(v_lst):.4f}  {n}")

print("\n" + "=" * 72)
print("PER-DATASET at L0 and L1 — Audio Jaccard")
print("=" * 72)
datasets = sorted(ds_acc.keys())
print(f"\n{'Dataset':>20}  N  {'AudJac_L0':>11}  {'AudJac_L1':>11}  {'AudSep_L0':>11}  {'AudSep_L1':>11}  {'AudAuto_L0':>12}  {'AudAuto_L1':>12}")
for ds in datasets:
    d = ds_acc[ds]
    n = len(d["jaccard"]["0"]["audio"])
    def gv(m, li): lst = d[m][li]["audio"]; return f"{_mean(lst):.4f}±{_std(lst):.4f}" if lst else "  N/A  "
    print(f"  {ds:>20}  {n}  "
          f"{gv('jaccard','0'):>14}  {gv('jaccard','1'):>14}  "
          f"{gv('separation','0'):>14}  {gv('separation','1'):>14}  "
          f"{gv('autocorr','0'):>15}  {gv('autocorr','1'):>15}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f"Score Analysis at L0 & L1  (n={len(out_records)} videos, "
    f"OmniZip rates: audio drop {args.rho_audio_drop:.0%}, video drop {args.rho_video_drop:.0%})",
    fontsize=12, y=1.02
)

metric_labels = {
    "jaccard":    ("Top-k Jaccard between questions",
                   f"Jaccard of top-{AUDIO_KEEP:.0%} audio / top-{VIDEO_KEEP:.0%} video kept sets",
                   "Higher = pruning decision is question-invariant"),
    "separation": ("Separation score at pruning threshold",
                   "(mean_kept − mean_dropped) / σ_all",
                   "Higher = signal separates kept/dropped tokens cleanly"),
    "autocorr":   ("Temporal autocorrelation of scores",
                   "Pearson(scores[t], scores[t+1])",
                   "Higher = saliency is block-structured (temporal coherence)"),
}

colors = {"audio": {"L0": "#27ae60", "L1": "#2ecc71"},
          "video": {"L0": "#2980b9", "L1": "#3498db"}}

x = [0, 1]   # L0, L1
width = 0.2

for ax, (metric, (title, ylabel, subtitle)) in zip(axes, metric_labels.items()):
    offsets = [-0.3, -0.1, 0.1, 0.3]
    bars = [
        ("audio", "L0", "0"), ("audio", "L1", "1"),
        ("video", "L0", "0"), ("video", "L1", "1"),
    ]
    for (mod, layer_label, li), offset in zip(bars, offsets):
        lst = acc[metric][li][mod]
        m = _mean(lst); s = _std(lst)
        ax.bar([0 + offset], [m], width=width,
               color=colors[mod][layer_label], alpha=0.85,
               label=f"{mod} {layer_label}", yerr=[s], capsize=4,
               error_kw={"elinewidth": 1.5})

    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([0]); ax.set_xticklabels(["L0 & L1"])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    if metric == "jaccard":
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)

plt.tight_layout()
out_plot = os.path.join(args.out_dir, "scores_analysis.png")
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot  → {out_plot}")
print(f"JSONL → {log_path}")

# ── Markdown report ───────────────────────────────────────────────────────────
md_path = os.path.join(args.out_dir, "scores_analysis.md")
with open(md_path, "w") as md:
    md.write("# Score Analysis — L0 & L1 Metrics\n\n")
    md.write(f"**Source:** `early_layer_relevance/scores.jsonl`  \n")
    md.write(f"**Videos with ≥2 questions:** {len(out_records)}  \n")
    md.write(f"**Pruning rates:** audio drop {args.rho_audio_drop:.0%}, video drop {args.rho_video_drop:.0%}  \n\n")
    md.write("## Global aggregate\n\n")
    md.write(f"| Metric | Layer | Audio | Video | N |\n")
    md.write(f"|--------|-------|-------|-------|---|\n")
    for metric in METRICS:
        for li in LAYERS:
            a = acc[metric][li]["audio"]; v = acc[metric][li]["video"]
            n = len(a)
            md.write(f"| {metric} | L{li} | "
                     f"{_mean(a):.4f} ± {_std(a):.4f} | "
                     f"{_mean(v):.4f} ± {_std(v):.4f} | {n} |\n")
    md.write("\n## Interpretation\n\n")

    # Jaccard
    jac_a0 = _mean(acc["jaccard"]["0"]["audio"])
    jac_a1 = _mean(acc["jaccard"]["1"]["audio"])
    md.write(f"**Top-k Jaccard:** Audio Jaccard at L0={jac_a0:.3f}, L1={jac_a1:.3f}. "
             f"{'High Jaccard confirms the pruning decision is nearly question-invariant.' if jac_a0 > 0.85 else 'Jaccard is moderate — questions do influence the top-k set.'}\n\n")

    # Separation
    sep_a0 = _mean(acc["separation"]["0"]["audio"])
    sep_a1 = _mean(acc["separation"]["1"]["audio"])
    md.write(f"**Separation score:** Audio at L0={sep_a0:.3f}, L1={sep_a1:.3f}. "
             f"{'Low separation at L0 confirms scores are near-uniform (flat distribution).' if sep_a0 < 0.5 else 'Score separation is meaningful even at L0.'} "
             f"{'L1 improves separation, suggesting the model begins differentiating by layer 1.' if sep_a1 > sep_a0 else ''}\n\n")

    # Autocorrelation
    auto_a0 = _mean(acc["autocorr"]["0"]["audio"])
    auto_a1 = _mean(acc["autocorr"]["1"]["audio"])
    md.write(f"**Temporal autocorrelation:** Audio at L0={auto_a0:.3f}, L1={auto_a1:.3f}. "
             f"{'High autocorrelation means the saliency signal is smooth/block-structured — adjacent audio tokens score similarly. This is consistent with the signal tracking speech segments rather than individual tokens.' if auto_a0 > 0.7 else 'Moderate autocorrelation — signal has some temporal structure but is not strongly block-structured.'}\n\n")

    md.write(f"**Plot:** `scores_analysis.png`  \n")
    md.write(f"**Per-video data:** `scores_analysis.jsonl`  \n")

print(f"Report → {md_path}")
