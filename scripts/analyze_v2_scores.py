"""
Compute Jaccard, separation, autocorrelation, and layer-to-layer Jaccard
from the layer_depth_experiment_v2 JSONL, which stores raw per-question
score arrays at L6 and L14 (via raw_audio_scores / raw_video_scores).

Produces:
  {out_dir}/v2_analysis.jsonl   — one record per video
  {out_dir}/v2_analysis.png     — comparison plot L0/L1 vs L6/L14
  {out_dir}/v2_analysis.md      — report

Run:
  python analyze_v2_scores.py
  python analyze_v2_scores.py --scores vizzing/layer_depth_experiment_v2/worldsense/layer_depth_stats.jsonl
"""

import argparse, json, math, os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--scores",  default=os.path.join(
    REPO_ROOT, "vizzing", "layer_depth_experiment_v2", "worldsense", "layer_depth_stats.jsonl"))
parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "vizzing", "analysis_v2"))
parser.add_argument("--rho_audio_drop", type=float, default=0.40)
parser.add_argument("--rho_video_drop", type=float, default=0.70)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

AUDIO_KEEP = 1.0 - args.rho_audio_drop   # 0.60
VIDEO_KEEP = 1.0 - args.rho_video_drop   # 0.30
RAW_LAYERS = ["6", "14"]

# ── helpers ───────────────────────────────────────────────────────────────────
def mean(lst): return sum(lst)/len(lst) if lst else float("nan")
def std(lst):
    if len(lst) < 2: return 0.0
    m = mean(lst)
    return math.sqrt(sum((x-m)**2 for x in lst)/len(lst))

def pearson(x, y):
    n = len(x)
    if n < 2: return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    dx  = math.sqrt(sum((xi-mx)**2 for xi in x))
    dy  = math.sqrt(sum((yi-my)**2 for yi in y))
    return (num/(dx*dy)) if dx>0 and dy>0 else float("nan")

def top_k_set(arr, k):
    return set(i for i,_ in sorted(enumerate(arr), key=lambda t: -t[1])[:k])

def jaccard(s1, s2):
    u = len(s1|s2)
    return len(s1&s2)/u if u>0 else 1.0

def separation(arr, keep_frac):
    n = len(arr)
    if n < 2: return float("nan")
    k = max(1, round(keep_frac * n))
    if k >= n: return float("nan")
    sd = sorted(arr, reverse=True)
    kept, dropped = sd[:k], sd[k:]
    s = std(arr)
    return (mean(kept)-mean(dropped))/s if s>0 else 0.0

def autocorr(arr):
    return pearson(arr[:-1], arr[1:]) if len(arr) >= 2 else float("nan")

def pairwise_jaccard(vecs, keep_frac):
    if len(vecs) < 2: return float("nan")
    vals = []
    for v1,v2 in combinations(vecs, 2):
        k = max(1, round(keep_frac * len(v1)))
        vals.append(jaccard(top_k_set(v1,k), top_k_set(v2,k)))
    return mean(vals)

def layer_to_layer_jaccard(vecs_l6, vecs_l14, keep_frac):
    """For each question, Jaccard between top-k at L6 and top-k at L14."""
    vals = []
    for v6, v14 in zip(vecs_l6, vecs_l14):
        if len(v6) != len(v14): continue
        k = max(1, round(keep_frac * len(v6)))
        vals.append(jaccard(top_k_set(v6,k), top_k_set(v14,k)))
    return mean(vals)

# ── load records ──────────────────────────────────────────────────────────────
print(f"Loading {args.scores} …")
records = []
with open(args.scores) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"Loaded {len(records)} video records\n")

# ── accumulators ──────────────────────────────────────────────────────────────
METRICS  = ["jaccard_q", "jaccard_l6_l14", "separation", "autocorr"]
MODS     = ["audio", "video"]
acc = {m: {li: {mod: [] for mod in MODS} for li in RAW_LAYERS} for m in METRICS}
# jaccard_l6_l14 is layer-to-layer, not per-layer, so store separately
l2l_acc = {mod: [] for mod in MODS}

out_records = []

for r in records:
    vid   = r["video_id"]
    ds    = r.get("dataset","?")
    ra    = r.get("raw_audio_scores", {})
    rv    = r.get("raw_video_scores", {})

    video_out = {"video_id": vid, "dataset": ds, "n_q": r.get("n_successful",0)}

    for li in RAW_LAYERS:
        a_vecs = ra.get(li, [])
        v_vecs = rv.get(li, [])

        # cross-Q Jaccard
        a_jac = pairwise_jaccard(a_vecs, AUDIO_KEEP)
        v_jac = pairwise_jaccard(v_vecs, VIDEO_KEEP)

        # separation (mean across questions)
        a_sep = mean([separation(v, AUDIO_KEEP) for v in a_vecs]) if a_vecs else float("nan")
        v_sep = mean([separation(v, VIDEO_KEEP) for v in v_vecs]) if v_vecs else float("nan")

        # temporal autocorrelation (mean across questions)
        a_auto = mean([autocorr(v) for v in a_vecs]) if a_vecs else float("nan")
        v_auto = mean([autocorr(v) for v in v_vecs]) if v_vecs else float("nan")

        video_out[f"L{li}"] = {
            "audio_jaccard_q": a_jac, "video_jaccard_q": v_jac,
            "audio_sep":       a_sep, "video_sep":       v_sep,
            "audio_autocorr":  a_auto,"video_autocorr":  v_auto,
        }

        for val, mod, metric in [
            (a_jac,  "audio", "jaccard_q"),  (v_jac,  "video", "jaccard_q"),
            (a_sep,  "audio", "separation"), (v_sep,  "video", "separation"),
            (a_auto, "audio", "autocorr"),   (v_auto, "video", "autocorr"),
        ]:
            if not math.isnan(val):
                acc[metric][li][mod].append(val)

    # layer-to-layer Jaccard (L6 vs L14)
    a6  = ra.get("6",  []);  a14 = ra.get("14", [])
    v6  = rv.get("6",  []);  v14 = rv.get("14", [])
    a_l2l = layer_to_layer_jaccard(a6, a14, AUDIO_KEEP)
    v_l2l = layer_to_layer_jaccard(v6, v14, VIDEO_KEEP)
    video_out["audio_jaccard_L6_L14"] = a_l2l
    video_out["video_jaccard_L6_L14"] = v_l2l
    if not math.isnan(a_l2l): l2l_acc["audio"].append(a_l2l)
    if not math.isnan(v_l2l): l2l_acc["video"].append(v_l2l)

    out_records.append(video_out)
    print(f"[{ds}] {vid[:55]}")
    for li in RAW_LAYERS:
        d = video_out[f"L{li}"]
        print(f"  L{li}  jac_q=a{d['audio_jaccard_q']:.3f}/v{d['video_jaccard_q']:.3f}"
              f"  sep=a{d['audio_sep']:.3f}/v{d['video_sep']:.3f}"
              f"  auto=a{d['audio_autocorr']:.3f}/v{d['video_autocorr']:.3f}")
    print(f"  L6→L14 Jaccard:  audio={video_out['audio_jaccard_L6_L14']:.4f}"
          f"  video={video_out['video_jaccard_L6_L14']:.4f}")

# ── print aggregate ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("GLOBAL AGGREGATE (mean ± std)")
print("="*70)
print(f"\n{'Metric':>16}  {'Layer':>6}  {'Audio':>14}  {'Video':>14}  N")
for metric in ["jaccard_q", "separation", "autocorr"]:
    for li in RAW_LAYERS:
        a = acc[metric][li]["audio"]; v = acc[metric][li]["video"]
        print(f"  {metric:>14}  L{li:>2}  "
              f"{mean(a):.4f}±{std(a):.4f}  "
              f"{mean(v):.4f}±{std(v):.4f}  {len(a)}")

print(f"\n  {'L6→L14 Jaccard':>14}   —   "
      f"audio={mean(l2l_acc['audio']):.4f}±{std(l2l_acc['audio']):.4f}  "
      f"video={mean(l2l_acc['video']):.4f}±{std(l2l_acc['video']):.4f}  {len(l2l_acc['audio'])}")

# ── load L0/L1 results for comparison ────────────────────────────────────────
L01_PATH = os.path.join(REPO_ROOT, "vizzing", "analysis", "scores_analysis.jsonl")
l01 = {"jaccard":{"0":{"audio":[],"video":[]},"1":{"audio":[],"video":[]}},
       "separation":{"0":{"audio":[],"video":[]},"1":{"audio":[],"video":[]}},
       "autocorr":{"0":{"audio":[],"video":[]},"1":{"audio":[],"video":[]}}}
if os.path.exists(L01_PATH):
    for line in open(L01_PATH):
        r2 = json.loads(line.strip())
        for li in ["0","1"]:
            d = r2.get(f"L{li}",{})
            for mod in MODS:
                for m,k in [("jaccard",f"{mod}_jaccard"),("separation",f"{mod}_sep"),("autocorr",f"{mod}_autocorr")]:
                    v2 = d.get(k)
                    if v2 is not None and not math.isnan(v2):
                        l01[m][li][mod].append(v2)
    print("\n" + "="*70)
    print("FULL COMPARISON: L0, L1, L6, L14 (Audio)")
    print("="*70)
    print(f"\n{'Metric':>14}  {'L0':>14}  {'L1':>14}  {'L6':>14}  {'L14':>14}")
    for metric, m2 in [("Cross-Q Jaccard","jaccard"),("Separation","separation"),("Autocorr","autocorr")]:
        row = []
        for li,src in [("0",l01[m2]["0"]["audio"]),("1",l01[m2]["1"]["audio"]),
                       ("6",acc["jaccard_q" if m2=="jaccard" else m2]["6"]["audio"]),
                       ("14",acc["jaccard_q" if m2=="jaccard" else m2]["14"]["audio"])]:
            row.append(f"{mean(src):.4f}±{std(src):.4f}")
        print(f"  {metric:>14}  {'  '.join(row)}")

# ── save JSONL ────────────────────────────────────────────────────────────────
out_path = os.path.join(args.out_dir, "v2_analysis.jsonl")
with open(out_path,"w") as f:
    for r in out_records:
        f.write(json.dumps(r)+"\n")

# ── plot ──────────────────────────────────────────────────────────────────────
layers_plot = ["L0","L1","L6","L14"]
colors_aud  = ["#a8e6cf","#27ae60","#5dade2","#1a5276"]
colors_vid  = ["#fad7a0","#e67e22","#c39bd3","#6c3483"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Metric comparison across Thinker layers  (n={len(records)} worldsense videos, "
    f"audio keep {AUDIO_KEEP:.0%} / video keep {VIDEO_KEEP:.0%})",
    fontsize=11, y=1.02
)

def get_vals(metric, li_str, mod):
    if li_str in ["0","1"]:
        m2 = {"jaccard":"jaccard","separation":"separation","autocorr":"autocorr"}[metric]
        return l01[m2][li_str][mod]
    else:
        m2 = {"jaccard":"jaccard_q","separation":"separation","autocorr":"autocorr"}[metric]
        return acc[m2][li_str][mod]

subplot_info = [
    ("jaccard",    "Cross-Q Jaccard at pruning threshold",
     "Jaccard of kept-token sets between question pairs",
     "Higher = pruning is question-invariant"),
    ("separation", "Separation score at pruning threshold",
     "(mean_kept − mean_dropped) / σ  |  uniform baseline ≈ 1.73",
     "Higher = signal cleanly separates kept vs dropped"),
    ("autocorr",   "Temporal autocorrelation",
     "Pearson(scores[t], scores[t+1])",
     "Higher = saliency is temporally block-structured"),
]

x = range(4)
width = 0.35

for ax, (metric, title, ylabel, subtitle) in zip(axes, subplot_info):
    a_means = [mean(get_vals(metric, li, "audio")) for li in ["0","1","6","14"]]
    a_stds  = [std(get_vals(metric,  li, "audio")) for li in ["0","1","6","14"]]
    v_means = [mean(get_vals(metric, li, "video")) for li in ["0","1","6","14"]]
    v_stds  = [std(get_vals(metric,  li, "video")) for li in ["0","1","6","14"]]

    xx = [i - width/2 for i in x]
    xv = [i + width/2 for i in x]
    ax.bar(xx, a_means, width, color=colors_aud, yerr=a_stds, capsize=3,
           label="Audio", alpha=0.85, error_kw={"elinewidth":1.5})
    ax.bar(xv, v_means, width, color=colors_vid, yerr=v_stds, capsize=3,
           label="Video", alpha=0.85, error_kw={"elinewidth":1.5})

    if metric == "separation":
        ax.axhline(1.73, color="gray", lw=1.2, ls="--", label="Uniform baseline")

    ax.set_xticks(list(x)); ax.set_xticklabels(layers_plot)
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    if metric == "jaccard": ax.set_ylim(0.85, 1.02)

# Add L6→L14 annotation to jaccard panel
axes[0].axhline(mean(l2l_acc["audio"]), color="#1a5276", lw=1.5, ls=":",
                label=f"L6→L14 Jac audio={mean(l2l_acc['audio']):.3f}")
axes[0].legend(fontsize=6.5)

plt.tight_layout()
out_plot = os.path.join(args.out_dir, "v2_analysis.png")
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()

# ── markdown report ───────────────────────────────────────────────────────────
md_path = os.path.join(args.out_dir, "v2_analysis.md")
with open(md_path,"w") as md:
    md.write("# Layer Depth Analysis — L6 & L14 Metrics\n\n")
    md.write(f"**Videos:** {len(records)} (worldsense)  \n")
    md.write(f"**Pruning rates:** audio drop {args.rho_audio_drop:.0%}, video drop {args.rho_video_drop:.0%}  \n\n")
    md.write("## Cross-Q Jaccard (pruning decision consistency)\n\n")
    md.write("| Layer | Audio | Video |\n|-------|-------|-------|\n")
    for li,src_a,src_v in [("L0",l01["jaccard"]["0"]["audio"],l01["jaccard"]["0"]["video"]),
                            ("L1",l01["jaccard"]["1"]["audio"],l01["jaccard"]["1"]["video"]),
                            ("L6",acc["jaccard_q"]["6"]["audio"],acc["jaccard_q"]["6"]["video"]),
                            ("L14",acc["jaccard_q"]["14"]["audio"],acc["jaccard_q"]["14"]["video"])]:
        md.write(f"| {li} | {mean(src_a):.4f} ± {std(src_a):.4f} | {mean(src_v):.4f} ± {std(src_v):.4f} |\n")
    md.write(f"\n**L6→L14 Jaccard** (same question, do L6 and L14 agree on which tokens to keep?):  \n")
    md.write(f"Audio = {mean(l2l_acc['audio']):.4f} ± {std(l2l_acc['audio']):.4f}  \n")
    md.write(f"Video = {mean(l2l_acc['video']):.4f} ± {std(l2l_acc['video']):.4f}  \n\n")
    md.write("## Separation score\n\n")
    md.write("Uniform baseline ≈ 1.73. Values above 1.73 indicate the signal is more discriminative than random.\n\n")
    md.write("| Layer | Audio | Video |\n|-------|-------|-------|\n")
    for li,src_a,src_v in [("L0",l01["separation"]["0"]["audio"],l01["separation"]["0"]["video"]),
                            ("L1",l01["separation"]["1"]["audio"],l01["separation"]["1"]["video"]),
                            ("L6",acc["separation"]["6"]["audio"],acc["separation"]["6"]["video"]),
                            ("L14",acc["separation"]["14"]["audio"],acc["separation"]["14"]["video"])]:
        above = "← above uniform" if mean(src_a) > 1.73 else "← below uniform"
        md.write(f"| {li} | {mean(src_a):.4f} ± {std(src_a):.4f} {above} | {mean(src_v):.4f} ± {std(src_v):.4f} |\n")
    md.write("\n## Temporal autocorrelation\n\n")
    md.write("| Layer | Audio | Video |\n|-------|-------|-------|\n")
    for li,src_a,src_v in [("L0",l01["autocorr"]["0"]["audio"],l01["autocorr"]["0"]["video"]),
                            ("L1",l01["autocorr"]["1"]["audio"],l01["autocorr"]["1"]["video"]),
                            ("L6",acc["autocorr"]["6"]["audio"],acc["autocorr"]["6"]["video"]),
                            ("L14",acc["autocorr"]["14"]["audio"],acc["autocorr"]["14"]["video"])]:
        md.write(f"| {li} | {mean(src_a):.4f} ± {std(src_a):.4f} | {mean(src_v):.4f} ± {std(src_v):.4f} |\n")

print(f"\nJSONL → {out_path}")
print(f"Plot  → {out_plot}")
print(f"MD    → {md_path}")
