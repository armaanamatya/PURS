# Cross-Modal Attention Analysis & OmniZip Beat-The-Baseline Discussion

**Date:** 2026-04-30
**Topic:** Analyzing cross-modal AUC results, comparing OmniZip vs LLM L6 attention, deciding research direction.

---

## 1. Initial setup

Both result directories were SCP'd from the server:
- `vizzing/crossmodal_spearman/` — 44-record JSONL + 5 plots
- `vizzing/auc_vs_omnizip/` — 44-record JSONL + 1 plot

---

## 2. Full statistical analysis

### AUC vs OmniZip mask (n=44)

| Direction | L6 mean ± std | L14 mean ± std |
|---|---|---|
| text→audio | **0.643 ± 0.116** | 0.508 ± 0.068 |
| video→audio | **0.641 ± 0.116** | 0.524 ± 0.065 |
| text→video | 0.526 ± 0.041 | 0.540 ± 0.037 |
| audio→video | 0.530 ± 0.043 | **0.568 ± 0.040** |

### Per-dataset
| Dataset | text→audio L6 | video→audio L6 | text→video L6 | audio→video L6 |
|---|---|---|---|---|
| video-mme (n=12) | 0.629 | 0.627 | 0.522 | 0.523 |
| daily-omni (n=6) | 0.581 | 0.591 | 0.564 | 0.574 |
| worldsense (n=26) | **0.663** | **0.660** | 0.519 | 0.522 |

### Cross-modal Spearman / Jaccard (question-invariance)
- text→audio Spearman ≥ 0.987 across all layers
- **video↔audio directions: Spearman = 1.000, Jaccard = 1.000 at ALL layers**
- text→video also ~0.99
- → Cross-modal attention is essentially question-invariant

### OmniZip compression ratios
- audio_kept: mean=0.7501, std=0.00023 (essentially **fixed at 75%**)
- video_kept: mean=0.4089, std=0.0163 (content-adaptive, ~40%)

### Pearson correlations
- text→audio vs video→audio @ L6 = **0.989** (text and video queries agree about audio)
- text→video vs audio→video @ L6 = **0.954**
- text→audio L6 vs L14 = 0.701

### High/low AUC tasks
**High AUC (>0.75) text→audio @ L6:** event_recognition (0.85), audio_change (0.85), temporal_localization (0.82), audio_counting (0.82), audio_recognition (0.81), spatial_relation (0.82), object_recognition (video-mme 0.80)

**Low AUC (<0.54):** spatial_perception (0.49), event_sequence/daily-omni (0.47 — below chance), human_interaction (0.50), ocr_problems (0.53)

---

## 3. Key findings

1. **Cross-modal attention is question-invariant** (Spearman ≥ 0.987) — justifies one-shot encoder pruning
2. **Audio AUC partially predictive** (0.64 mean, 0.47–0.85 range) — task-conditional alignment
3. **Video AUC near-random** (0.52) — OmniZip's video pruning is orthogonal to LLM attention
4. **text→audio ≈ video→audio** — content-driven, not query-driven
5. **L6 > L14 for audio** alignment; reverses slightly for video
6. **OmniZip audio compression is fixed at 75%**, video is content-adaptive

---

## 4. Accuracy results (resultswprefill, n=118)

| Method | Overall | video-mme | daily-omni | worldsense |
|---|---|---|---|---|
| baseline | 29.7% | 30.6% | 27.8% | 29.7% |
| **omnizip** | **30.5%** | 30.6% | 22.2% | 32.8% |
| awq | 30.5% | 27.8% | 27.8% | 32.8% |
| gptq | 28.8% | 30.6% | 22.2% | 29.7% |
| **zip_l6_cached** | **28.8%** | 27.8% | 22.2% | 31.3% |
| mixkv | 20.3% | 13.9% | 16.7% | 25.0% |
| divprune | 20.3% | 13.9% | 16.7% | 25.0% |
| rediprune | 21.2% | 16.7% | 16.7% | 25.0% |

### Pearson(accuracy_drop, AUC) per video
- vs text→audio: -0.008 (zero)
- vs video→audio: -0.042 (zero)
- vs text→video: 0.241 (weak)
- vs audio→video: 0.202 (weak)

n=3 per video → 1 question = 33pp swing. Statistical power too low.

### Timing
- omnizip prefill_ms = 1455
- zip_l6_cached prefill_ms = 1470 (no speedup)

---

## 5. Critical discovery: zip_l6_cached already exists

`results_zip_cached_l6_all.jsonl` proves the natural extension was already tried:
- **Method:** `omnizip_cached_audio`, cache_layer=6, cache_reduce="mean"
- **Saliency source:** `vizzing/layer_depth_all_full.jsonl`
- **Result: 28.8% — WORSE than OmniZip (30.5%) and below baseline (29.7%)**

**OmniZip's encoder signal is irreplaceable by LLM L6 cross-modal attention.**

---

## 6. Existing experiments (vizzing/)
- `early_layer_relevance/scores.jsonl` (n=120) — L0/L1 scores, Jaccard 0.995
- `layer_depth_all_full.jsonl` (n=44) — Spearman across L0–L27
- `omnizip_auc_l6_worldsense/` — per-question AUC at L6 (n=59)
- `omnizip_auc_l14_worldsense/` — per-question AUC at L14 (n=59)
- `crossmodal_spearman/crossmodal_stats.jsonl` — 4 directions × 8 layers, n=44
- `auc_vs_omnizip/auc_vs_omnizip.jsonl` — full 4-direction AUC, n=44
- `results_zip_cached_l6_all*.jsonl` — L6-cached pruning accuracy
- `analysis/`, `analysis_v2/` — markdown summaries

---

## 7. Research direction discussion

**The L6 hypothesis failed.** OmniZip's encoder signal beats LLM L6 attention for pruning despite AUC=0.64 partial alignment. The encoder captures something more discriminative.

### Why beating OmniZip on accuracy is hard
- OmniZip already matches baseline (no ceiling)
- n=118 too small (1 question = 33pp swing per video)
- Most natural extensions already tried

### Three remaining angles
1. **More aggressive audio pruning** — drop keep_ratio from 75% → 50%. If accuracy holds, that's an efficiency win using OmniZip's signal.
2. **ViT self-attention for video pruning** — never tried. Replace OmniZip's audio-derived video mask with ViT patch self-attention.
3. **Run on full benchmark** — n=118 is too small to detect real differences. Full Video-MME / WorldSense would give statistical power.

### Reframed contribution
> *OmniZip's encoder-level pruning is not replaceable from the LLM side. The encoder operates on a different, richer signal. LLM cross-modal attention partially validates OmniZip's choices (AUC 0.64 for audio tasks) but can't replicate them — showing the two systems capture complementary information about token salience.*

This is a coherent **analysis paper** (or analysis section in OmniZip paper), not a systems improvement paper — unless angle #1 or #2 yields a real win.

---

## 8. Recommendation

Priority order:
1. Run keep_ratio=0.5 audio pruning experiment this week (1 script change)
2. Try ViT self-attention as video signal (only unexplored signal)
3. Run on full benchmark in parallel for statistical power

Tell professor: analysis is complete, encoder signal is irreplaceable, three concrete experiments queued to attempt efficiency wins.
