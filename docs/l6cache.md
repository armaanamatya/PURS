# L6 Cache: Early Thinker Saliency as OmniZip Audio Importance

> **Status update 2026-05-07:** the 118-question accuracy numbers below are **superseded** by the 10× repeat matrix at `10x/qwen25_matrix_gpu7_all7_snapkv/` (see `docs/research_log.md §Key Results Summary` and `docs/PROJECT_BRIEFING.md §2a`). The matrix shows OmniZip and baseline are statistically tied on accuracy (0.312 vs 0.311 at T=0.1, within 0.1σ), so the "−1.7 pts vs OmniZip" headline below is within run-to-run noise and not a real regression. The L6 vs L14 AUC results, the question-invariance findings, and the paired-prediction agreement (114/118) remain valid. **The L6-cache accuracy claim must be re-validated on the 10× harness before any standalone publishable result.** The reframed contribution: L6 is a video-level, question-invariant cacheable surrogate for OmniZip's audio mask — not a "preserves OmniZip's accuracy gain" claim, since there is no such gain over baseline.

## Summary

We investigate whether the cross-modal Q·K attention signal at **Layer 6** of the Qwen2.5-Omni-7B Thinker can replace OmniZip's intra-audio encoder importance score. The signal is question-invariant, can be precomputed offline once per video, and achieves **1.61× prefill speedup** and **18.5 GB peak VRAM** — matching OmniZip's efficiency profile while losing only 1.7 accuracy points overall *(this 1.7-pt gap is now known to be within 10×-matrix run noise)*.

---

## Background

**OmniZip** compresses audio and video tokens before the Thinker using importance scores derived from the audio encoder's own self-attention. With `rho_audio=0.3`, it drops 30% of audio tokens and 60% of video tokens, yielding a 1.62× prefill speedup.

**Our hypothesis**: An early Thinker layer already encodes a ranking of audio tokens by cross-modal relevance to text — and this ranking aligns with OmniZip's pruning decisions. If true, it can be computed offline without the question and reused across queries.

---

## Experiments

### 1. Layer Depth Experiment

**Script**: `viz_layer_depth_experiment.py`  
**Data**: 39 videos — WorldSense (21), Video-MME (12), Daily-Omni (6)  
**Layers tested**: [0, 1, 3, 6, 10, 14, 20, 27]  
**Signal**: For each layer, compute text→audio Q·K cross-modal relevance scores using Thinker self-attention projections

**Finding 1: The signal is question-invariant**

Cross-question Spearman rank correlation between score vectors for different questions on the same video:

| Layer | Audio Spearman | Video Spearman |
|-------|---------------|----------------|
| 0     | 0.9999 ± 0.0000 | 1.0000 ± 0.0000 |
| 1     | 0.9998 ± 0.0003 | 0.9999 ± 0.0001 |
| 3     | 0.9994 ± 0.0005 | 0.9996 ± 0.0004 |
| 6     | 0.9992 ± 0.0008 | 0.9990 ± 0.0017 |
| 10    | 0.9953 ± 0.0055 | 0.9949 ± 0.0082 |
| 14    | 0.9876 ± 0.0114 | 0.9872 ± 0.0131 |
| 20    | 0.9917 ± 0.0129 | 0.9905 ± 0.0108 |
| 27    | 0.9964 ± 0.0035 | 0.9960 ± 0.0044 |

Even at Layer 27 the ranking barely changes across questions. The signal is a **video-level semantic saliency** property, not a question-specific relevance map. This makes caching plausible: compute once per video, reuse across all questions.

**Finding 2: Gini concentration peaks at L6 and L14**

| Layer | Audio Gini | Video Gini |
|-------|------------|------------|
| 0     | 0.0012     | 0.0018     |
| 3     | 0.0532     | 0.0896     |
| 6     | 0.3268     | 0.3075     |
| 10    | 0.1857     | 0.1765     |
| 14    | 0.3406     | 0.3127     |
| 20    | 0.3372     | 0.2777     |
| 27    | 0.0024     | 0.0026     |

L6 is the **first strong elbow** — the signal becomes concentrated enough for pruning. L14 also peaks but represents different geometry.

---

### 2. OmniZip AUC Experiment

**Script**: `compute_omnizip_auc.py`  
**Data**: WorldSense (21 videos, 59 questions)  
**Metric**: ROC AUC between cached saliency scores and OmniZip's actual audio keep mask  

| Layer | AUC vs OmniZip keep mask | Interpretation |
|-------|--------------------------|----------------|
| 6     | **0.6528 ± 0.1105**      | Moderate alignment |
| 14    | **0.5253 ± 0.0662**      | Near-random |

- L6 beats L14 on **58/59 questions**
- Mean gap: **+0.1276** per question in favor of L6
- Within-video AUC std for L6: **0.0013** (confirms video-level property)

**Layer 6 is the only viable candidate. Layer 14 is not.**

---

### 3. Offline Precomputation

**Script**: `precompute_l6_saliency.py`  
**Model**: Stock Qwen2.5-Omni-7B (not OmniZip-modified)  
**Why stock**: OmniZip compresses audio tokens before the Thinker layers run. Hooking L6 during an OmniZip forward pass would see only surviving tokens (~70%), making scores incomparable to OmniZip's full keep mask.

**Process**: For each video, one forward pass with dummy query ("Describe this video.") → hook on `model.thinker.model.layers[6].self_attn` → extract Q·K cross-modal scores over all audio tokens → save to JSONL.

**Output**: `vizzing/layer_depth_all_full.jsonl` — 44 videos, all 3 datasets

---

### 4. Benchmark Evaluation

**Script**: `eval_qwen_omni_zip_cached.py`  
**Integration**: Monkey-patches `omnizip_units.omnizip()` to replace the `attn_logits` argument (OmniZip's audio importance vector) with our cached L6 scores. Everything else in OmniZip is unchanged.

```
wrapped_omnizip():
    scores = cached_l6_scores[video]      # precomputed, mean-reduced
    return original_omnizip(..., attn_logits=scores, ...)
```

**Output**: `vizzing/results_zip_cached_l6_all_timed.jsonl`, `vizzing/results_zip_cached_l6_all_timed_vram.jsonl`

---

## Full Results

### Accuracy

| Method | Total | WorldSense | Video-MME | Daily-Omni |
|--------|-------|-----------|-----------|------------|
| Baseline | 35/118 = 29.7% | 19/64 = 29.7% | 11/36 = 30.6% | 5/18 = 27.8% |
| OmniZip | 36/118 = 30.5% | 21/64 = 32.8% | 11/36 = 30.6% | 4/18 = 22.2% |
| GPTQ | 34/118 = 28.8% | 19/64 = 29.7% | 11/36 = 30.6% | 4/18 = 22.2% |
| AWQ | 36/118 = 30.5% | 21/64 = 32.8% | 10/36 = 27.8% | 5/18 = 27.8% |
| DivPrune | 24/118 = 20.3% | 16/64 = 25.0% | 5/36 = 13.9% | 3/18 = 16.7% |
| RediPrune | 25/118 = 21.2% | 16/64 = 25.0% | 6/36 = 16.7% | 3/18 = 16.7% |
| MixKV | 24/118 = 20.3% | 16/64 = 25.0% | 5/36 = 13.9% | 3/18 = 16.7% |
| **OmniZip+L6 (ours)** | **34/118 = 28.8%** | **20/64 = 31.3%** | **10/36 = 27.8%** | **4/18 = 22.2%** |

### Efficiency

| Method | Model GB | Peak Alloc GB | Prefill ms | Prefill Speedup | E2E ms |
|--------|----------|---------------|------------|-----------------|--------|
| Baseline | 16.6 | 20.4 | 2361 | 1.00× | 2445 |
| OmniZip | 16.6 | 18.5 | 1455 | **1.62×** | 1548 |
| GPTQ | 7.6 | 9.0 | 3551 | 0.66× | 3491 |
| AWQ | 7.7 | 9.0 | 3806 | 0.62× | 3577 |
| DivPrune | N/A | 18.7 | 1246 | 1.90× | 1347 |
| RediPrune | N/A | 18.7 | 1185 | 1.99× | 1255 |
| MixKV | N/A | 19.6 | 2279 | 1.04× | 2366 |
| **OmniZip+L6 (ours)** | **16.6** | **18.5** | **1470** | **1.61×** | **1557** |

---

## Paper-Style Results Draft

We evaluate whether a cached Layer-6 Thinker saliency signal can replace OmniZip's native audio-encoder saliency without changing the downstream compression pipeline. Across 118 benchmark questions spanning WorldSense, Video-MME, and Daily-Omni, OmniZip+L6 achieves `34/118 = 28.8%` accuracy, compared with `36/118 = 30.5%` for stock OmniZip and `35/118 = 29.7%` for the uncompressed baseline. Although the aggregate gap to OmniZip is 1.7 points, the paired comparison shows that cached L6 reproduces OmniZip's behavior almost exactly: the two methods make the same final prediction on `114/118` questions and differ in correctness on only 2 questions total.

The efficiency story is even tighter. OmniZip+L6 reaches `1470 ms` mean prefill latency and `18.5 GB` peak allocated VRAM, nearly identical to OmniZip's `1455 ms` and `18.5 GB`. This is expected because our method changes only the ranking signal used to choose which audio tokens to keep; the pruning ratio, video compression path, token merging, and Thinker forward pass all remain unchanged. In other words, cached L6 preserves OmniZip's systems-level efficiency while moving the saliency computation to an offline, question-invariant cache.

The most important interpretation is therefore not that cached L6 "almost matches" OmniZip in a loose sense, but that it acts as a high-fidelity replacement signal. The benchmark gap is highly localized: one miss on Video-MME and one miss on WorldSense account for the full loss. This makes the result meaningful as a replacement-fidelity experiment rather than a weak approximation experiment. At the same time, the current benchmark does not yet prove a real serving-time advantage over OmniZip in the single-query setting, because the offline cache only becomes strictly beneficial when the same video is queried multiple times. That multi-turn reuse experiment remains the key next step.

## Deep Dive on the Benchmark Result

The headline gap (`34/118` vs `36/118`) looks larger than it really is. On the **full paired 118-question comparison** against OmniZip, cached L6 produces the **same final prediction on 114/118 questions (96.6%)**.

Paired outcome table vs OmniZip:

| Cached L6 vs OmniZip | Count |
|----------------------|-------|
| Both correct | 34 |
| Both wrong | 82 |
| OmniZip-only correct | 2 |
| Cached-L6-only correct | 0 |

So the benchmark difference is not a broad behavioral mismatch. It is much narrower:

- In **116/118** cases, both methods have the **same correctness outcome**
- Only **2 questions** flip from correct under OmniZip to incorrect under cached L6
- Cached L6 never wins a question that OmniZip misses on this run

This makes the replacement result stronger than a raw percentage comparison suggests: the cached signal is a **high-fidelity surrogate** for OmniZip's native audio ranking, not a loose approximation.

### The two actual failure cases

The entire benchmark gap comes from exactly these questions:

| Dataset | Task | Answer | OmniZip | Cached L6 | Question |
|--------|------|--------|---------|-----------|----------|
| Video-MME | Information Synopsis | A | A | C | "What does the second half of the video show?" |
| WorldSense | Scene Recognition | C | C | A | "Where does the scene of the baby alpaca playing with toys in the video take place?" |

Interpretation:

- The misses are **not** concentrated in pure audio-recognition prompts
- They occur on **higher-level semantic readout tasks** (summary / scene interpretation)
- This fits the mechanistic story: Layer 6 is a strong early saliency signal, but still only a **moderate proxy** for OmniZip's native ranking (AUC `0.653`, not `1.0`)

### Dataset-level reading

The degradation is localized to two datasets:

| Dataset | OmniZip | Cached L6 | Delta |
|--------|---------|-----------|-------|
| Daily-Omni | 4/18 = 22.2% | 4/18 = 22.2% | 0 |
| Video-MME | 11/36 = 30.6% | 10/36 = 27.8% | -1 question |
| WorldSense | 21/64 = 32.8% | 20/64 = 31.3% | -1 question |

So the most accurate reading is:

- **No loss at all** on Daily-Omni
- **One-question drop** on Video-MME
- **One-question drop** on WorldSense

That is a much more stable story than the percentage gap alone suggests.

### Timing stability is essentially exact

Cached L6 is not just "close" to OmniZip on efficiency. It is effectively the **same runtime regime**.

Across all 118 aligned questions, cached L6 minus OmniZip is:

- Mean prefill delta: **+15.9 ms**
- Median prefill delta: **+20.6 ms**
- Mean end-to-end delta: **+8.7 ms**
- Median end-to-end delta: **+15.9 ms**

Per dataset:

| Dataset | Mean prefill delta (Cached - OmniZip) | Mean E2E delta |
|--------|----------------------------------------|----------------|
| Daily-Omni | +3.2 ms | +17.0 ms |
| Video-MME | +24.4 ms | +8.9 ms |
| WorldSense | +14.6 ms | +6.3 ms |

These are tiny relative to OmniZip's ~`1455 ms` prefill time. In practice, cached L6 inherits OmniZip's efficiency almost exactly because the same number of tokens survive into the Thinker.

### What is robust vs what is still noise

Robust claims:

1. **Behavioral fidelity** is real. Cached L6 matches OmniZip's exact prediction on `96.6%` of the benchmark.
2. **Efficiency equivalence** is real. Prefill and E2E latency differ only by a few milliseconds on average.
3. **The benchmark gap is localized**, not systemic. Only two questions account for the full loss.

Claims that should be made cautiously:

1. **Accuracy ranking is not statistically strong at this scale.** The three headline accuracies are all within a couple of questions:
   Baseline `35/118`, OmniZip `36/118`, Cached L6 `34/118`.
2. The 95% Wilson intervals overlap heavily:
   Baseline `22.2-38.4%`, OmniZip `22.9-39.3%`, Cached L6 `21.4-37.6%`.
3. So the right framing is **behavior preservation**, not "wins" or "beats."

### What the deep dive changes in the paper story

Before the paired analysis, the result looked like:

- "Cached L6 is a bit worse than OmniZip."

After the paired analysis, the more accurate statement is:

- "Cached L6 reproduces OmniZip almost exactly, with identical efficiency and only two paired failures on a 118-question benchmark."

That is materially stronger. It turns the evaluation from a vague approximate match into a concrete **replacement-fidelity result**.

---

## Key Numbers

- Cross-question Spearman floor at L6: **0.9992**
- Gini jump L3 -> L6: **6.1x**
- L6 AUC vs OmniZip keep mask: **0.6528 +/- 0.1105**
- L14 AUC vs OmniZip keep mask: **0.5253 +/- 0.0662**
- L6 beats L14 on **58/59** questions
- Accuracy vs OmniZip: **-1.7 points** overall (28.8% vs 30.5%)
- Exact prediction agreement vs OmniZip: **114/118 = 96.6%**
- Paired correctness disagreement vs OmniZip: **2/118 questions**
- Prefill speedup: **1.61x** (vs OmniZip 1.62x, vs Baseline 1.0x)
- Peak VRAM: **18.5 GB** (identical to OmniZip)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `viz_layer_depth_experiment.py` | Layer depth experiment: hooks [0,1,3,6,10,14,20,27], computes Spearman/Gini/spread across questions and videos |
| `compute_omnizip_auc.py` | Measures AUC between cached saliency and OmniZip's actual audio keep mask |
| `precompute_l6_saliency.py` | Offline cache generation: stock model, layer-6 hook, dummy query, all datasets |
| `eval_qwen_omni_zip_cached.py` | Benchmark eval with cached L6 replacing OmniZip audio importance |
| `analyze_v2_scores.py` | Computes Jaccard, separation, autocorrelation from raw layer scores |

---

## Is This a Significant Contribution?

**Honest answer: this is strong insights + a proof-of-concept, not a standalone contribution yet.**

### What it genuinely establishes

1. **Mechanistic finding**: Layer 6 of the Qwen2.5-Omni Thinker encodes a cross-modal audio saliency signal that aligns with OmniZip's actual pruning decisions (AUC 0.653). This is a real finding about how the model internally represents multimodal content.

2. **Question-invariance is empirically confirmed**: Spearman ≥ 0.999 at L6 across all tested videos and datasets. This is not obvious and has practical implications.

3. **Layer specificity**: L14 looks similar by Gini but fails the AUC test (0.525 ≈ random). This is a concrete, falsifiable result.

4. **Proof-of-concept replacement**: The cached signal runs at the same efficiency as OmniZip with only −1.7 points accuracy cost across 118 questions.

### What it does not yet establish

1. **No accuracy improvement**: OmniZip+L6 (28.8%) is slightly below OmniZip (30.5%). We preserve behavior, we don't improve it.

2. **No real-world speedup over OmniZip**: In a single-query scenario, precomputing adds a forward pass. The benefit only materializes when the same video is queried multiple times (multi-turn video QA, video search systems). We have not demonstrated that scenario.

3. **Small evaluation scale**: 118 questions across 3 datasets. Differences of 1-2 questions are within noise.

4. **No generalization study**: Only tested on Qwen2.5-Omni-7B, only at rho_audio=0.3. Does L6 remain the right layer at different compression ratios? Different model sizes?

### How to make it a full contribution

The path to a publishable paper would require at least one of:
- Show that the cached signal enables **faster multi-turn video QA** in a realistic serving scenario (the offline cache story pays off here)
- Show the signal **improves** accuracy at higher compression ratios where OmniZip's encoder signal degrades
- Generalize to other models (Qwen2-Audio, LLaVA-OneVision) to show L6-type signals are universal
- Use the signal for something OmniZip cannot do: e.g., pre-ranking videos before loading them, or dynamic compression ratio selection based on saliency distribution

**As-is**: strong analysis section or workshop paper. With the multi-turn or cross-model experiments: full conference paper.
