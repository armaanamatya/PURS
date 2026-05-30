# Early Thinker Layer Relevance - Findings

**Main conclusion:** the right replacement candidate is **Layer 6**, not Layer 14.

The early Thinker signal is not question-specific pruning. It is a mostly question-invariant, video-level saliency signal. That signal becomes useful for OmniZip only at Layer 6. Layer 14 looks strong by Gini, but it does **not** match OmniZip's actual audio selection behavior.

---

## 1. Setup

### Layer-depth experiment

- Model: Qwen2.5-Omni-7B Thinker
- Data: 39 videos across WorldSense (21), Video-MME (12), Daily-Omni (6)
- Layers tested: [0, 1, 3, 6, 10, 14, 20, 27]
- Signal: for each layer, compute text-to-audio and text-to-video cross-modal relevance using Thinker self-attention projections
- Original metrics: cross-question Spearman, Gini, score spread

### Follow-up OmniZip alignment experiment

- Data: WorldSense only
- Videos: 21
- Questions: 59 total
- Metric: ROC AUC between cached saliency scores and OmniZip's actual audio keep mask
- Comparison: Layer 6 vs Layer 14

---

## 2. The signal is question-invariant

Across the 39-video layer-depth experiment, cross-question Spearman never drops below **0.917**, even at the deepest layer tested.

| Layer | Audio Spearman (mean +- std) | Video Spearman (mean +- std) |
|-------|-------------------------------|-------------------------------|
| 0     | 0.9999 +- 0.0000              | 1.0000 +- 0.0000              |
| 1     | 0.9998 +- 0.0003              | 0.9999 +- 0.0001              |
| 3     | 0.9994 +- 0.0005              | 0.9996 +- 0.0004              |
| 6     | 0.9992 +- 0.0008              | 0.9990 +- 0.0017              |
| 10    | 0.9953 +- 0.0055              | 0.9949 +- 0.0082              |
| 14    | 0.9876 +- 0.0114              | 0.9872 +- 0.0131              |
| 20    | 0.9917 +- 0.0129              | 0.9905 +- 0.0108              |
| 27    | 0.9964 +- 0.0035              | 0.9960 +- 0.0044              |

Interpretation:

- The signal is **not** "which tokens matter for this particular question".
- It is closer to a **video-level semantic content saliency** signal.
- That makes caching plausible: compute once per video, reuse across questions.

---

## 3. The Gini curve still matters, but it is not enough

The Gini curve has a clear elbow at Layer 6 and a second peak at Layer 14:

| Layer | Audio Gini | Video Gini | Note |
|-------|------------|------------|------|
| 0     | 0.0012     | 0.0018     | Flat |
| 1     | 0.0159     | 0.0254     | Still near-flat |
| 3     | 0.0532     | 0.0896     | Mild concentration |
| 6     | 0.3268     | 0.3075     | First strong elbow |
| 10    | 0.1857     | 0.1765     | Dip |
| 14    | 0.3406     | 0.3127     | Peak |
| 20    | 0.3372     | 0.2777     | Plateau |
| 27    | 0.0024     | 0.0026     | Collapse |

Important correction:

- Gini correctly says the signal becomes concentrated around Layer 6.
- But Gini alone was **not** enough to choose between Layer 6 and Layer 14.
- The AUC experiment shows that similar Gini does **not** mean similar pruning behavior.

So Gini is a good diagnostic, but not the final decision metric.

---

## 4. The decisive result: Layer 6 aligns with OmniZip, Layer 14 mostly does not

Agreement with OmniZip's actual audio keep mask:

| Layer | AUC vs OmniZip keep mask (mean +- std) | Interpretation |
|-------|-----------------------------------------|----------------|
| 6     | **0.6528 +- 0.1105**                    | Moderate agreement |
| 14    | **0.5253 +- 0.0662**                    | Near-random overall |

Additional facts:

- Layer 6 beats Layer 14 on **58/59 questions**
- Mean question-level AUC gap: **+0.1276** in favor of Layer 6
- The only Layer 14 > Layer 6 case is tiny: **0.5121 vs 0.5062**

This is the key change in the story:

- Earlier conclusion: "L6 and L14 are both plausible because their Gini is similar"
- Updated conclusion: **false**
- Actual conclusion: **Layer 6 is the only serious replacement candidate**

---

## 5. Where Layer 6 helps most

Largest Layer 6 gains over Layer 14 are on audio-heavy or audio-grounded tasks:

| Video/task family | Mean L6 - L14 AUC gain |
|-------------------|------------------------|
| Audio source localization | +0.3455 |
| Event recognition | +0.2756 |
| Human-object interaction | +0.2595 |
| Audio recognition | +0.2347 |
| Audio change | +0.2083 |
| Text and diagram understanding | +0.1942 |
| Scene recognition | +0.1749 |
| Temporal localization | +0.1697 |

This is exactly where an LLM-aware audio saliency signal should help most.

---

## 6. The cache-once-per-video story survives the stronger metric

The AUC itself is extremely stable across different questions on the same video.

For Layer 6:

- Mean within-video AUC std across questions: **0.0013**
- Mean within-video AUC range across questions: **0.0030**

That is very strong evidence that the useful part of the signal is truly a **video-level property**.

So the updated picture is:

- The saliency map is question-invariant
- The overlap with OmniZip is also question-invariant
- Caching once per video is well supported by the data

---

## 7. Mechanistic interpretation

The signal is better framed as **semantic content saliency**:

- It measures which audio/video tokens are geometrically compatible with the LLM's language space
- It is not a direct per-question relevance map
- It is still useful because OmniZip ultimately needs a ranking signal for which tokens to preserve

The important refinement is this:

- **Layer 6** captures a form of early cross-modal content geometry that still overlaps with OmniZip's current audio retention behavior
- **Layer 14** appears to represent something else: more semantic or reasoning-heavy structure that does not line up with OmniZip's pruning mask

So for compression, "deeper" is not better here.

---

## 8. What this means for the paper

The contribution should now be framed as:

1. Early Thinker layers expose a cacheable, video-level saliency signal.
2. Layer 6 provides meaningful overlap with OmniZip's current audio selection behavior.
3. Layer 14 does not, despite looking good under Gini.
4. Therefore Layer 6 is the correct candidate for a benchmark replacement experiment.

This is stronger than the original vague "maybe early layers are useful" story. We now have:

- a specific layer,
- a specific failure mode for the deeper layer,
- and a concrete metric showing why.

---

## 9. Next experiment

Run the full benchmark with OmniZip's audio importance replaced by cached Layer 6 saliency.

Baseline:

- `eval_qwen_omni_zip.py`

Replacement:

- `eval_qwen_omni_zip_cached.py --layer 6 --cache_reduce mean`

Interpretation target:

- If accuracy holds or improves, Layer 6 is a real replacement signal
- If accuracy drops, Layer 6 overlaps with OmniZip enough to be plausible, but not enough to preserve behavior
- Either way, the result is publishable because the AUC already shows Layer 6 is meaningful and Layer 14 is not

---

## 10. Scripts now available

Implemented:

- `compute_omnizip_auc.py`
  Measures how well a cached saliency score matches OmniZip's actual audio keep mask.

- `eval_qwen_omni_zip_cached.py`
  Runs the real benchmark while replacing OmniZip's audio score with cached external saliency.

Still optional:

- `precompute_l6_saliency.py`
  A cleaner neutral-query offline precompute script for full benchmark-scale cache generation.

---

## Key numbers

- Cross-question Spearman floor: **0.917**
- Gini jump L3 -> L6: **6.1x**
- Layer 6 AUC vs OmniZip keep mask: **0.6528 +- 0.1105**
- Layer 14 AUC vs OmniZip keep mask: **0.5253 +- 0.0662**
- Layer 6 beats Layer 14 on **58/59** questions
- Mean Layer 6 within-video AUC range across questions: **0.0030**

Bottom line:

**Use Layer 6 for the replacement benchmark. Do not use Layer 14.**
