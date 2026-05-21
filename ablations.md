# Ablation Space

## 1. OmniZip — Internal Parameter Sweeps

| Ablation | What varies | What it proves |
|---|---|---|
| **`rho_video` sweep** | 0.3, 0.45, 0.6, 0.75, 0.9 | Compression-accuracy Pareto for video tokens |
| **`rho_audio` sweep** | 0.1, 0.2, 0.3, 0.5, 0.7 | How aggressively audio tokens can be compressed |
| **Audio guidance OFF** | Replace audio-guided scoring with random/uniform selection at same `rho_video` | **The most critical ablation** — proves the audio cross-modal signal drives the gain, not just "keeping 60% of tokens" |
| **`contextual_ratio` sweep** | 0.0, 0.025, 0.05, 0.1, 0.2 | Does the uniform-coverage "safety net" matter, or does pure audio guidance suffice? |
| **`g` (group size) sweep** | 1, 2, 3, 5, 8 | Temporal/spatial pooling granularity vs accuracy |
| **`rho_video=1.0` sanity** | Keep all tokens | Must match baseline; verifies no scoring overhead penalty |

## 2. ReDiPrune — Internal Ablations

| Ablation | What varies | What it proves |
|---|---|---|
| **`alpha` sweep** | 0.0 → 1.0 | **Core claim**: alpha=0 collapses to DivPrune; alpha=1 is pure text-relevance. The interesting zone is in between. |
| **`subset_ratio` sweep** | 0.25, 0.33, 0.5, 0.67 | Compression-accuracy Pareto, comparable to rho_video |
| **Frame-level vs token-level** | `--prune_mode frame` vs `--prune_mode token` | Coarse vs fine pruning granularity at same keep-ratio |
| **Random text query** | Replace question text with random noise as the relevance query | Tests whether task-relevance is real or just regularization |
| **`tau` sweep** | 0.05, 0.1, 0.2, 0.5 | Sharpness of relevance scoring |

## 3. MixKV — Internal Ablations

| Ablation | What varies | What it proves |
|---|---|---|
| **`select_method`** | snapkv → vnorm → headwisemixkv | Does mixing attention + diversity + value-norm beat attention alone? Already parameterized. |
| **`budget` sweep** | 64, 128, 256, 512 | KV cache compression-accuracy tradeoff |

## 4. Cross-Method Comparisons at Matched Compression

**4a. OmniZip vs DivPrune at matched keep-ratio**
→ Audio-guided selection vs pure visual diversity. Isolates: does the audio query add value over "just pick diverse frames"?

**4b. OmniZip vs ReDiPrune at matched keep-ratio**
→ Audio as the guiding query vs text (question) as the guiding query. Which modality better identifies which visual tokens matter? This is the conceptually sharpest cross-paper ablation available.

**4c. ReDiPrune (alpha=0) vs DivPrune**
→ These should produce identical results — a free sanity check that the ReDiPrune implementation is correct.

**4d. Input token pruning vs KV-cache compression at matched VRAM**
→ OmniZip/ReDiPrune prune *before* the model; MixKV prunes *during* decode. Same VRAM budget, different compression point. Which is more accuracy-preserving?

**4e. Quantization (GPTQ/AWQ) vs token compression (OmniZip) at matched throughput**
→ Weight compression vs context compression — different axes entirely. Do they degrade differently on audio-heavy vs vision-heavy questions?

## 5. Stacking / Combining Methods

| Stack | Hypothesis |
|---|---|
| **OmniZip + MixKV** | Orthogonal axes (input tokens + KV cache). Should compound gains with limited accuracy loss. |
| **ReDiPrune + MixKV** | Frame selection → KV compression. Does two-stage outperform either alone? |
| **GPTQ + OmniZip** | Weight quant + token pruning — do they interact gracefully? |
| **GPTQ + OmniZip + MixKV** | Maximum compression — what's the floor? |

## 6. Input Modality Ablations

**6a. `--no_audio` across all methods**
→ Run every method with video-only input. Critical for OmniZip: without audio, what does the selector fall back to? Does accuracy drop more for OmniZip than for text-guided ReDiPrune?

**6b. Audio flows into model but does NOT guide visual selection (OmniZip)**
→ Audio still present as input tokens but doesn't drive the keep-mask. Tests whether OmniZip's benefit comes from (a) the audio selection signal or (b) just the general presence of the audio modality.

## 7. Stratified Analysis (No New Runs — Use Existing `results.jsonl`)

**7a. VideoMME short / medium / long**
→ Compression should help more on long videos (larger token budgets squeezed harder). Does OmniZip's gain over baseline grow with video length?

**7b. VideoMME with subtitles vs without**
→ Audio speech content overlaps with subtitle text. If subtitles are provided, does OmniZip's audio guidance add less (since the text already carries the speech info)?

**7c. Question category stratification**
→ Split by question type (temporal, counting, speech understanding, visual OCR, etc.). OmniZip should dominate on audio-heavy categories; ReDiPrune on text-query-aligned categories.

**7d. Temperature sensitivity per method (already collected)**
→ `temp=0.1` vs `temp=0.9` × 10 repeats. Do compressed methods show higher variance? Especially: does token pruning amplify stochastic generation more than weight quantization?

## Priority Ranking

### Run first (highest scientific payoff)
1. Audio guidance OFF in OmniZip (same `rho_video`, random scoring) — proves the audio signal is the contribution
2. `alpha` sweep in ReDiPrune — proves text-relevance contribution
3. OmniZip vs ReDiPrune at matched compression — audio query vs text query
4. OmniZip + MixKV stacked

### Run second (strong supporting ablations)
5. `rho_video` sweep in OmniZip (3–4 points on the Pareto curve)
6. `--no_audio` across all methods
7. Frame-level vs token-level in ReDiPrune
8. Input token pruning vs KV compression at matched VRAM

### Analyze from existing results (free)
9. Short/medium/long video stratification
10. With/without subtitles stratification
11. Temperature variance per method
