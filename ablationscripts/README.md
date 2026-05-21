# Ablation Studies: Token Compression for Qwen2.5-Omni-7B Video QA

See also: [../ablations.md](../ablations.md) for the high-level ablation plan.

---

## 1. Overview

These ablations test training-free token compression methods for **Qwen2.5-Omni-7B** on video question-answering benchmarks (VideoMME, WorldSense, Daily-Omni). The model accepts video frames, audio, and text as joint input and must select a multiple-choice answer.

**Core research question:** Which compression signals matter most — audio salience, visual diversity, text relevance, or attention patterns? How do these methods compare at matched keep ratios, and can they be stacked?

**Methods under study:**
| Method | Signal used | Compresses |
|---|---|---|
| OmniZip | Audio attention → video budget | Visual tokens, audio tokens |
| DivPrune | Feature diversity (FPS) | Visual tokens |
| ReDiPrune | Diversity + text relevance | Visual tokens |
| MixKV | Attention + value norm + key similarity | KV cache during generation |

**Standard eval config** shared across all scripts:
- `fps=2.0`, `max_pixels=100352`
- `max_frames_videomme=768`, `max_frames_other=128`
- `max_new_tokens=256`, `temperature=0.1`
- Model: `/data/armaan/models/Qwen2.5-Omni-7B`

---

## 2. Method Math

### 2.1 OmniZip

OmniZip compresses both audio tokens and visual tokens, using the audio attention pattern as the signal that drives video compression.

#### Audio Compression

Given N audio tokens $a_1, \ldots, a_N$ extracted from the audio encoder, with attention logits collected from the cross-attention layers:

```
importance_i = sum_over_dim( mean_over_heads( attn_logits ) )    shape: [N]

dominant_num   = round( (1 - rho_audio) × N )
S_dominant     = top-{dominant_num}( importance )                 # attention-selected

contextual_num = round( contextual_ratio × N )
S_contextual   = uniform_sample( contextual_num,
                     from: all_indices \ S_dominant )             # safety-net anchors

S_keep = S_dominant ∪ S_contextual
```

Tokens outside `S_keep` are merged into anchors in `S_contextual`:

```
For each anchor a_c ∈ S_contextual:
    merge_candidates = top-g nearest tokens by cos_sim( a_i, v_k )
                       for v_k ∈ video tokens           # g = merge group size

    w_j = softmax( max_k cos_sim( a_j, v_k ) )         # video-similarity weights

    new_anchor = ( a_c + Σ_j w_j · a_j ) / ( 1 + Σ_j w_j )
```

**Parameters:**
- `rho_audio`: compression aggressiveness. Higher → more audio pruned → weaker guidance signal for video.
- `contextual_ratio`: fraction of audio kept as uniform anchors regardless of importance.
- `g`: merge group size. Larger → each anchor absorbs more tokens.

#### Audio-Guided Video Compression

OmniZip divides the video timeline into temporal groups and allocates more budget to moments that have high audio retention (important audio → important video):

```
For temporal group i:
    audio_group_retention_i = |S_keep ∩ group_i| / |group_i|

    base_video_ratio_i = 0.75 - 0.40 × audio_group_retention_i

    # Intuition:
    #   audio_retention = 1.0  →  base_ratio = 0.35  →  keep 65% of video  (important moment)
    #   audio_retention = 0.0  →  base_ratio = 0.75  →  keep 25% of video  (unimportant moment)

    # Clamp to [0.35, 0.75], then normalize so:
    Σ_i base_ratio_i = rho_video × n_groups             # budget constraint satisfied
```

**Visual selection strategy per group (omnizip_istm):**
- Even-indexed frames → **dpcknn**: density-peak k-NN diversity selection
- Odd-indexed frames → keep tokens **least similar** to the previous frame (novelty/motion)

**Parameters:**
- `rho_video`: global visual compression target. Higher → more video pruned overall.

---

### 2.2 DivPrune

Pure diversity-based visual token selection via Greedy Farthest-Point Sampling (FPS) on cosine distance:

```
Distance: d(v_i, v_j) = 1 - cos_sim(v_i, v_j)

Initialization:
    sel[0] = argmax_i ( min_{j≠i} d(v_i, v_j) )        # most isolated token

For t = 1 .. K-1:
    min_dist_i = min_{j ∈ sel[:t]} d(v_i, v_j)          # distance to nearest selected
    sel[t]     = argmax_i ( min_dist_i )                  # farthest unselected token
```

Result: K tokens that maximally cover the feature space (maximize the minimum pairwise distance among selected tokens). No text or audio signal is used.

---

### 2.3 ReDiPrune

Extends DivPrune with a text-conditioned relevance term:

```
rel_i = cos_sim( v_i, q_text )
        where q_text = mean_pool( embed_tokens( question ) )

Pre-filter (relevance gate):
    candidates = { i : rel_i ≥ tau }

Greedy selection:
    sel[0] = argmax_{i ∈ cand}( rel_i )                  # most relevant token first

    for t = 1 .. K-1:
        min_dist_i = min_{j ∈ sel[:t]} (1 - cos_sim(v_i, v_j))   # diversity term
        score_i    = min_dist_i + α × rel_i                        # combined score
        sel[t]     = argmax_{i ∈ cand \ sel}( score_i )
```

**Limiting cases:**
- `α = 0`: reduces to DivPrune (pure diversity, no text signal)
- `α = 1`: diversity and relevance have equal weight
- `α → ∞`: pure text relevance (greedy relevance selection)
- `tau = 0`: no pre-filter; all tokens are candidates

**Parameters:**
- `alpha`: trade-off between diversity and text relevance
- `tau`: relevance gate threshold; filters out irrelevant tokens before selection
- `mode`: controls whether selection is `diversity`, `relevance`, or `mixed`

---

### 2.4 MixKV

KV cache compression applied during the prefill phase. Works per attention layer per head.

```
Window queries: Q_w = Q[:, :, -W:, :]          where W = window_size

Attention scores:
    A = softmax( Q_w @ K^T / √d )               shape: [B, H, W, kv_len]

SnapKV attention score (pooled):
    s_attn = mean_pool_5( mean_over_W( A[:, :, :, :-W] ) )
             shape: [B, H, kv_len - W]

Value norm score:
    s_vnorm = normalize( ||V[i]||_2 )

Key similarity (diversity proxy):
    s_sim = normalize( -cos_sim( K_i, mean(K) ) )    # tokens far from mean = diverse

Scoring strategies:
    snapkv:
        score = s_attn

    vnorm:
        scale = mean(s_attn) / mean(s_vnorm)          # match magnitudes
        score = s_attn + s_vnorm × scale

    headwisemixkv:
        score_h = hs_h × s_sim + (1 - hs_h) × (s_attn + s_vnorm)
        where hs_h = pre-calibrated per-head similarity weight ∈ [0, 1]

Selection:
    capacity = budget - window_size
    selected = top-{capacity} tokens per head
    final KV = [ selected_{capacity} ; window_{W} ]    length = budget
```

**Parameters:**
- `budget`: total KV tokens to keep per head
- `method`: `snapkv`, `vnorm`, or `headwisemixkv`
- `window_size`: number of recent tokens always retained (W)

---

## 3. Ablation Scripts

### 3.1 `omnizip_rho_video.py`
**Output dir:** `$OUTPUT_ROOT/omnizip_rho_video/`

**What is varied:** `rho_video` — the global visual token compression ratio.

**Sweep values:** typically `[0.2, 0.35, 0.45, 0.55, 0.65, 0.75]`

**Hypothesis:** Accuracy degrades monotonically as `rho_video` increases, but OmniZip's audio guidance allows it to retain more accuracy at high compression than a uniform or random baseline, because it preferentially protects visually important moments.

**How to run:**
```bash
python omnizip_rho_video.py \
    --model /data/armaan/models/Qwen2.5-Omni-7B \
    --metadata /data/armaan/purs/videos/metadata.json \
    --videos /data/armaan/purs/videos \
    --output_root /data/armaan/purs/ablation_outputs
```

**Output files:**
- `results.jsonl` — one record per question, all fields documented in Section 4
- `sweep_summary.json` — aggregated accuracy and latency per `rho_video` value

**How to interpret:** Plot accuracy vs. `rho_video`. The slope reveals the compression sensitivity. Compare with `omnizip_audio_off` at same `rho_video` values to isolate audio guidance benefit.

---

### 3.2 `omnizip_rho_audio.py`
**Output dir:** `$OUTPUT_ROOT/omnizip_rho_audio/`

**What is varied:** `rho_audio` — audio compression aggressiveness.

**Sweep values:** typically `[0.0, 0.2, 0.4, 0.6, 0.8]`

**Hypothesis:** As `rho_audio` increases, the audio guidance signal degrades (fewer audio tokens retained = noisier importance estimates), which in turn degrades video compression quality. The curve should plateau at low `rho_audio` (audio is overparameterized) and drop at high values.

**How to run:** Same args as above.

**How to interpret:** Compare accuracy at each `rho_audio` value with `rho_audio=0` (no audio compression). The gap reveals how sensitive the audio guidance is to its own compression.

---

### 3.3 `omnizip_audio_off.py`
**Output dir:** `$OUTPUT_ROOT/omnizip_audio_off/`

**What is varied:** Audio guidance ON vs. OFF. When OFF, video compression falls back to uniform/random allocation without audio signal.

**Hypothesis:** This is the most important ablation. If OmniZip's benefit comes from audio guidance, accuracy with audio OFF should match or be worse than DivPrune (pure diversity), and audio ON should significantly outperform both.

**How to run:** Same args as above.

**How to interpret:** Binary comparison. A statistically significant accuracy gap (audio ON vs. OFF) at matched `rho_video` is the core paper claim. Check per-category breakdown — expect the gap to be largest on audio-informative question types (e.g., dialog understanding, event timing).

---

### 3.4 `omnizip_contextual.py`
**Output dir:** `$OUTPUT_ROOT/omnizip_contextual/`

**What is varied:** `contextual_ratio` — fraction of audio tokens kept as uniform anchors.

**Sweep values:** typically `[0.0, 0.05, 0.10, 0.20, 0.30]`

**Hypothesis:** Very low `contextual_ratio` is risky (anchors may cluster in one region, leaving gaps in coverage). An optimal value exists that balances selectivity (low) with coverage (higher). The relationship should be non-monotonic with a sweet spot.

**How to interpret:** Plot accuracy vs. `contextual_ratio`. Identify the optimal value and whether the default setting is near optimal.

---

### 3.5 `omnizip_g_sweep.py`
**Output dir:** `$OUTPUT_ROOT/omnizip_g_sweep/`

**What is varied:** `g` — merge group size (how many tokens each anchor absorbs).

**Sweep values:** typically `[1, 2, 4, 8, 16]`

**Hypothesis:** `g=1` means no merging (just dropping tokens). Moderate `g` allows information from dropped tokens to be preserved via weighted averaging. Very large `g` may dilute anchor quality with dissimilar tokens.

**How to interpret:** The `g` that minimizes accuracy loss at fixed compression ratio is the optimal merge factor. Should show a curve with a plateau or optimum.

---

### 3.6 `omnizip_sanity.py`
**Output dir:** `$OUTPUT_ROOT/omnizip_sanity/`

**What is varied:** Sanity check run — baseline (no compression) vs. OmniZip at moderate settings.

**Hypothesis:** Establishes the accuracy ceiling (uncompressed baseline) and verifies that the evaluation harness is functioning correctly. OmniZip result should be close to but slightly below baseline.

**How to interpret:** If OmniZip significantly underperforms baseline even at low compression, there is a bug. If OmniZip matches baseline, the compression is lossless at this setting.

---

### 3.7 `rediprune_alpha.py`
**Output dir:** `$OUTPUT_ROOT/rediprune_alpha/`

**What is varied:** `alpha` — weight on text relevance in the combined score `min_dist + α × rel`.

**Sweep values:** typically `[0.0, 0.25, 0.5, 1.0, 2.0, 5.0]`

**Hypothesis:** `alpha=0` is DivPrune. Positive `alpha` should improve accuracy on text-query-aligned question types. The optimal `alpha` reveals how much the model benefits from text conditioning. Very large `alpha` may hurt by ignoring diversity entirely.

**How to run:** Same standard args.

**How to interpret:** This is the central ablation for ReDiPrune. Compare accuracy at each `alpha` vs. `alpha=0` (DivPrune baseline). Expect improvement on factual/descriptive questions, possible neutral or slight drop on questions requiring broad scene coverage.

---

### 3.8 `rediprune_ratio.py`
**Output dir:** `$OUTPUT_ROOT/rediprune_ratio/`

**What is varied:** The keep ratio (fraction of visual tokens retained).

**Sweep values:** matched to OmniZip's `rho_video` values to enable direct comparison.

**Hypothesis:** At low keep ratios, ReDiPrune (text-aware) outperforms DivPrune (diversity only) on query-relevant tasks, but may underperform OmniZip (audio-guided) on temporal tasks.

**How to interpret:** Cross-compare the accuracy-vs-ratio curves from `rediprune_ratio.py` and `omnizip_rho_video.py` at matched keep ratios. Pareto analysis is appropriate here (Section 6).

---

### 3.9 `rediprune_mode.py`
**Output dir:** `$OUTPUT_ROOT/rediprune_mode/`

**What is varied:** The selection mode — `diversity`, `relevance`, or `mixed`.

**Hypothesis:** `mixed` mode should outperform pure diversity or pure relevance, confirming that the combination is the key design choice.

**How to interpret:** Three-way comparison. If `mixed` does not beat both extremes, the combination may not be effective, or the default `alpha` needs tuning.

---

### 3.10 `rediprune_random_query.py`
**Output dir:** `$OUTPUT_ROOT/rediprune_random_query/`

**What is varied:** Whether the text query used for relevance is the actual question vs. a random/null query.

**Hypothesis:** Using a random query should degrade ReDiPrune to near-DivPrune performance, proving that the text relevance signal is meaningful (not just any text embedding provides benefit).

**How to interpret:** Binary comparison. A significant gap confirms that query-conditioned relevance is the active ingredient in ReDiPrune.

---

### 3.11 `rediprune_tau.py`
**Output dir:** `$OUTPUT_ROOT/rediprune_tau/`

**What is varied:** `tau` — the relevance gate threshold for pre-filtering candidates.

**Sweep values:** typically `[0.0, 0.1, 0.2, 0.3, 0.5]`

**Hypothesis:** `tau=0` (no pre-filter) allows all tokens as candidates. As `tau` increases, only highly relevant tokens survive the gate, which may improve precision on focused questions but hurt recall on broad visual questions. Optimal `tau` depends on question type distribution.

**How to interpret:** Non-monotonic behavior expected. Some benchmarks may prefer low `tau` (broad visual questions) while others benefit from higher `tau` (specific object queries).

---

### 3.12 `mixkv_method.py`
**Output dir:** `$OUTPUT_ROOT/mixkv_method/`

**What is varied:** The scoring method — `snapkv`, `vnorm`, or `headwisemixkv`.

**Sweep:** All three methods at the same budget.

**Hypothesis:** `headwisemixkv` should outperform `snapkv` and `vnorm` by combining signals adaptively per head. `vnorm` should outperform `snapkv` by adding value-norm information.

**How to interpret:** Three-way accuracy comparison at matched budget. The winning method validates which KV scoring signals are informative for omnimodal video QA.

---

### 3.13 `mixkv_budget.py`
**Output dir:** `$OUTPUT_ROOT/mixkv_budget/`

**What is varied:** `budget` — number of KV tokens retained per head.

**Sweep values:** typically `[128, 256, 512, 1024, 2048]`

**Hypothesis:** Accuracy improves with budget and plateaus. The plateau point reveals the effective KV cache redundancy in Qwen2.5-Omni for video QA.

**How to interpret:** Plot accuracy vs. budget and latency/VRAM vs. budget. The knee of the accuracy curve identifies the operating point that balances quality and efficiency.

---

### 3.14 `stack_omnizip_mixkv.py`
**Output dir:** `$OUTPUT_ROOT/stack_omnizip_mixkv/`

**What is varied:** Whether OmniZip (input token compression) and MixKV (KV cache compression) are applied alone, combined, or not at all.

**Conditions:** `none`, `omnizip_only`, `mixkv_only`, `omnizip+mixkv`

**Hypothesis:** OmniZip and MixKV operate at different stages (input vs. generation), so their benefits should be largely additive. The stacked system should achieve the largest memory reduction and speedup with acceptable accuracy loss.

**How to interpret:** Four-way comparison of accuracy, latency, and VRAM. If the stacked system is significantly worse than either alone, there is an interaction (e.g., MixKV's attention scores are disrupted by the compressed input token set).

---

### 3.15 `stack_rediprune_mixkv.py`
**Output dir:** `$OUTPUT_ROOT/stack_rediprune_mixkv/`

**What is varied:** ReDiPrune (input) + MixKV (KV cache) stacking.

**Hypothesis:** Similar to `stack_omnizip_mixkv` but using diversity+relevance pruning as the input stage. Expect the combined system to provide a practical Pareto-optimal operating point.

---

### 3.16 `stack_gptq_omnizip.py`
**Output dir:** `$OUTPUT_ROOT/stack_gptq_omnizip/`

**What is varied:** GPTQ quantization (4-bit weights) combined with OmniZip token compression.

**Hypothesis:** Quantization (weight compression) and token compression are orthogonal axes. Their combination should provide multiplicative reductions in memory: quantization reduces parameter size, OmniZip reduces sequence length / KV cache size.

**How to interpret:** Compare `baseline`, `gptq_only`, `omnizip_only`, and `gptq+omnizip` on accuracy, VRAM, and latency. Quantify whether the accuracy drop compounds or is subadditive.

---

### 3.17 `noaudio_all_methods.py`
**Output dir:** `$OUTPUT_ROOT/noaudio_all_methods/`

**What is varied:** All methods evaluated with audio INPUT completely removed vs. audio present.

**Hypothesis:** Removing audio input entirely should hurt all methods that rely on audio information. OmniZip's audio guidance becomes meaningless without audio, so it should degrade most. DivPrune and ReDiPrune are audio-input-agnostic and may be less affected.

**How to interpret:** Per-method accuracy drop when audio is removed reveals how much each method depends on audio content vs. visual content. Important for understanding applicability to video-only settings.

---

## 4. Output Format

### `results.jsonl`

Each line is a JSON record with the following fields:

| Field | Type | Description |
|---|---|---|
| `dataset` | str | Benchmark name: `videomme`, `worldsense`, `daily_omni` |
| `task_type` | str | Question category (e.g., `temporal`, `spatial`, `dialog`) |
| `video` | str | Video filename or ID |
| `question` | str | The question text |
| `answer` | str | Ground-truth answer (e.g., `"A"`) |
| `prediction` | str | Model's predicted answer |
| `correct` | bool | `prediction == answer` |
| `orig_nframes` | int | Total frames in the video at the sampled fps |
| `used_nframes` | int | Frames actually passed to the model after compression |
| `prefill_ms` | float | Time (ms) to process the input (prefill phase) |
| `e2e_ms` | float | Total end-to-end time (ms) including generation |
| `vram_alloc_delta_gb` | float | VRAM allocated for this sample (GB delta from before) |
| `method` | str | Compression method name (e.g., `omnizip`, `rediprune`) |
| `config` | dict | Full config dict used for this run |
| `ablation_param` | str | The parameter being swept (e.g., `rho_video`) |
| `ablation_value` | any | The value of that parameter for this record |

### `sweep_summary.json`

Aggregated results across all sweep values:

```json
{
  "ablation_param": "rho_video",
  "results": [
    {
      "value": 0.35,
      "accuracy_overall": 0.712,
      "accuracy_per_task": {
        "temporal": 0.68,
        "spatial": 0.74,
        "dialog": 0.71
      },
      "mean_prefill_ms": 1230.4,
      "mean_e2e_ms": 2150.0,
      "mean_vram_alloc_delta_gb": 4.2,
      "mean_used_nframes": 482,
      "n_samples": 500
    }
  ]
}
```

**Fields in each entry:**
- `value`: the sweep parameter value
- `accuracy_overall`: fraction of questions answered correctly
- `accuracy_per_task`: per-category breakdown
- `mean_prefill_ms`, `mean_e2e_ms`: latency statistics
- `mean_vram_alloc_delta_gb`: average VRAM usage
- `mean_used_nframes`: effective visual coverage
- `n_samples`: number of questions evaluated

---

## 5. Priority / Recommended Run Order

### HIGH PRIORITY — Run First
These ablations directly validate the core paper claims:

1. **`omnizip_audio_off.py`** — Most critical ablation. Proves the audio guidance signal is the unique contribution of OmniZip. If accuracy with audio ON significantly exceeds audio OFF at matched compression, the claim holds.

2. **`rediprune_alpha.py`** — Proves that text relevance adds value beyond diversity alone. The curve from `alpha=0` (DivPrune) to optimal `alpha` is the core ReDiPrune contribution.

3. **`noaudio_all_methods.py`** — Establishes whether any method benefits from having audio input at all. This separates "audio guidance during compression" (OmniZip) from "audio content in the input" (all methods).

### MEDIUM PRIORITY — Run Second
These build the accuracy-efficiency Pareto curves needed for the paper figures:

4. **`omnizip_rho_video.py`** — Generates the primary OmniZip efficiency curve.

5. **`rediprune_ratio.py`** — Generates the ReDiPrune efficiency curve for cross-method comparison.

6. **`mixkv_budget.py`** — Generates the MixKV efficiency curve.

7. **`stack_omnizip_mixkv.py`** — Tests whether stacking is effective (key for system-level claim).

### LOW PRIORITY — Run Third
Hyperparameter sweeps and secondary ablations:

8. `omnizip_rho_audio.py` — Audio compression sensitivity
9. `omnizip_contextual.py` — Contextual ratio sensitivity
10. `omnizip_g_sweep.py` — Merge group size sensitivity
11. `rediprune_mode.py` — Mode comparison
12. `rediprune_random_query.py` — Null query ablation
13. `rediprune_tau.py` — Gate threshold sensitivity
14. `mixkv_method.py` — Scoring method comparison
15. `stack_rediprune_mixkv.py` — Alternative stacking
16. `stack_gptq_omnizip.py` — Quantization stacking
17. `omnizip_sanity.py` — Sanity check (run early if debugging)

---

## 6. Analysis Hints

### Computing Pareto Curves

From `sweep_summary.json` files across methods:

```python
import json, matplotlib.pyplot as plt

summaries = {
    "omnizip": json.load(open("omnizip_rho_video/sweep_summary.json")),
    "rediprune": json.load(open("rediprune_ratio/sweep_summary.json")),
    "mixkv": json.load(open("mixkv_budget/sweep_summary.json")),
}

for method, s in summaries.items():
    accs = [r["accuracy_overall"] for r in s["results"]]
    latencies = [r["mean_e2e_ms"] for r in s["results"]]
    plt.plot(latencies, accs, marker="o", label=method)

plt.xlabel("Mean e2e latency (ms)")
plt.ylabel("Accuracy")
plt.legend()
```

For VRAM-accuracy Pareto, substitute `mean_vram_alloc_delta_gb` for `mean_e2e_ms`.

A method dominates if it achieves strictly higher accuracy at strictly lower latency than another. Use `scipy.spatial.ConvexHull` on the (latency, accuracy) point cloud per method to trace the Pareto frontier.

### Question Categories: Expected Method Advantages

**OmniZip expected to excel on:**
- `dialog` — audio directly encodes speech; audio salience correctly identifies the dialogue frames
- `temporal_sequence` — audio rhythm and silence patterns align with event timing
- `audio_description` — direct audio-visual correspondence

**ReDiPrune expected to excel on:**
- `object_identification` — text query ("what is X") aligns directly with visual token relevance
- `attribute_comparison` — text-conditioned tokens capture the queried attribute
- `spatial_relation` — question specifies the spatial context needed

**DivPrune expected to excel on (vs. nothing):**
- Questions requiring broad scene coverage where no single signal dominates
- Long videos with uniform content distribution

**MixKV benefits all methods uniformly** since it operates on the KV cache post-input-compression.

### Statistical Significance

With `temperature=0.1` the model is near-deterministic, so variance across runs is low. For significance testing:
- Run 3–5 repeats on a stratified sample of questions
- Report mean ± std across repeats in `sweep_summary.json`
- Use McNemar's test on paired correct/incorrect outcomes to test if method A is significantly better than method B at the same compression level
- Minimum detectable effect at N=500 questions (two-tailed, α=0.05): approximately 3–4% absolute accuracy difference
