# Evaluation Scripts Documentation: MixKV & DivPrune for Qwen2.5-Omni

This document describes the two evaluation scripts created to benchmark Qwen2.5-Omni-7B
with KV-cache compression (MixKV) and visual token pruning (DivPrune). Both scripts are
adapted from the baseline `eval_qwen_omni.py` and implement the core algorithms from
the respective papers in a training-free manner.

---

## Table of Contents

1. [Baseline Script: eval_qwen_omni.py](#1-baseline-script-eval_qwen_omnipy)
2. [MixKV Script: eval_qwen_omni_mixkv.py](#2-mixkv-script-eval_qwen_omni_mixkvpy)
3. [DivPrune Script: eval_qwen_omni_divprune.py](#3-divprune-script-eval_qwen_omni_divprunepy)
4. [Default Configurations](#4-default-configurations)
5. [How the Methods Can Stack](#5-how-the-methods-can-stack)

---

## 1. Baseline Script: eval_qwen_omni.py

The baseline script loads Qwen2.5-Omni-7B, iterates over video MCQ entries in
`metadata.json`, and runs standard inference with no compression. Both new scripts
preserve the following baseline infrastructure unchanged:

| Component | Description |
|---|---|
| **Prompt builders** | `build_user_prompt_for_dataset()` with per-dataset templates (Video-MME, WorldSense, Daily-Omni, default) |
| **Video resolution** | `resolve_video_path()` searches by filename then recursive glob |
| **Audio detection** | `check_video_has_audio()` via PyAV; falls back to no-audio on decode failure |
| **Answer parsing** | `parse_answer()` imported from `mcq_answer_parse.py` |
| **Logging** | `Tee` / `StderrTee` classes for dual stdout+file logging |
| **VRAM tracking** | Per-question peak/current VRAM logged to `--vram_log` JSONL |
| **Generation** | Text-only output via `disable_talker()` + `return_audio=False`; `thinker_do_sample=False` |
| **Processor** | Uses `qwen_omni_utils.process_mm_info` from `OmniZip-main/qwen-omni-utils/src/` |

Both new scripts use identical prompt templates, answer parsing, video lookup,
audio handling, and result JSONL format so that results are directly comparable.

---

## 2. MixKV Script: eval_qwen_omni_mixkv.py

### 2.1 Paper Reference

**MixKV: Mixed Key-Value Cache Compression for Long-Context LVLMs** (ICLR 2026).

The paper observes that different attention heads exhibit heterogeneous KV redundancy.
Some heads ("visual heads") attend strongly to visual tokens and benefit from
diversity-based selection, while others benefit from importance-based selection.
MixKV mixes three scoring signals with per-head weights to select which KV tokens
to keep in cache.

### 2.2 What Was Added Over the Baseline

The following components are entirely new (not present in `eval_qwen_omni.py`):

#### a) `MixKVCompressor` class (lines 253-399)

A self-contained KV cache compressor attached to each attention layer. Implements
three scoring functions from the paper:

| Scoring Function | Paper Section | Implementation |
|---|---|---|
| **Attention importance** (`_attention_scores`) | Sec 3.2 "Observation-based Scoring" | Computes `Q[-window:] @ K^T / sqrt(d)`, applies causal mask, softmax, averages over the window queries, collapses GQA groups via mean, then avg-pools with `kernel_size=5`. Directly follows SnapKV (Sec 3.1) and `MixKV-main/mixkv/mixkv_utils.py:SnapKVCluster.update_kv()` lines 292-317. |
| **Key diversity** (`_similarity_scores`) | Sec 3.3 "Key Similarity Scoring" | Normalizes keys via `F.normalize`, computes cosine similarity of each key to the mean key vector, negates (so dissimilar = high score), min-max normalizes. Reimplements `calcul_similarity_score()` from `mixkv_utils.py` lines 233-250. |
| **Value norm** (`_value_norm_scores`) | Sec 3.3 "Value Norm Scoring" | Computes L2 norm of each value vector in the non-window region, min-max normalizes. From `mixkv_utils.py` lines 332-339. |

The `compress_kv()` method combines scores per the selected method:

| `--select_method` | Paper Equivalent | Combination Formula |
|---|---|---|
| `snapkv` / `attn` | SnapKV baseline | `score = attn_score` |
| `vnorm` | Attn + VNorm | `score = attn_score + scale * vnorm_score` |
| `headwisemixkv` | Full MixKV (Eq. 4 in paper) | `score = head_score * sim + (1 - head_score) * (attn + vnorm)` |

For `headwisemixkv`, the per-head weight `head_score` comes from pre-computed KV
similarity JSON (same format as `MixKV-main/visual_head/head_score/qwen_kv_similarity.json`).
Heads with high key redundancy (high similarity) get more weight on diversity;
heads with low redundancy get more weight on importance. This implements Equation 4
from Section 3.4 of the paper. If no JSON is provided, falls back to `head_score=0.5`
(equal weight).

After scoring, the top-k tokens are selected via `sort` + `gather` and concatenated
with the recent window tokens. This matches the selection logic in
`mixkv_utils.py:SnapKVCluster.update_kv()` lines 385-398.

#### b) `load_head_similarity_scores()` (lines 224-250)

Loads pre-computed per-(layer, head) KV similarity scores from a JSON file.
Format follows `MixKV-main/visual_head/head_score/qwen_kv_similarity.json`:
keys are `"layer-head-key"`, values are lists of scores across calibration samples.
Returns a `(num_layers, num_kv_heads)` tensor of averaged scores.

#### c) Monkeypatch: `_make_mixkv_sdpa_forward()` (lines 404-490)

Replaces `Qwen2_5OmniSdpaAttention.forward` on each decoder layer. The patched
forward is identical to the original SDPA forward (from
`OmniZip-main/omnizip/modeling_qwen2_5_omni.py` lines 1670-1763) except for the
cache update logic:

```
Original (line 1697-1699):
    key_states, value_states = past_key_value.update(key_states, value_states, ...)

Patched (lines 438-453):
    if q_len == 1:          # decode step
        normal cache update
    else:                   # prefill step
        k_comp, v_comp = self._mixkv_compressor.compress_kv(key_states, query_states, value_states)
        past_key_value.update(k_comp, v_comp, ...)
```

Key design decision: During prefill, the **full uncompressed KV** is used for computing
the attention output of the current tokens (line 456-457). Only the **compressed KV**
is stored in the cache for future decode steps. This matches the MixKV paper's approach
and the reference implementation in `MixKV-main/mixkv/qwen_model.py` lines 82-91.

The original MixKV repo patches `Qwen2VLFlashAttention2.forward` (targeting Qwen2-VL).
Our patch targets `Qwen2_5OmniSdpaAttention.forward` (Qwen2.5-Omni's thinker uses SDPA
by default with `attn_implementation="sdpa"`). The attention class structure is nearly
identical — both use `apply_multimodal_rotary_pos_emb` with `mrope_section`, same GQA
structure, same QKV projection shapes.

#### d) `apply_mixkv_to_model()` (lines 493-525)

Walks `model.thinker.model.layers`, attaches a `MixKVCompressor` instance to each
layer's `self_attn`, and replaces the forward method via `types.MethodType`. Each
compressor is configured with the layer's `num_key_value_heads` and `num_key_value_groups`
read from the attention module.

#### e) New CLI arguments

| Argument | Default | Purpose |
|---|---|---|
| `--budget` | `256` | Total KV token capacity per head (includes window) |
| `--window_size` | `32` | Recent tokens always retained in full |
| `--select_method` | `snapkv` | Token scoring method |
| `--head_score_path` | `None` | Path to pre-computed head similarity JSON |

#### f) Result JSONL additions

Each result dict adds `"method": "mixkv-{select_method}"` and `"budget": N` fields
for tracking which compression config produced each answer.

---

## 3. DivPrune Script: eval_qwen_omni_divprune.py

### 3.1 Paper Reference

**DivPrune: Diversity-Driven Visual Token Pruning for Large Multimodal Models**
(CVPR 2025).

The paper prunes redundant visual tokens before they enter the LLM decoder using a
greedy max-min diversity algorithm on cosine distance. The key insight is that many
visual tokens (especially from adjacent patches) are highly similar and can be pruned
without accuracy loss.

### 3.2 What Was Added Over the Baseline

#### a) `pairwise_cosine_distance()` (lines 210-221)

Computes the `(N, N)` distance matrix as `1 - cosine_similarity`. Directly implements
the distance metric from the paper. Matches the reference implementation at
`divprune-main/LLaVA/llava/model/llava_arch.py` line 155:
```python
cosine_matrix = 1.0 - (self.pairwise_cosine_similarity(visual_feature_vectors))
```

#### b) `divprune_select()` (lines 224-258) — Core Algorithm

This is the paper's Algorithm 1: greedy farthest-point diversity selection.
Reimplements the `DivPrune()` method from `llava_arch.py` lines 152-171.

**Step-by-step mapping to the reference code:**

| Our Implementation | Reference (`llava_arch.py`) | Paper |
|---|---|---|
| Line 247-249: First token = `argmax` of 2nd-smallest distance | Lines 164-165: `topk(m2, 2, largest=False).values[1,:]` then `argmax` | Alg. 1 step 1: seed with most isolated token |
| Lines 251-256: For subsequent tokens, `min` over selected rows then `argmax` | Lines 162-163, 167: `index_select` rows of selected, `min(dim=0)`, `argmax` | Alg. 1 step 2: iteratively pick token maximizing min-distance to selected set |
| Line 242: Pre-compute full `(N,N)` distance matrix | Line 154-155: Pre-compute `cosine_matrix` | Same — avoids redundant pairwise computations |

The algorithm has `O(k * N)` complexity where `k` = tokens to keep, `N` = total tokens.

#### c) `divprune_select_frames()` (lines 261-279) — Frame-Level Mode

Adapts DivPrune from token-level (as in the paper) to frame-level for Qwen2.5-Omni.

The paper applies DivPrune on **visual token embeddings** inside
`prepare_inputs_labels_for_multimodal()` (llava_arch.py lines 365-385), after the
vision encoder and projector produce per-patch embeddings.

For Qwen2.5-Omni, the vision encoder is inside the thinker model and not easily
intercepted. The frame-level mode instead:
1. Takes the raw video tensor `(nframes, C, H, W)` returned by `process_mm_info()`
2. Creates frame-level features via global average pooling: `mean(dim=(-2, -1))` → `(nframes, C)`
3. Runs `divprune_select()` on these frame features
4. Sorts selected indices to maintain temporal order
5. Passes only selected frames to the processor

This is a coarser granularity than the paper (frame groups vs. individual patches)
but requires no model surgery and is the recommended starting point.

#### d) Token-Level Mode (lines 289-326 in `run_inference`)

An experimental mode that prunes `pixel_values_videos` patches after the processor
but before the model forward. This is closer to the paper's approach:
1. Takes `inputs["pixel_values_videos"]` — the flattened visual patches
2. Runs `divprune_select()` on the patch features
3. Replaces `pixel_values_videos` with the pruned subset
4. Scales down `video_grid_thw` temporal dimension proportionally

This approximates the paper's token-level pruning without hooking inside the model.

#### e) New CLI arguments

| Argument | Default | Purpose |
|---|---|---|
| `--subset_ratio` | `0.5` | Fraction of frames/tokens to KEEP |
| `--prune_mode` | `frame` | `frame` (pre-model) or `token` (post-processor) |

#### f) Result JSONL additions

Each result adds: `"method": "divprune-{mode}"`, `"subset_ratio"`, `"orig_frames"`,
`"pruned_frames"`. The summary also reports the effective frame reduction ratio.

---

## 4. Default Configurations

### 4.1 MixKV Defaults

| Parameter | Our Default | MixKV Paper/Repo Default | Source |
|---|---|---|---|
| `budget` | 256 | 256 (also tested 64, 128, 512) | `mixkv_utils.py`: `max_capacity_prompt = int(os.getenv('BUDGET'))`, eval scripts use 256 |
| `window_size` | 32 | 32 | `mixkv_utils.py:SnapKVCluster.__init__`: `window_size=64` in signature, but `init_snapkv()` uses `config.window_size = 32` |
| `kernel_size` | 5 | 5 | `init_snapkv()`: `config.kernel_size = 5` |
| `pooling` | avgpool | avgpool | `init_snapkv()`: `config.pooling = 'avgpool'` |
| `gqa_func` | mean | mean (for Qwen) | `init_snapkv()`: `config.gqa_func = 'mean'` for qwen model types |
| `select_method` | snapkv | headwisemixkv (paper's full method) | We default to `snapkv` because `headwisemixkv` requires pre-computed head scores that don't yet exist for Qwen2.5-Omni |
| `head_score` | 0.5 fallback | Pre-computed from calibration | `qwen_kv_similarity.json` exists for Qwen2-VL but not Qwen2.5-Omni; 0.5 = equal weight on diversity vs. importance |
| `dtype` | bfloat16 | bfloat16 | Standard for Qwen models |
| `attn_implementation` | sdpa | flash_attention_2 (in MixKV repo) | We use SDPA because Qwen2.5-Omni defaults to it; the original MixKV patches FlashAttention2 |

### 4.2 DivPrune Defaults

| Parameter | Our Default | DivPrune Paper/Repo Default | Source |
|---|---|---|---|
| `subset_ratio` | 0.5 | 0.098 (keep ~10%) | `run_Divprune.sh`: `SUBSET_RATIO=0.098`, `llava_arch.py:DivPrune`: `threshold_ratio=0.1` |
| `prune_mode` | frame | token (in paper, on LLaVA) | Paper prunes individual visual tokens; we default to frame-level because Qwen2.5-Omni's vision encoder is harder to intercept |

**Why `subset_ratio=0.5` instead of 0.098?** The paper's 0.098 ratio was calibrated
for LLaVA-1.5, which produces 576 visual tokens per image (keeping ~56). Qwen2.5-Omni
at `fps=2.0` produces far fewer frames to begin with (typically 4-30 depending on video
length). Pruning to 10% at frame level would often leave only 1-3 frames, which is too
aggressive. Start with 0.5 and sweep down (0.3, 0.2) to find the accuracy-efficiency
tradeoff.

### 4.3 Shared Defaults (from baseline)

| Parameter | Default | Source |
|---|---|---|
| `fps` | 2.0 | Matches `eval_qwen_omni.py`, lmms-eval Qwen2.5-Omni config |
| `max_pixels` | 360*420 (151,200) | Matches baseline |
| `max_new_tokens` | 4096 | Matches lmms-eval qwen2_5_omni default |
| `dtype` | bfloat16 | Standard for Qwen models on A100/H100 |
| `attn_implementation` | sdpa | Qwen2.5-Omni default |

---

## 5. How the Methods Can Stack

These three compression methods operate at different points in the pipeline:

```
Video Frames
    │
    ▼
[DivPrune]  ← prune frames/patches by diversity (pre-model)
    │
    ▼
Vision Encoder → Visual Tokens
    │
    ▼
[OmniZip]   ← compress audio-visual tokens in attention (in-model, rho_video/rho_audio)
    │
    ▼
LLM Decoder Layers
    │
    ▼
[MixKV]     ← compress KV cache per-head during prefill (post-encoding, affects decode)
    │
    ▼
Text Output
```

They are complementary:
- **DivPrune** reduces the number of visual inputs the model processes
- **OmniZip** reduces token count inside the attention mechanism
- **MixKV** reduces memory during autoregressive decoding

A combined run would use DivPrune's frame pruning, then feed the reduced frames
into a model with MixKV-patched attention layers.

---

## 6. File Summary

| File | Lines | What It Does |
|---|---|---|
| `eval_qwen_omni.py` | 572 | Baseline: standard Qwen2.5-Omni inference |
| `eval_qwen_omni_zip.py` | 584 | OmniZip: token compression via modified model class |
| `eval_qwen_omni_mixkv.py` | ~560 | **New**: MixKV KV-cache compression via SDPA monkeypatch |
| `eval_qwen_omni_divprune.py` | ~480 | **New**: DivPrune diversity-based frame/token pruning |
