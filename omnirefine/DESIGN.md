# OmniRefine — System Design

Reference implementation of **OmniRefine: Alignment-Aware Cooperative
Compression for Efficient Omnimodal Large Language Models**
(Deng et al., arXiv:2605.12056v1, 12 May 2026).

This document is the `/system-design` deliverable: data flow, the
layer-probe boundary, and module decomposition.

---

## 1. What the paper proposes

A **training-free, two-stage** token compressor for Omni-LLMs
(Qwen2.5-Omni), applied to interleaved audio/video tokens *before* the LLM
prefill stage.

1. **CPCR — Correspondence-Preserving Chunk Refinement** (Sec 3.3).
   Native fixed-duration audio/video chunks are only *coarse* alignment
   priors. CPCR refines their boundaries into cross-modally aligned
   compression units by solving a temporally-constrained joint segmentation
   over frames and audio tokens with dynamic programming (Algorithm 1).

2. **MACC — Modality-Aware Cooperative Compression** (Sec 3.4).
   Within each refined chunk:
   - **Video**: tree-structured spatio-temporal compression (coarse-to-fine
     2×2 quadtree per frame + temporal merge across frames).
   - **Audio**: semantic-anchor compression (anchor detection + saliency
     selection + nearest-anchor assignment + weighted fusion).
   - **Cross-modal budget**: the audio budget is referenced to the observed
     video retention so the two modalities stay coordinated.

## 2. The load-bearing boundary: the probe layer L

OmniRefine is training-free, so it has **no parameters of its own** — every
signal it needs (frame/audio embeddings, region representations `z(·)`,
audio saliency) must be read from a **partial forward pass** of the model at
some layer **L**. The compression then produces a *keep-mask*, the sequence
is pruned, and prefill resumes.

```
            ┌──────────────── adapter.py (model-specific) ────────────────┐
 inputs ──► │ prefill 0..L ─► read H_L, attention_L, native chunk indices │
            └───────────────────────────┬─────────────────────────────────┘
                                         │ ProbeInputs  (pure data)
                                         ▼
            ┌──────────── pipeline.compress()  (pure, model-free) ─────────┐
            │  Stage 1  cpcr.py        : CPCR  → refined chunks            │
            │  Stage 2  video_compress : quadtree + temporal merge → R_v   │
            │           budget.py      : Eq 13/14  R_v → R_a               │
            │           audio_compress : anchors + assign + fuse → keep    │
            └───────────────────────────┬─────────────────────────────────┘
                                         │ KeepMask  (ids + merged reps)
                                         ▼
            ┌──────────────── adapter.py (model-specific) ────────────────┐
            │ prune KV / tokens, scatter merged reps, resume prefill L+1   │
            └──────────────────────────────────────────────────────────────┘
```

**The paper never states which layer L is used** (the motivating analysis in
Sec 3.1 merely *inspects* layers 0 and 8). It is therefore a config
parameter (`OmniRefineConfig.layer_probe`, default 8) and is flagged in
REPRODUCTION_NOTES.md item #2. This is the single most consequential free
choice in any reproduction.

## 3. Module decomposition

| File | Responsibility | Paper anchor |
|------|----------------|--------------|
| `config.py` | All hyperparameters as a dataclass | Sec 4.1, Appendix A |
| `utils.py` | cosine, mean-pool `z(·)`, integral images | Eq 12 |
| `cpcr.py` | Stage 1: similarity field + DP refinement | Sec 3.3, Eq 3–7, Alg 1 |
| `video_compress.py` | Quadtree + temporal merge (`Compress_v`) | Sec 3.4, Eq 6–8, 11 |
| `audio_compress.py` | Semantic-anchor audio (`Compress_a`) | Sec 3.4, Eq 9–11 |
| `budget.py` | Cross-modal budget allocation | Appendix B.1, Eq 13–14 |
| `pipeline.py` | `compress(ProbeInputs) → KeepMask` (pure core) | Sec 3.2 |
| `torch_runtime.py` | Qwen prefill bridge: `inputs_embeds` → `ProbeInputs` → sequence mask | Sec 3.2, Sec 3.3, Sec 3.4 |
| `adapter.py` | Model integration boundary (stubs) | Sec 3.2, 4.1 |

## 4. Design decisions

- **Pure NumPy core, no torch.** The algorithm is tensor-level and
  training-free; keeping the core model-free makes every invariant testable
  without a GPU or checkpoint. Adapters convert torch tensors at the seam
  (`utils.as_numpy`).
- **Algorithm 1 (appendix) is the source of truth** for the DP, not the
  main-text Eq 5 — they disagree (see REPRODUCTION_NOTES #3). Integral
  images give O(1) block statistics → overall DP is
  `O(F·N·S_Vmax·S_Amax)` inside the band.
- **Hard retention bounds are enforced post-hoc** (Appendix A): the
  threshold-driven quadtree yields an observed `R_v`; we clamp into
  `[v_min, v_max]` by dropping low-saliency nodes (above) or adding back
  high-saliency individual tokens (below).
- **Saliency is an explicit input**, not invented internally — the paper's
  "fused attention-based importance" is under-specified, so the adapter
  supplies it and the choice is documented rather than hidden.
- **Prefill bridge is intentionally layer-0.** `torch_runtime.py` is a
  practical Qwen hook for already-merged `inputs_embeds`: it scatters merged
  Eq 11 anchor reps and returns the same style of boolean mask used by local
  OmniZip. Applying OmniRefine after an internal decoder layer still requires
  pruning K/V caches for already-run layers, which the paper does not specify
  and this bridge does not guess.

## 5. What is intentionally out of scope (minimal mode)

- The `probe()` / `apply()` model wiring (raises `NotImplementedError`):
  version-specific to the Qwen2.5-Omni runtime and KV-cache layout.
- Accuracy reproduction (WorldSense 46.7% @ 44% retention, etc.): requires
  the model + benchmarks; see REPRODUCTION_NOTES #7.
- Training: none — the method is training-free.
