# Analysis: motion / frame-difference patch pruning for VideoZip

Date: 2026-05-29. Question: "divide a frame into n×n patches, measure motion/frame-difference
per patch through the video, prune or merge those video patch tokens — and let video guide
audio pruning. Would this work?"

## Verdict

Yes — the mechanism is sound and *already validated*. But it is not new: it is almost exactly
**OmniSIFT** (arXiv:2602.04804, Feb 4 2026), which beats OmniZip on Qwen2.5-Omni. So the real
question is not "would it work" but "what is the training-free / mechanistic wedge vs OmniSIFT."

## Two mechanisms were fused in one sentence — they are different operations

The proposal states the *goal* as "video guides audio pruning" but describes the *mechanism* of
motion-based pruning of **video** tokens. These have different prior art:

1. **motion / frame-diff → prune VIDEO tokens** = OmniSIFT's STVP / DyCoke (temporal merge).
2. **video → prune AUDIO tokens** = OmniSIFT's VGAS / the OmniZip-inverse = our VideoZip goal.

Answer both, but name the conflation: motion-based video pruning does **not** by itself tell you
which *audio* to keep. The video→audio link is a separate, harder problem.

## What OmniSIFT actually does (the published version of this idea)

Two-stage, **trained** (4.85M params, straight-through estimator), per 2-frame chunk:

- **STVP (Spatio-Temporal Video Pruning):**
  - *Spatial saliency* (frame 1): cosine distance of each token from the frame's mean-pooled
    vector. High = diverges from global frame context = informative.
  - *Temporal saliency* (frame 2): cosine distance between a patch token and its position-matched
    token in frame 1. High = changed over time = motion/new content. **This is the user's
    "frame difference per patch" — but computed in vision-encoder token-embedding space, at the
    2×2-merged token grid, not pixel space.**
  - TopK retention per ρ_v over both scores.
- **VGAS (Vision-Guided Audio Selector):** cross-attention with audio as query, pruned video as
  key/value → MLP+sigmoid saliency per audio token. This is the *learned* video→audio link.

Result (Qwen2.5-Omni-7B, 35% retained): OmniSIFT 50.5 avg vs OmniZip 54.1 vs Full 48.1 on
video-SALMONN-2 (lower better); wins or ties on the others; on-par latency/memory with
training-free baselines.

## Why pure motion will underperform (keep these design constraints)

1. **Pixel-space motion is corrupted.** Global camera pan/zoom makes *every* patch "move";
   lighting/exposure shifts and codec/JPEG noise add false motion. The ranking gets dominated by
   ego-motion. → Compute the difference on **encoder hidden states** (cosine, like OmniSIFT), not
   raw pixels. It's nearly free post-encoder and far more semantic.
2. **Motion ≠ importance.** Static-but-critical content is lost: on-screen text/slides, a held
   object, a still face, a chart. High-motion-but-irrelevant content is kept: crowds, foliage,
   water, shake. This is exactly why OmniSIFT *combines* temporal with **spatial** saliency. A
   motion-only training-free version will regress on text/static-scene tasks — keep the spatial term.
3. **Per-chunk, not global ranking.** Qwen2.5-Omni groups video into 2-frame chunks
   time-aligned (TMRoPE) with audio. Ranking motion *globally* across the whole video would
   starve low-motion segments of all tokens. Operate per chunk.
4. **Grid must match Qwen's real token layout.** 14px patches, 2×2 spatial merge, 2-frame
   temporal grouping. "n×n patches" must map onto that merged token grid or the pruning won't
   correspond to actual LLM tokens.

## The video→audio half: do NOT conflate L6 with VGAS

- VGAS is genuinely cross-modal (video → audio decision).
- Our **L6 finding is question-invariant AUDIO self-saliency** — it does not route video into the
  audio choice. Offering L6 as "training-free VGAS" is a category error.
- If we want real *video-guided* audio selection training-free, the natural analog is to **harvest
  the model's own audio↔video cross-attention at some Thinker layer (read it, don't train it)** —
  not L6.
- If we only want good training-free audio compression, L6 is the better lever — but then "video
  guides audio" is a red herring and we should say so and drop that framing.

## Two ideas to test, NOT assert as advantages

- **Skip encoding pruned patches → save encoder FLOPs** (vs OmniSIFT which prunes post-encoder).
  Architecturally non-trivial: ViT attention spans all patches and the 2-frame temporal merge
  needs both frames. Hypothesis to benchmark, not a free win.
- **Merge vs prune.** The OmniSIFT table shows TopK-prune (trained) beating DyCoke merge
  (training-free) — but that's confounded by training. Evidence is mixed; A/B it, don't assume
  merge wins.

## The bar for "would it work" on THIS project

From the 10× speedup matrix: VideoZip is a Pareto play on prefill/VRAM at iso-accuracy, not an
accuracy win. So success = **training-free AND** either (a) beat OmniZip's prefill at iso-accuracy,
or (b) approach OmniSIFT's accuracy *without* its 4.85M trained params. A training-free STVP-analog
(spatial+temporal cosine saliency, per-chunk) plus a *read-not-trained* cross-attention audio
selector is the defensible, novel wedge.
