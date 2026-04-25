# Future Planning

## Core Strategic Point

If we keep the same token budget and only swap OmniZip's scorer, we should expect to **match OmniZip**, not clearly beat it. To beat OmniZip on single-query inference, we likely need at least one of:

- fewer tokens at the same accuracy
- smarter video pruning
- adaptive per-instance compression
- a stronger hybrid saliency signal than either scorer alone

## Most Promising Next Directions

### 1. Hybrid score instead of pure replacement

Use the cached L6 score together with OmniZip's native encoder score rather than forcing one to fully replace the other.

Why this may help:

- single heuristics are often not enough
- L6 may capture question-invariant semantic importance
- OmniZip's encoder saliency may still capture local modality-specific cues
- a hybrid score may preserve the strengths of both

Potential form:

```text
score_hybrid = alpha * score_L6 + (1 - alpha) * score_encoder
```

Relevant literature:

- AgilePruner: attention and diversity help in different regimes
- VisPruner: attention alone is not always an ideal pruning signal

### 2. Adaptive `rho_audio` / `rho_video`

Right now OmniZip uses fixed pruning ratios. We can instead let the saliency distribution determine how aggressively to prune each sample.

Why this may help:

- highly peaked saliency suggests the content is compressible
- flatter or higher-entropy saliency suggests the content is harder and should be pruned less
- fixed ratios are unlikely to be optimal for every video

Simple idea:

- if audio saliency is sharp, increase pruning
- if audio saliency is flat, reduce pruning
- do the same on the video side using text-to-video saliency

Relevant literature:

- ATP-LLaVA: fixed token ratios are often suboptimal
- TrimTokenator: adaptive compression improves the tradeoff

### 3. Two-stage audio pruning

Use cached L6 as a coarse first-pass filter, then run OmniZip's native scorer only on the remaining audio tokens for a fine pass.

Why this may help:

- keeps the cacheable offline advantage of L6
- still allows OmniZip's native scorer to refine difficult borderline cases
- is the cleanest path to potentially beating OmniZip while staying very close to the existing pipeline

Relevant literature:

- FastAV: two-stage global/fine pruning strategy

### 4. Upgrade the video side, not just the audio side

Since we already measured text-to-video saliency, we can use that signal to improve OmniZip's video pruning, not only its audio saliency.

Why this may help:

- video tokens often have strong temporal redundancy
- importance alone may keep many near-duplicate tokens
- combining importance with similarity-aware merging may give a better tradeoff

Relevant literature:

- FrameFusion: important tokens can still be redundant, so similarity-aware merging helps

### 5. Add a diversity term to the retained set

Our current saliency signal ranks importance, but the selected tokens may still be redundant.

Why this may help:

- top-ranked tokens can cluster around the same event or frame region
- diversity-aware selection can spread coverage across time or content types
- this is especially relevant on the video side

Relevant literature:

- DivPrune
- VisPruner

### 6. Replace OmniZip's time-group retention score too

OmniZip does not only use token-level audio saliency. It also aggregates audio retention over time groups to guide video pruning.

Potential extension:

- aggregate L6 token saliency into group-level retention scores
- use those scores in place of OmniZip's current time-group guidance

Why this may help:

- L6 may align more directly with semantically meaningful audio events
- if the group-level signal improves, the downstream video pruning step may improve too

### 7. Lightweight learned residual on top of cached L6

If we are willing to train a little, we can keep cached L6 as the backbone signal and learn a small residual correction head.

Why this may help:

- preserves the mechanistic insight
- stays much lighter than a fully learned compressor
- may recover the small residual gap to OmniZip

Relevant literature:

- OmniSIFT: small learned modules can outperform purely training-free baselines

## Top 3 Priorities

If we want the most practical next bets, the strongest three are:

1. hybrid audio score
2. adaptive compression ratio
3. importance + similarity on the video side

These have the best chance of actually beating OmniZip rather than only matching it.

## Why This Direction Fits The Literature

- Fixed-ratio pruning is often suboptimal: ATP-LLaVA, TrimTokenator
- Attention-only importance is often suboptimal: VisPruner, AgilePruner
- Similarity and redundancy matter a lot in video: FrameFusion
- Decoupling or adaptive sparsity can improve speed/quality tradeoffs: TopV, SparseVILA

## Concrete Ablation Plan

### Audio scoring ablations

1. L6 only
2. OmniZip encoder score only
3. `alpha * L6 + (1 - alpha) * encoder`
4. L6 coarse pass + encoder fine pass

### Adaptive pruning ablations

1. dynamic `rho_audio`
2. dynamic `rho_video`
3. dynamic `rho_audio` + dynamic `rho_video`

### Video-side ablations

1. L6-based text-to-video score
2. L6 importance + similarity-aware merge
3. replace group-level retention score with L6-derived score

## Practical Recommendation

If time is limited, the best order is:

1. implement the hybrid audio score
2. test adaptive `rho_audio`
3. test video-side importance + similarity

That sequence stays closest to the current result, is easy to explain in the paper, and has the best chance of showing a real improvement over OmniZip.
