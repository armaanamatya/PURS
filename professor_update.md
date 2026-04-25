# Professor Progress Update

## Short Version

Hi Professors,

Quick update on the OmniZip / early-layer saliency project. I ran a layer sweep over Qwen2.5-Omni Thinker layers `[0, 1, 3, 6, 10, 14, 20, 27]` to see whether an internal saliency signal could replace OmniZip's audio saliency score. I started with Spearman, Gini, and score spread because I first wanted to know two basic things: whether the signal changes across questions, and whether it becomes concentrated enough to support pruning. The first finding was that the ranking is almost completely question-invariant across prompts, which suggested caching might be possible. The second finding was that the first strong concentration jump happens at Layer 6, with another peak at Layer 14. That left L6 and L14 as the main candidates, but those descriptive metrics were not enough to tell which one actually matches OmniZip.

Because of that ambiguity, I added stronger pruning-oriented metrics: ROC AUC against OmniZip's actual audio keep mask, top-k Jaccard overlap, separation at the pruning threshold, and temporal autocorrelation. The reason for this second pass was that I needed to test not just whether the signal looks structured, but whether it actually reproduces OmniZip's keep/drop behavior. That was the decisive step: Layer 6 gets `0.653 +/- 0.111` AUC, while Layer 14 is `0.525 +/- 0.066` (near random), and L6 beats L14 on `58/59` questions. So L6 is the only layer that looks both cacheable and aligned with OmniZip.

I then used precomputed Layer-6 saliency to replace OmniZip's native audio saliency in the real pipeline. On the full 118-question benchmark, OmniZip+L6-cache gets `34/118 = 28.8%` versus `36/118 = 30.5%` for stock OmniZip, while preserving essentially the same efficiency: `1470 ms` prefill versus `1455 ms`, with the same `18.5 GB` peak allocated VRAM. In the paired comparison, the cached-L6 system gives the exact same prediction as OmniZip on `114/118` questions, so it is acting as a high-fidelity replacement signal rather than a loose approximation.

I think this is a good basis to build on because it suggests an early Thinker layer already contains a question-invariant, cacheable saliency signal that overlaps with OmniZip's pruning decisions. The next missing experiment is the multi-turn setting, where the same video is queried multiple times and the cache can actually be reused.

Best,
Armaan

## Slightly More Detailed Version

Hi Professors,

I wanted to send a more complete update on the Layer-6 cache idea for OmniZip.

The main question I tested was whether an early Thinker-layer signal inside Qwen2.5-Omni could replace OmniZip's handoff audio saliency signal. OmniZip normally uses saliency derived from the audio encoder's own self-attention. My hypothesis was that an early Thinker layer might already encode a ranking of which audio tokens matter most, and that this ranking might be stable enough to precompute once per video and reuse across questions.

To test that, I ran a layer sweep over `[0, 1, 3, 6, 10, 14, 20, 27]` and looked at several metrics. The first pass used cross-question Spearman correlation, Gini concentration, and score spread. Those gave two important early findings:

- The saliency ranking is almost completely question-invariant, which means it looks more like a video-level semantic saliency signal than a question-specific relevance map.
- The first strong concentration jump happens at Layer 6, with another strong peak at Layer 14.

That left an ambiguity: L6 and L14 both looked promising if I only looked at concentration. So I added stronger metrics that are closer to the actual pruning problem:

- ROC AUC against OmniZip's real audio keep mask
- Top-k Jaccard overlap between question-specific selections
- Separation between kept and dropped tokens at the pruning threshold
- Temporal autocorrelation to see whether the signal forms coherent blocks instead of noise

The AUC test turned out to be the decisive one. Layer 6 gets `0.6528 +/- 0.1105` versus OmniZip's actual audio keep mask, while Layer 14 gets only `0.5253 +/- 0.0662`. Layer 6 beats Layer 14 on `58/59` questions. So although L14 looks concentrated by Gini, it is not actually aligned with OmniZip's real pruning behavior. That is why I ended up focusing on L6.

After that, I ran the actual replacement experiment. I precomputed Layer-6 saliency offline once per video using the stock Qwen2.5-Omni model, then patched OmniZip so that its `attn_logits` audio saliency input comes from the cached Layer-6 scores instead of the audio encoder's native score. Everything else in OmniZip stayed the same: same pruning ratios, same video compression, same token merging, same Thinker and decoder.

On the full 118-question benchmark:

- Baseline: `35/118 = 29.7%`
- OmniZip: `36/118 = 30.5%`
- OmniZip + cached L6: `34/118 = 28.8%`

Efficiency is essentially unchanged relative to OmniZip:

- OmniZip prefill: `1455 ms`
- OmniZip + cached L6 prefill: `1470 ms`
- Both use `18.5 GB` peak allocated VRAM

The paired comparison is the most encouraging part. Cached L6 and OmniZip give the exact same prediction on `114/118` questions. Only 2 questions flip from correct under OmniZip to incorrect under the cached version, and there are no broad failure patterns. So the right interpretation is not that cached L6 is a weaker system overall, but that it is a high-fidelity surrogate for OmniZip's audio saliency.

Why I think this is promising:

1. It reveals a mechanistic result: an early Thinker layer already contains a usable cross-modal saliency signal.
2. That signal is effectively question-invariant, which makes caching realistic.
3. It is strong enough to replace OmniZip's native audio saliency with only a very small paired degradation.
4. It opens the door to a real systems advantage in multi-turn settings, where the same video is queried multiple times.

So my current view is that this is already a strong insight plus proof-of-concept. The missing experiment that would really elevate it is to measure the multi-turn scenario directly: for repeated queries on the same video, the Layer-6 cache is reused once, while OmniZip still has to recompute its native saliency every time.

Best,
Armaan
