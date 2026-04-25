# Other Points

## Research Questions

This note captures three related high-level directions for OmniZip-style multimodal pruning/compression:

1. Can we compute a **video saliency score inside the video encoder** rather than only after visual encoding or inside the LLM?
2. Can we compute a **text saliency score inside the text encoder** and use it as a pruning/compression signal?
3. Can we use **text-to-video alignment** or a cross-modal alignment model as the signal source for pruning?

## Short Answer

All three ideas are viable. The main difference is how crowded each space already is.

- **Video saliency inside the video encoder:** plausible and supported by nearby prior work. The more interesting version is doing this at **intermediate video-encoder layers** for VideoLLMs or omni models rather than only at the encoder output.
- **Text saliency inside the text encoder:** also plausible. This has strong precedent in NLP token pruning, but appears less explored specifically as a multimodal pruning/control signal.
- **Text-to-video alignment as a pruning signal:** definitely viable, but this is the most explored of the three because many recent methods already use language-guided or cross-modal relevance for visual token pruning.

## Novelty Assessment

From most novel to least novel:

1. **Text saliency inside the text encoder for multimodal pruning**
2. **Video saliency inside intermediate video-encoder layers for VideoLLMs / omni models**
3. **Text-to-video alignment as a pruning signal**

Why this ordering:

- Cross-modal text-guided visual pruning already has a large and active literature.
- Encoder-internal visual pruning exists, but the VideoLLM and omni-specific version is still less saturated.
- Text-side saliency is common in NLP, yet comparatively underused as a multimodal pruning signal.

## What Prior Work Already Covers

### 1. Video / vision saliency inside or around the encoder

These papers are the closest prior art for "video saliency score inside the video encoder."

- **DynamicViT** introduces layer-wise token importance prediction inside a vision transformer. This is an important precursor to encoder-internal saliency pruning.  
  Source: [DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](https://arxiv.org/abs/2106.02034)

- **LaCo** performs compression within intermediate vision-encoder layers for multimodal LLMs. This is very close in spirit to the "inside the encoder" framing.  
  Source: [LaCo: Layer-wise Compression for Efficient Multimodal Large Language Models](https://openreview.net/forum?id=UDCiCnVGhq)

- **SPIDER** argues that middle-layer vision-encoder tokens are especially informative for token reduction because semantic focus shifts across layers.  
  Source: [SPIDER: Adaptive Token Selection in Vision Encoder for Efficient Multimodal LLMs](https://openreview.net/forum?id=aGpSK6QH3w)

- **HIVTP** uses intermediate vision-encoder attention signals to estimate token importance, reinforcing the idea that early or middle layers can provide useful saliency cues.  
  Source: [HIVTP: Hierarchical Importance-based Vision Token Pruning for Efficient Multimodal Large Language Models](https://arxiv.org/abs/2509.23663)

- **PruneVid** is a strong VideoLLM reference point. It is not purely "inside the video encoder," but it is highly relevant because it prunes video tokens based on question-aware importance.  
  Source: [PruneVid: Visual Token Pruning for Efficient Video Large Language Models](https://arxiv.org/abs/2412.16117)

### 2. Text saliency inside the text encoder

These papers support the idea that text tokens can carry explicit saliency or pruning scores inside a text encoder.

- **TR-BERT** dynamically reduces text tokens by deciding how long each token should continue through the encoder. This is a classic NLP reference for encoder-side text token saliency.  
  Source: [TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference](https://arxiv.org/abs/2105.11618)

- **Saliency-driven Dynamic Token Pruning for LLMs** uses hidden-state saliency to prune text tokens hierarchically. This is relevant if we want a saliency score derived directly from text representations rather than attention alone.  
  Source: [Saliency-driven Dynamic Token Pruning for Large Language Models](https://arxiv.org/abs/2504.04514)

- **PuMer** is especially relevant because it reduces both image and text tokens in a vision-language model. This is a useful bridge between pure NLP token pruning and multimodal token reduction.  
  Source: [PuMer: Pruning and Merging Tokens for Efficient Vision Language Models](https://aclanthology.org/2023.acl-long.721/)

### 3. Text-to-video or cross-modal alignment as the pruning signal

This area is already active and probably the least novel of the three ideas.

- **LVPruning** uses language-guided cross-attention to score visual tokens. This is close to "text as the supervision signal for pruning visual tokens."  
  Source: [LVPruning: Language-guided Visual Token Pruning for Multimodal Large Language Models](https://arxiv.org/abs/2501.13652)

- **CATP** uses cross-attention in multimodal models as a token-importance estimator.  
  Source: [CATP: Cross-Attention Token Pruning for Efficient Multimodal Large Language Models](https://arxiv.org/abs/2404.08567)

- **MI-Pruner** explicitly computes mutual information between visual and textual features before full multimodal interaction. This is especially relevant if we want a more principled alignment-based pruning score.  
  Source: [MI-Pruner: Mutual Information-based Token Pruning for Efficient Multimodal Large Language Models](https://arxiv.org/abs/2604.03072)

- **ConsensusDrop** combines encoder-side visual saliency with cross-modal saliency. This is directly relevant to the idea of mixing unimodal and multimodal signals.  
  Source: [ConsensusDrop: Consensus-aware Visual Token Compression for Multimodal LLMs](https://arxiv.org/abs/2602.00946)

- **CenterCLIP** comes from text-video retrieval and reduces redundant video tokens while preserving text-video alignment. It is not framed as LLM pruning, but conceptually it supports text-video alignment as a compression signal source.  
  Source: [CenterCLIP: Token Clustering for Efficient Text-Video Retrieval](https://arxiv.org/abs/2205.00823)

## Main Insight

The strongest research direction is probably **not** "use language to prune vision tokens" by itself, because that idea already has many close neighbors.

The more interesting angle is a **joint encoder-side saliency framework**:

- derive **video saliency** from intermediate video-encoder states
- derive **text saliency** from intermediate text-encoder states
- derive **cross-modal agreement** between the two
- use the combination to decide whether to **keep, prune, or merge** tokens

This is more distinct than a standard language-guided visual pruning method because it treats both modalities as first-class saliency carriers rather than using text only as a query.

## Why This Could Matter for OmniZip

For an OmniZip-like setup, this suggests a few concrete possibilities:

- Instead of audio-only guidance for video pruning, use **multiple internal saliency channels**:
  - audio saliency
  - video-encoder saliency
  - text-encoder saliency
  - cross-modal agreement

- The pruning rule could become **agreement-aware**:
  - keep tokens that are salient in both unimodal and cross-modal space
  - merge tokens that are low-saliency but redundant
  - prune tokens that are consistently low across both sources

- This may help separate:
  - tokens that are visually salient but irrelevant to the prompt
  - tokens that are text-relevant but visually weak
  - tokens that are globally redundant

## Suggested Research Positioning

If we want something that feels more publishable than incremental, a promising claim would be:

**Joint encoder-side unimodal + cross-modal saliency for multimodal token pruning**

That framing sounds stronger than:

- only question-guided visual pruning
- only attention-based visual saliency
- only post-encoder token scoring

## Caution / Reality Check

One paper worth keeping in mind as a sanity check is:

- **Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?**  
  Source: [Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.802/)

This is useful because it questions whether current pruning methods and benchmarks are actually identifying the right redundancy. It is a good paper to cite if we need to justify stronger analysis of saliency definitions and evaluation setup.

## Bottom Line

- **Video saliency inside the encoder:** yes, with meaningful prior art nearby
- **Text saliency inside the text encoder:** yes, and likely the most underexplored multimodal angle
- **Text-to-video alignment as pruning signal:** yes, but already fairly crowded

The most promising next-step idea is a **joint saliency framework** that combines encoder-internal video saliency, encoder-internal text saliency, and cross-modal agreement into one pruning policy.
