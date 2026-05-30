# Info Measurement in Encoders for Token Pruning — Research Chat
**Date:** 2026-05-03

---

## Question

> How to measure information within a vision encoder or audio encoder depending on the question query given? Would this be a good way to prune tokens? Find out if this is novel or has been explored, and where I can add to it in terms of my research on token compression for multimodal models.

---

## What the Idea Is

Measuring information **inside** the encoder (vision or audio) at intermediate layers, conditioned on the text query, to decide which tokens to prune *during or right after encoding* — not inside the LLM.

---

## What Already Exists (Not Novel)

| Method | Where it prunes | Query-guided? |
|---|---|---|
| FastV, ATP-LLaVA, SparseVLM | Inside LLM decoder layers | Partial (attention to text tokens) |
| VisionZip, FasterVLM | Post-encoder, pre-LLM | No — visual salience only |
| **ICLR 2026 Task-Related Token Compression** | Pre-LLM, trains FCN on first-layer LLM attn | Yes — gradient-weighted relevance |
| **OmniSIFT** (arXiv 2602.04804) | Post-encoder audio selection via visual anchors | Partially — visual-conditioned audio |
| **FastAV** (arXiv 2601.13143) | Post audio/visual encoding, LLM backbone | No query signal |
| SpeechPrune | LLM backbone, audio tokens | No |

---

## Where There IS a Gap (Novel Territory)

**The encoder's internal computation conditioned on query is almost untouched.** Specifically:

1. **Measuring information entropy / attention entropy at intermediate encoder layers guided by the query** — e.g., which ViT layers produce high-entropy attention *specific to the query concept*. Nobody feeds query signal into ViT/audio encoder mid-forward to guide pruning.

2. **Audio encoder query conditioning** is nearly completely unexplored. SpeechPrune and FastAV work downstream in the LLM. Nothing operates *inside the Whisper/Tmer encoder* with query awareness.

3. **Modality-asymmetric encoder-level pruning** — prune the audio encoder more aggressively for vision-heavy questions and vice versa, using the query to decide *per-modality* compression ratio, not just which tokens within a modality.

4. **Mutual information between query embedding and intermediate encoder features** as a token importance score — this is an information-theoretic framing that's been proposed conceptually (FlowMM touches cross-modal flow) but not implemented as a pruning signal inside the encoder.

---

## Where This Fits in OmniZip Research

OmniZip currently uses audio signals (training-free) to guide token compression for Qwen2.5-Omni. The gap to fill:

- **Add a query-conditioned information measurement stage inside the audio encoder** (e.g., cross-attention between query tokens and encoder hidden states at layer $k$, use the resulting relevance map to prune before the encoder's final output)
- This would make OmniZip *query-aware* rather than purely audio-signal-driven
- Contrast: OmniSIFT does query-conditioned audio selection but at the post-encoder stage using visual anchors, not the text query directly — this approach would operate earlier and with the actual instruction

**Strongest novel claim**: *Training-free, query-conditioned token pruning inside the audio/vision encoder using attention-based mutual information estimates between the instruction query and intermediate encoder representations.*

---

## Must-Read Papers

- [ICLR 2026 Task-Related Token Compression](https://openreview.net/pdf/0025ce0e767142af1acd8cf5974dcb96e2bc83f7.pdf) — closest related work
- [OmniSIFT: Modality-Asymmetric Token Compression](https://arxiv.org/html/2602.04804) — most relevant to Qwen2.5-Omni context
- [FastAV: Audio-Visual Token Pruning](https://arxiv.org/html/2601.13143v1) — baseline for AV-LLM pruning
- [FlowMM: Cross-Modal Information Flow](https://arxiv.org/pdf/2511.05534) — analytical framing for cross-modal flow
- [Token Pruning in MLLMs: Are We Solving the Right Problem?](https://arxiv.org/html/2502.11501v1) — critical survey for positioning novelty
- [MADTP CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Cao_MADTP_Multimodal_Alignment-Guided_Dynamic_Token_Pruning_for_Accelerating_Vision-Language_Transformer_CVPR_2024_paper.pdf)
- [VisionTrim ICLR 2026](https://github.com/hanxunyu/VisionTrim)
- [Awesome Token Reduction (survey repo)](https://github.com/ZLKong/Awesome-Collection-Token-Reduction)
