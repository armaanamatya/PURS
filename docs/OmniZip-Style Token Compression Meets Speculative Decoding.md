# OmniZip-Style Token Compression Meets Speculative Decoding

## Overview

This document explains how attention-based multimodal token compression (like OmniZip) can be combined with speculative decoding to speed up large multimodal models while keeping accuracy high.[^1][^2][^3]
It first covers the basic ideas, then the gap in current work, and finally a concrete proposal with simple math that shows what changes and why it should help.[^4][^5][^6]

## Background

### Multimodal LLMs and token explosion

Large multimodal models (LMMs) like Qwen2.5-Omni, LLaVA, and Qwen2-VL take images, video, audio, and text and turn them into long sequences of tokens before feeding them to a language model.[^2][^4]
When the input is a long video with audio, the number of tokens can be huge, so inference becomes slow and memory-hungry.

### Attention-based token compression (OmniZip)

OmniZip is a training-free method that speeds up omnimodal LLMs by compressing audio and video tokens before they go into the main model.[^1]
It uses encoder attention and audio signals to decide which video tokens in each short time window are important, then drops the rest, reaching about 3× prefill speedup and notable memory savings on Qwen2.5-Omni with almost no loss in accuracy.[^1]

In simple terms:

- Audio and video are split into time windows.  
- Audio tokens in a window act as anchors.  
- Attention and similarity scores between audio and video say which video tokens matter for that window.  
- Only the top tokens per window are kept; the rest are pruned.

### Speculative decoding

Speculative decoding speeds up autoregressive decoding by using a small **draft** model and a larger **target** model.[^5][^2]
The draft model proposes several next tokens at once, and the target model then checks these tokens in parallel and accepts the longest prefix that matches its own distribution.[^7][^5]

Key concepts:

- Let \(\alpha\) be the **acceptance rate**: the average fraction of draft tokens that the target accepts.  
- Let \(L\) be the sequence length, and let \(K\) be how many tokens the draft proposes per step.  
- The more tokens are accepted per verification, the fewer target forward passes are needed.

In text-only LLMs, speculative decoding often gives 2×–3× speedups with little quality loss.[^8][^7]
For multimodal LMMs, recent work shows that naive speculative decoding is less effective because multimodal inputs are large and visual tokens are redundant, but acceptance can still be improved with multimodal-aware design.[^9][^2]

## Existing combinations of compression and speculative decoding

Several recent papers start to combine speculative decoding with multimodal token compression, mostly on images and videos.

### SpecVLM: verifier-guided staged video token pruning

SpecVLM ("Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Staged Video Token Pruning") focuses on video LLMs.[^3][^6]
It observes that speculative decoding is surprisingly robust to pruning video tokens and proposes a two-stage, attention-guided pruning process:

- Use attention from the **verifier** (target model) to assign importance scores to video tokens.  
- First stage: prune tokens by attention thresholds.  
- Second stage: further prune spatially uniform areas to remove redundancy in low-attention regions.[^6][^3]

SpecVLM shows that adding this pruning on top of speculative decoding gives extra speedups on video captioning benchmarks while keeping quality.[^3][^6]

### FLASH / SpecFLASH: latent-aware visual compression in speculative decoding

FLASH and its SpecFLASH variant propose latent-aware semi-autoregressive speculative decoding tailored for multimodal tasks.[^10][^11][^5]
They exploit visual redundancy and object co-occurrence in images and videos to compress visual tokens inside the speculative drafter before decoding.

Important points:

- Visual tokens are compressed using lightweight latent modules that group or merge similar features.  
- A semi-autoregressive head predicts several tokens per step on top of the compressed sequence.  
- This gives around 2.5× speedups on video captioning and visual instruction tasks, with minimal accuracy loss.[^10][^5]

While the compression is not explicitly defined as "attention-based" like OmniZip, the effect is similar: many visual tokens are removed or merged before speculative decoding.

### DREAM: feature injection and visual token compression

DREAM (NeurIPS 2025) is a speculative decoding framework for multimodal models that includes cross-attention feature injection and **visual token compression**.[^12]
It fuses target features into the draft model and adaptively compresses visual tokens to reduce the cost of drafting, reaching up to about 3.6× speedups in some VLMs while keeping good acceptance.

### Multimodal Speculative Decoding (MSD)

The MSD work ("Speculative Decoding Reimagined for Multimodal LLMs") shows that simply applying text-only speculative decoding ideas to multimodal models does not work well and proposes a **modality-aware** speculative framework.[^13]
It separates the treatment of text and visual tokens during drafting and verification, but does not provide an audio-guided attention-based compression like OmniZip.

### AMD ROCm blog and AASD: practical and KV-level compression

AMD’s ROCm blog on multimodal speculative decoding shows practical engineering patterns such as compressing visual tokens **only in the draft model**, leaving the target model unchanged.[^2]
This reduces the cost per draft step, and overall speed often improves even if the acceptance rate drops slightly.[^2]

AASD ("Accelerate Inference by Aligning Speculative Decoding in Multimodal LLMs") proposes a KV Projector to compress the multimodal KV cache together with target–draft alignment tricks, allowing faster speculative decoding without obvious accuracy loss.[^14]

### Surveys on compression and multimodal efficiency

Recent surveys give a broad view of token compression and multimodal efficiency:

- "When Tokens Talk Too Much" / "A Survey of Multimodal Long-Context Token Compression" categorize compression into transformation-, similarity-, attention-, and query-based methods.[^15][^16][^4]
- "Efficient Inference of Large Vision Language Models" highlights token compression, memory management, and multimodal speculative decoding as key directions for efficient VLMs.[^17][^18]

These surveys note that attention-based decoder-side pruning (such as ranking tokens by attention and keeping top-k) is a natural building block, but they do not fully combine audio-guided compression like OmniZip with speculative decoding.

## Gap in current work

From the above, there are two main threads:

- **Attention-based or latent-based visual token compression** for multimodal LMMs (OmniZip, visual pruning, FlexSelect, LongVU, etc.).[^4][^15][^1]
- **Multimodal speculative decoding** methods that prune tokens or compress features, mostly for images/videos, to speed up the draft stage (SpecVLM, FLASH/SpecFLASH, DREAM, MSD, AMD ROCm, AASD).[^5][^6][^13][^12][^3][^2]

However, there is still a clear gap:

- OmniZip gives **audio-guided, window-wise attention-based compression** for video tokens in omni LLMs, but it is used only as a prefill-time module, not coupled to speculative decoding.[^1]
- Existing multimodal speculative methods prune mainly based on visual patterns or latent features and do not use **audio as a guiding signal** for compression or for the draft–target split.[^6][^5][^3]
- Most works either compress both draft and target in similar ways or focus only on images/video; none explicitly design a **two-level compression scheme (strong for draft, mild for target) driven by cross-modal attention including audio** in omni-style models.

This leaves a space for a method that:

- Uses audio-guided attention and cross-modal similarity (like OmniZip) to compute importance scores per token.  
- Builds two different compression levels: one for the **draft model** and one for the **target model**.  
- Integrates these masks directly into a speculative decoding loop for omni LLMs (video+audio+text), while staying training-free.

## Proposed method: audio-guided attention-based speculative compression

### High-level idea

The proposed method is a **training-free, audio-guided attention-based token compression scheme for speculative decoding in omni LLMs**.  
It keeps more tokens for the target (verifier) and fewer tokens for the draft, both decided from the same cross-modal attention signals.

The goals are:

- Reduce the cost per draft step by giving the draft model a much shorter token sequence.  
- Keep the target’s context rich enough to maintain high acceptance and accuracy.  
- Exploit audio as a strong signal for what parts of the video matter in each time window, extending OmniZip into the speculative decoding stage.

### Notation and setup

Consider a multimodal input segmented into \(T\) time windows. For each window \(t\):

- Let \(A_t = \{a_{t,1}, \dots, a_{t,N_a}\}\) be audio tokens.  
- Let \(V_t = \{v_{t,1}, \dots, v_{t,N_v}\}\) be video tokens.  
- Let \(X_t\) be any text tokens aligned to that window (optional).

The full multimodal token sequence is the concatenation over time and modalities:

\[
S = [X_1, A_1, V_1, X_2, A_2, V_2, \dots, X_T, A_T, V_T]. \quad [^1]
\]

Let \(M_d\) be the draft model and \(M_v\) be the target (verifier) model.

### Step 1: compute cross-modal attention scores

In an OmniZip-style prefill, run the encoders and a few initial layers of the LLM (or a small probe) to get cross-modal attention matrices within each window.[^1]

For example, consider attention from audio to video tokens in a given layer and head:

\[
\text{Att}^{(l,h)}_{t}(a_{t,i} \rightarrow v_{t,j}) \in [0, 1]. \quad [^2]
\]

Define an audio-guided importance score for each video token as the average attention it receives from audio tokens across heads and layers:

\[
w_{t,j}^{\text{audio}} = \frac{1}{|\mathcal{L}||\mathcal{H}| N_a} \sum_{l \in \mathcal{L}} \sum_{h \in \mathcal{H}} \sum_{i=1}^{N_a} \text{Att}^{(l,h)}_{t}(a_{t,i} \rightarrow v_{t,j}). \quad [^19]
\]

Optionally, include attention from text to video or video to video to refine this score. A general importance score can be:

\[
w_{t,j} = \alpha_1 w_{t,j}^{\text{audio}} + \alpha_2 w_{t,j}^{\text{text}} + \alpha_3 w_{t,j}^{\text{video}}, \quad [^20]
\]

where \(\alpha_k\) are simple scalar weights (for example, \(\alpha_1 = 1, \alpha_2 = \alpha_3 = 0\) for pure audio guidance at first).

### Step 2: build two compression masks (draft vs target)

For each window \(t\), sort video tokens by importance score \(w_{t,j}\).

Define two thresholds or top-k counts:

- \(k_t^{(v)}\): the number of video tokens kept for the verifier (target).  
- \(k_t^{(d)}\): the number of video tokens kept for the drafter, with \(k_t^{(d)} \leq k_t^{(v)}\).

Build binary masks:

\[
\begin{aligned}
\mathbb{I}^{(v)}_{t,j} &= 1 \text{ if } v_{t,j} \text{ is among the top } k_t^{(v)} \text{ by } w_{t,j}, \\
\mathbb{I}^{(d)}_{t,j} &= 1 \text{ if } v_{t,j} \text{ is among the top } k_t^{(d)} \text{ by } w_{t,j}.
\end{aligned} \quad [^21]
\]

The compressed sequences for verifier and draft are:

\[
S^{(v)} = \text{compress}(S; \mathbb{I}^{(v)}), \quad S^{(d)} = \text{compress}(S; \mathbb{I}^{(d)}), \quad [^7]
\]

where \(\text{compress}\) removes all video tokens with mask value 0, and may also apply OmniZip’s interleaved spatio-temporal compression inside the kept tokens.[^1]

### Step 3: speculative decoding loop with dual compression

Use a standard speculative decoding loop, but give different contexts to draft and target:

1. Initialize the sequence with the compressed token sets: the draft sees \(S^{(d)}\), the verifier sees \(S^{(v)}\).  
2. At each decoding step:
   - The draft model proposes up to \(K\) next tokens \(y_{t+1:t+K}^{(d)}\) conditioned on \(S^{(d)}\) and accepted history.  
   - The verifier model re-computes hidden states and logits on top of \(S^{(v)}\) and the proposed tokens, then accepts the longest matching prefix.[^7][^5]
3. If the acceptance rate \(\alpha\) drops too low, adjust thresholds \(k_t^{(d)}\) upward to give the draft more context.

This loop is compatible with frameworks like FLASH, SpecFLASH, and MSD, because those frameworks already support a two-model speculative setup with multimodal inputs.[^11][^13][^5]

### Step 4: dynamic adjustment using verifier attention

To get closer to SpecVLM, add dynamic adjustment of masks based on verifier attention during decoding:[^3][^6]

- Periodically (for example every \(m\) decoding steps), examine recent attention from new text tokens to visual tokens in the verifier.  
- Recompute or update importance scores \(w_{t,j}\) with these new attention patterns.  
- Gradually increase pruning for tokens that stay low-importance, especially in the draft mask \(\mathbb{I}^{(d)}\).

In other words, early decoding uses mild compression so that attention maps are reliable, and later decoding uses stronger pruning as the model concentrates on fewer relevant regions.

## Simple complexity and speed analysis

Let:

- \(C(S)\) be the cost of one forward pass of the LMM with context token sequence \(S\).  
- \(C_d = C(S^{(d)})\) be the cost per draft model pass.  
- \(C_v = C(S^{(v)})\) be the cost per verifier pass.

Without speculative decoding, using OmniZip-style compression once, the per-step cost is basically \(C_v\).[^1]

With speculative decoding and dual compression:

- Each iteration, the draft proposes \(K\) tokens, and the verifier checks them.  
- On average, \(\alpha K\) tokens are accepted per verifier pass, where \(0 < \alpha \leq 1\).[^7][^5]

The expected number of target (verifier) passes needed to generate \(L\) tokens is roughly:

\[
N_v \approx \frac{L}{\alpha K}. \quad [^22]
\]

If the draft model is much smaller and sees a shorter sequence, \(C_d\) is much smaller than \(C_v\). The total cost is approximately:

\[
\text{Cost}_{\text{total}} \approx N_v (C_v + C_d). \quad [^10]
\]

Compared to a baseline that runs only the target with the same compressed context, whose cost is \(L C_v\), the speedup factor is:

\[
S \approx \frac{L C_v}{N_v (C_v + C_d)} = \frac{\alpha K C_v}{C_v + C_d}. \quad [^8]
\]

The method aims to:

- Reduce \(C_d\) by strong compression in the draft path (small \(|S^{(d)}|\)).  
- Keep \(C_v\) moderate by mild compression, and maintain a good acceptance rate \(\alpha\) using richer context.  
- Choose \(K\) and the masks to maximize \(S\) while meeting accuracy and latency targets.

Because attention-based compression reduces the number of visual tokens roughly linearly, \(C_d\) and \(C_v\) shrink with the compressed token counts (up to overheads in the rest of the model).[^17][^4]

## Where this method fits in the literature

Compared to OmniZip:

- OmniZip compresses tokens before feeding them into the LMM and focuses on prefill speed and memory reductions for omni LLMs.[^1]
- The proposed method **extends OmniZip’s audio-guided attention and window-wise compression into the speculative decoding loop**, adding a second compression level for the draft model.

Compared to SpecVLM and related video-focused methods:

- SpecVLM uses verifier attention to rank and prune video tokens for speculative decoding but does not use audio guidance in omni settings.[^6][^3]
- The proposed method combines **audio-guided attention (OmniZip) and verifier attention (SpecVLM-style)**, using them together to control draft and verifier masks in omni models.

Compared to FLASH, SpecFLASH, DREAM, and MSD:

- Those works show that visual compression inside speculative decoding is powerful, but they mainly rely on latent similarity and do not define an explicit, training-free audio-guided attention rule.[^11][^13][^12][^5]
- The proposed method provides such a rule and a simple math formulation that can be attached to any of these speculative frameworks.

Compared to AMD ROCm’s practical patterns and AASD:

- AMD’s blog shows drafter-only visual compression and reports good empirical gains, but does not provide a cross-modal attention-driven design for omni inputs.[^2]
- AASD compresses KV caches and aligns draft and target attention, but again does not exploit audio tokens as explicit anchors for compression.[^14]

## Possible experimental setup

A basic experimental plan for Qwen2.5-Omni or similar omni LLMs could be:

- **Baselines**:  
  - No compression, no speculative decoding.  
  - OmniZip-only compression (original method).  
  - Speculative decoding (text-only-like, no extra compression in the drafter).  
  - SpecVLM-style video pruning + speculative decoding on video LLMs, if available.[^3][^2][^1]

- **Proposed variants**:  
  - Audio-guided dual masks (draft vs verifier) with fixed \(k_t^{(d)}\) and \(k_t^{(v)}\).  
  - Dynamic masks updated from verifier attention as decoding proceeds.  
  - Different weightings of audio vs text vs video attention in the importance score.[^3][^1]

- **Metrics**:  
  - Task accuracy (e.g., QA accuracy, captioning metrics) on multimodal benchmarks.  
  - End-to-end latency and throughput on GPUs, including edge devices.  
  - Memory usage and KV cache size.  
  - Acceptance rate \(\alpha\) and average draft length \(K\) for speculative decoding.

This would show whether the audio-guided attention-based speculative compression improves both speed and quality compared with existing methods.

## Summary of contributions

The key contributions of this proposal are:

- A simple, training-free **audio-guided attention-based importance score** for video tokens, extending OmniZip ideas into decoding.  
- A **dual-mask compression scheme** that keeps different sets of tokens for draft and verifier models in speculative decoding.  
- A concrete integration of these masks into a speculative decoding loop, with basic math showing how the speedup depends on compression and acceptance.  
- A clear position in the literature between OmniZip-style token compression and recent multimodal speculative decoding work, filling the gap of audio-aware, attention-driven compression for omni LLMs.

---

## References

1. [Let’s take a look at Qwen2.5 omni models.
We want to look at patient outcome and hospital efficiency.

Armaan Project:
OmniZip paper as baseline
Goal of project: to make a better token compression method for like in omnizip

Turn compression o...

.../Qwen/qwen25-omni)

Further, understand what exactly the compression algorithm is doing and how maybe to improve it to some extent for the datasets.
- give me some ideas to improve

what is the dataset in the paper like in  omnizip and qwen-2.5-omni](https://www.perplexity.ai/search/dc9bd3e9-ca69-4148-9416-65264d656cfd) - OmniZip compresses video and audio tokens by using audio to decide which video tokens to keep, opera...

2. [Beyond Text: Accelerating Multimodal AI Inference with Speculative ...](https://rocm.blogs.amd.com/software-tools-optimization/multimodal-spec-dec/README.html) - In this blog you will learn, step-by-step, how speculative decoding can help you unlock significant ...

3. [Enhancing Speculative Decoding of Video LLMs via Verifier-Guided ...](https://arxiv.org/html/2508.16201v1) - To achieve this, it performs a two-stage pruning process: Stage I selects highly informative tokens ...

4. [A Survey of Token Compression for Efficient Multimodal Large ...](https://arxiv.org/html/2507.20198v5)

5. [A Latent-Guided Semi-autoregressive Speculative Decoding ... - arXiv](https://arxiv.org/abs/2505.12728) - In this paper, we introduce SpecFLASH, a speculative decoding framework tailored to LMMs that explic...

6. [[PDF] Enhancing Speculative Decoding of Video LLMs via Verifier-Guided ...](https://aclanthology.org/2025.emnlp-main.366.pdf) - Numerous redundant tokens divert attention away from impor- tant ones, and thus, moderate token remo...

7. [Speculative decoding: cost-effective AI inferencing - IBM Research](https://research.ibm.com/blog/speculative-decoding) - Speculative decoding has emerged as a promising optimization technique for speeding up AI inferencin...

8. [Looking back at speculative decoding - Google Research](https://research.google/blog/looking-back-at-speculative-decoding/) - Speculative decoding has proven to be an effective technique for faster and cheaper inference from L...

9. [Speculative Decoding Reimagined for Multimodal Large Language ...](https://www.emergentmind.com/papers/2505.14260) - This paper introduces Multimodal Speculative Decoding (MSD) to accelerate Multimodal Large Language ...

10. [SpecFLASH: A Latent-Guided Semi-autoregressive Speculative ...](https://arxiv.org/html/2505.12728v3) - In this paper, we introduce SpecFLASH, a speculative decoding framework tailored to LMMs that explic...

11. [FLASH: Latent-Aware Semi-Autoregressive Speculative Decoding ...](https://ui.adsabs.harvard.edu/abs/arXiv:2505.12728) - ... token compression mechanism. Second, recognizing that visual objects ... FLASH: Latent-Aware Sem...

12. [NeurIPS Poster DREAM: Drafting with Refined Target Features and ...](https://neurips.cc/virtual/2025/loc/san-diego/poster/136216)

13. [Speculative Decoding Reimagined for Multimodal Large Language ...](https://arxiv.org/html/2505.14260v1) - This paper introduces Multimodal Speculative Decoding (MSD) to accelerate Multimodal Large Language ...

14. [Presentation](https://62dac.conference-program.com/presentation/?id=RESEARCH005&sess=sess104)

15. [A Survey of Multimodal Long-Context Token Compression ...](https://arxiv.org/html/2507.20198v3)

16. [When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression across Images, Videos, and Audios](https://www.emergentmind.com/papers/2507.20198) - Multimodal large language models (MLLMs) have made remarkable strides, largely driven by their abili...

17. [Efficient Inference of Large Vision Language Models - arXiv](https://arxiv.org/html/2603.27960v1) - ... token compression, memory management and serving, efficient ... multimodal speculative decoding ...

18. [Efficient Inference of Large Vision Language Models | alphaXiv](https://www.alphaxiv.org/overview/2603.27960v1) - Speculative Decoding. Speculative decoding uses a "draft ... A Survey of Token Compression for Effic...

19. [Speculative decoding | LLM Inference Handbook - BentoML](https://bentoml.com/llm/inference-optimization/speculative-decoding) - Speculative decoding is an inference-time optimization that speeds up LLM token generation without r...

20. [Speculative decoding for high-throughput long-context inference](https://www.together.ai/blog/speculative-decoding-for-high-throughput-long-context-inference) - We demonstrate that speculative decoding can improve throughput and latency by up to 2x on 8 A100s i...

21. [FLASH: Latent-Aware Semi-Autoregressive Speculative Decoding ...](https://openreview.net/forum?id=yyqbLqLhGl) - First, to address redundancy in visual tokens, we propose a lightweight latent-aware token compressi...

22. [Speculative Decoding and Beyond: An In-Depth Survey of Techniques](https://arxiv.org/html/2502.19732v4) - Self-speculative decoding approaches generate draft tokens by relying directly on a subset (Layer Sk...

