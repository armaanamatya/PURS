# OmniZip: Audio-Guided Dynamic Token Compression for Omnimodal LLMs

## Executive Summary

The emergence of Omnimodal Large Language Models (OmniLLMs) has enabled unified audio-video understanding, yet these models face significant computational bottlenecks due to the massive volume of multimodal tokens. Processing these sequences entails quadratic attention complexity, hindering real-time deployment. **OmniZip** is introduced as a training-free, audio-guided framework designed to optimize multimodal token representation and accelerate inference. 

Based on token attention analyses, research reveals that audio tokens consistently dominate attention heatmaps, while video tokens exhibit substantial redundancy. OmniZip leverages this "listen-to-prune" paradigm, utilizing salient audio tokens to dynamically guide the pruning of video tokens within fixed-length time windows. Empirical results demonstrate that OmniZip achieves up to a **3.42× inference speedup** and a **10G reduction in GPU memory consumption** while maintaining approximately **99.1% of baseline accuracy**. Crucially, the method is compatible with FlashAttention and requires no additional training or parameter tuning.

---

## Technical Analysis of Multimodal Redundancy

### The Dominance of Audio Modality
Attention distribution analysis across OmniLLM layers reveals regularly recurring vertical bands aligned with audio-token positions. This indicates that:
*   **Attention Priority:** Audio tokens receive consistently higher attention scores than video tokens.
*   **Layer-wise Decay:** Attention to both modalities decreases with layer depth, suggesting that deeper layers allocate less attention to raw tokens.
*   **Locality:** Mutual attention between audio and video tokens is most pronounced within the same time window, decaying rapidly across windows. This motivates a window-granularity approach to compression.

### The "Listen-to-Prune" Paradigm
OmniZip operates on the principle that audio retention serves as a proxy for information density and event-boundary priors. By identifying salient audio tokens, the system can determine which time windows are "information-dense" (requiring conservative video pruning) and which are "information-sparse" (allowing aggressive video pruning).

---

## Comparative Analysis of Compression Approaches

The following table compares the OmniZip framework against representative prior methods adapted for omnimodal settings (FastV, DyCoke, and VisionZip) based on the specified technical axes.

| Axis | OmniZip | FastV | DyCoke (V&A) | VisionZip |
| :--- | :--- | :--- | :--- | :--- |
| **1. Guidance Direction** | **Audio-Guided:** Audio saliency drives the dynamic pruning rate of video tokens. | **Visual-Centric:** Uses LLM attention scores (typically visual) to prune tokens. | **Temporal-Centric:** Focuses on reducing temporal redundancy. | **Global Selection:** Performs independent global selection for each modality. |
| **2. Training Requirements** | **Training-free:** Post-processing technique; no additional parameters. | **Training-free:** Plug-and-play inference acceleration. | **Learned Components:** Utilizes a specific TTM module for token processing. | **Training-free:** Global selection strategy. |
| **3. Cross-modal Anchoring** | **Bi-directional/Similarity-based:** Uses cross-modal similarity to merge audio anchors with video-related tokens. | **Uni-directional:** Prunes based on internal LLM attention scores. | **Uni-directional:** Neglects spatial redundancy and cross-modal alignment. | **Independent:** Extracts tokens independently, ignoring semantic alignment. |
| **4. Pruning Insertion** | **Before LLM:** Operates window-by-window after the projectors but before the LLM backbone. | **Inside LLM Layers:** Pruning occurs at the $L$-th layer (e.g., Layer 5) during prefill. | **Before LLM:** First-stage TTM module processes tokens before input. | **Encoder-Output:** Performs global selection before the LLM. |
| **5. Reported Efficiency Gains** | **3.42× prefill speedup;** 10G memory reduction; ~99.1% accuracy retention. | **1.16× speedup (3B);** Incurs OOM on 7B models due to attention matrix needs. | **1.58× prefill speedup;** 4G memory reduction; suboptimal omnimodal accuracy. | **Suboptimal:** Disrupts temporal structure; prone to OOM during matrix extraction. |

---

## The OmniZip Framework Architecture

OmniZip processes tokens through three distinct stages to ensure cross-modal semantic and temporal alignment.

### 1. Audio Token Selection and Anchor Consolidation
*   **Saliency Filtering:** Audio tokens are filtered based on the mean attention scores from the last layer of the audio encoder.
*   **Anchor Sampling:** To maintain context coverage, anchors are uniformly sampled from non-salient tokens.
*   **Similarity Merging:** The system evaluates cross-modal similarity ($S_{cross} = \hat{H}_a\hat{H}_v^\top$) to select and merge non-salient audio tokens most related to the paired video segment into the anchors.

### 2. Audio-Guided Video Compression
Using the audio retention rate ($S_a(i)$) for each time group, OmniZip calculates a dynamic pruning ratio ($\rho'_v(i)$). 
*   **High Saliency Windows:** Pruned conservatively to preserve key event boundaries.
*   **Low Saliency Windows:** Pruned aggressively to eliminate redundancy.

### 3. Interleaved Spatio-Temporal Compression (ISTC)
The ISTC module addresses video redundancy through a dual-strategy approach:
*   **Temporal Redundancy:** Computes cosine similarity between tokens in adjacent frames to prune highly similar tokens.
*   **Spatial Redundancy:** Employs **Density-Peak Clustering with K-Nearest Neighbors (DPC-KNN)**. It calculates local density ($\rho_i$) and distance ($\delta_i$) to retain salient video tokens while discarding spatially redundant ones.

---

## Performance Benchmarks

Inference efficiency was evaluated using **Qwen2.5-Omni (7B and 3B)** models on an NVIDIA A6000 GPU.

### Efficiency Comparison (WorldSense Benchmark)
| Method | GPU Memory Reduction | Prefilling Speedup | Accuracy (Acc.) |
| :--- | :--- | :--- | :--- |
| **Full Tokens (Baseline)** | 35G | 1.00× (291ms) | 46.8 |
| **FastV** | OOM | N/A | N/A |
| **DyCoke (V&A)** | 31G | 1.58× (184ms) | 44.6 |
| **OmniZip (45%)** | 28G | 2.51× (116ms) | **45.9** |
| **OmniZip (35%)** | **25G** | **3.42× (85ms)** | 45.3 |

### Accuracy across Benchmarks
OmniZip demonstrates superior robustness compared to random pruning and single-modal methods:
*   **AVUT (Audio-centric):** Higher audio merging ($G=15$) yields the best results.
*   **VideoMME:** Speedup effects are most pronounced here ($3.8\times$) due to longer video sequences.
*   **ShortVid-Bench:** Achieves $2.7\times$ speedup while maintaining high accuracy.

---

## Important Quotes and Contextual Insights

> "Audio tokens are consistently assigned greater attention than video tokens across layers, whereas large regions of video tokens exhibit significantly lower attention scores, suggesting substantial redundancy and the dominant role of audio tokens in the inference process."

**Context:** This observation serves as the foundational justification for the "listen-to-prune" strategy, identifying audio as the more information-dense modality in omnimodal settings.

> "Our method does not require accessing attention-score matrices inside the LLM, enabling compatibility with FlashAttention without incurring additional compute or memory overhead."

**Context:** Highlighting a critical advantage over methods like FastV and VisionZip, which often trigger Out-of-Memory (OOM) errors because they require the explicit materialization of large attention matrices.

> "Excessive pruning of either audio or video significantly degrades model performance... identifying an optimal pruning ratio is crucial for maximizing compression effectiveness."

**Context:** From the sensitivity analysis, suggesting that the audio pruning rate should generally remain lower than the video pruning rate to preserve performance.

---

## Actionable Insights

*   **Dynamic Task Adaptation:** For audio-centric tasks (like those in the AVUT benchmark), increase the number of tokens merged by each audio anchor ($G=15$). For balanced audio-video tasks (like WorldSense), a lower $G=3$ is recommended to avoid introducing noise.
*   **Parameter Balancing:** To achieve the best trade-off between speed and accuracy, set the audio pruning rate ($\rho_a$) lower than the video pruning rate ($\rho_v$). Experimental data suggests an optimal balance near $\rho_a=0.3$ and $\rho_v=0.6$ for a 45% retention ratio.
*   **Deployment Strategy:** OmniZip is recommended for practical OmniLLM deployment where memory is a constraint. It can reduce the GPU footprint by up to 10G on an A6000 48G GPU, enabling larger models to run on hardware that would otherwise trigger OOM errors.
*   **Scaling Benefits:** The speedup and memory benefits of OmniZip become more pronounced as model size increases (e.g., from 3B to 7B) and as video sequence length increases (e.g., VideoMME).