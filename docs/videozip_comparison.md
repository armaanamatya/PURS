# 3-Way Comparison: OmniSIFT vs OmniZip vs VideoZip

*Generated via NotebookLM chat with all 3 sources targeted; saved as note (21006f4b-4c69-45ee-b4fa-6d871cd0f1f4).*

### 1. Side-by-Side Comparison

| Axis | OmniSIFT | OmniZip | VideoZip |
| :--- | :--- | :--- | :--- |
| **(a) Guidance direction** | **Video → Audio:** Prunes spatial/temporal video redundancy first, then uses retained visual tokens to guide audio token selection [1-3]. | **Audio → Video:** Identifies salient audio tokens first, then uses audio retention rates to dynamically guide video pruning [4-6]. | **Video → Audio:** Video tokens guide the audio token compression and ratio allocation [6, 7]. |
| **(b) Training requirements** | **Learned:** Requires training. Introduces 4.85M parameters (via a lightweight multi-head cross-attention layer and MLP) optimized end-to-end via a straight-through estimator [1, 8-11]. | **Training-free:** Acts as an inference-time token compressor without additional learned parameters [4, 12, 13]. | **Training-free:** Modifies the OmniZip algorithm natively without requiring parameter training [6, 7]. |
| **(c) Cross-modal anchoring** | **Uni-directional:** Pruned video tokens act as keys and values to interact with audio queries, anchoring the selection of audio tokens [14]. | **Uni-directional:** Employs cross-modal similarity (between audio and video tokens) to select and merge secondary non-salient audio tokens [6, 15, 16]. | **Bi-directional:** Video determines audio compression rates, while audio also guides the selection of video anchors via an audio-anchored ISTM module (using `dpcknn`) [6, 17]. |
| **(d) Where pruning is inserted** | **Before LLM:** Operates post-encoder. Token chunks from vision and audio encoders are processed by STVP and VGAS modules before entering the LLM backbone [3, 18-20]. | **Before LLM:** Operates post-encoder. Segments multimodal streams into time windows, selects and restructures tokens, then feeds the condensed sequence to the LLM [13, 21]. | **Before LLM:** Operates post-encoder. Intercepts the forward pass before the LLM (modifying the OmniZip token selection pipeline) [22, 23]. |
| **(e) Reported efficiency gains** | **>40% total inference time reduction** and >4.6 GB memory reduction on Qwen2.5-Omni-7B; surpasses full-token baseline accuracy even at an aggressive **25% token retention** [8, 24, 25]. | **2.51x - 3.42x prefill speedup** and 10 GB memory reduction on Qwen2.5-Omni-7B; maintains ~97% of original accuracy at **35% retention** [4, 26, 27]. | Gains a **free 1.6x prefill speedup** strictly from its L6-cached video saliency implementation, stacked on top of token reduction gains [28, 29]. |

### 2. Unique-to-VideoZip

VideoZip is proposed as a direct inversion of the OmniZip architecture and introduces three major architectural claims that neither OmniSIFT nor OmniZip can replicate:
*   **Bidirectional Cross-Modal Anchoring:** While VideoZip predominantly uses video to guide audio token selection, it integrates an "audio-anchored ISTM" process. By utilizing a modified density-peak clustering (`dpcknn`) guided by audio, it ensures that audio diversity acts as a secondary signal to anchor video token selection [6, 17].
*   **L6-Cached Video Saliency:** VideoZip extracts a question-invariant video saliency map directly from the early layer (L6) hidden states. Because this cached signal is reused across multiple queries for the same video, both modalities are pruned with just one cached forward pass. This bypasses live per-query attention computations, netting an immediate 1.6x prefill speedup [28, 29].
*   **Adaptive Guide-Mode Selection:** VideoZip introduces a system to calculate attention entropy, allowing the model to adaptively select whether the video or the audio should act as the guide modality dynamically based on the input [30, 31]. 

### 3. Where Each Wins

Based on the architectural assumptions and proposed hypotheses, each method favors specific task profiles:
*   **OmniSIFT Wins:** Favored in complex temporal or cross-modal reasoning tasks that operate under extreme token constraints (e.g., a 25% token budget). Because its parameters are optimized end-to-end, it maintains high resilience for identifying precise event sequences and audio-visual alignments where training-free methods degrade [8, 32, 33].
*   **OmniZip Wins:** Favored when audio acts as the primary information carrier or carries unique information absent from the visual feed (e.g., off-screen speech, sound events), or when the video has massive temporal redundancy (e.g., static talking heads). It excels in speech QA or benchmarks like AIR-Bench [4, 34].
*   **VideoZip Wins:** Favored when the video feed holds the primary query targets and the audio has high temporal redundancy (e.g., ambient noise, background music). It is specifically optimal for scenarios where audio is merely confirmatory—such as identifying a speaker on-screen and subsequently pruning all non-speaker audio [34]. Tasks like ActivityNet-QA and VideoMME heavily favor this profile [34].
