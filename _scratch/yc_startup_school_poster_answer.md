# YC Startup School 2026 — Poster Session Application

**Interested in presenting?** Yes

---

## What research would you be interested in presenting? (short version — ~180 words)

**VideoZip: Training-free Compression for Omnimodal Video+Audio LLMs**

Modern omnimodal LLMs (e.g., Qwen2.5-Omni) burn enormous compute processing long videos because every audio and video token flows through the model. Prior work (OmniZip) uses audio to guide which video tokens to drop. I'm exploring the inverse — using video saliency to compress the *audio* stream, which is often the more redundant modality in long-form video.

By probing the model layer-by-layer, I found that **Thinker layer 6 encodes a question-invariant audio saliency signal** — meaning we can compute it once and cache it per video, reusing it across any number of downstream questions. On VideoMME this matches OmniZip's accuracy (114/118) while delivering a **1.61× prefill speedup, with no training required**.

The broader bet: mechanistic interpretability isn't just academic — it tells you *which intermediate signals are reusable*, which translates directly into lower inference cost. The poster would walk through the layer-level analysis, the VideoZip prototype, and what it implies for anyone running video+audio inference at scale.

---

## Tighter alternative (~90 words, if there's a character cap)

I work on training-free compression for omnimodal LLMs (Qwen2.5-Omni). Existing work uses audio to guide *video* token pruning; I'm flipping it — using video saliency to compress the *audio* stream. By probing the model layer-by-layer, I found that Thinker layer 6 encodes a question-invariant audio saliency signal that can be cached per video and reused across questions. On VideoMME this matches the state-of-the-art (114/118) with a 1.61× prefill speedup, training-free. The poster would cover the mechanistic finding and its implications for cheap video+audio inference.
