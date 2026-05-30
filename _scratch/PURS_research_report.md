# PURS Research Accomplishment Report

**To:** Chengming; Armaan Amatya
**From:** Armaan Amatya
**Date:** 2026-05-18
**Subject:** Research accomplishments supported by the PURS program

---

## Research area

Through PURS, I explored the broader field of token compression for omni-modal foundation models, along with several adjacent research areas including Video LLMs, audio-language models, and long-context multimodal systems. This involved extensive literature review, tracing the historical development of the field, analyzing trends across recent papers, and studying the contributions of major labs and authors shaping this space.

## Literature review and technical exploration

I conducted a structured survey of efficient multimodal inference, focusing on:

- **Token pruning and merging** in vision-language models (e.g., FastV, ToMe, DivPrune, ReDiPrune) and their extensions to video and omni-modal settings.
- **KV-cache compression** approaches (FastKV, MixKV, AngelSlim) and their interaction with multimodal prefill.
- **Audio-guided and cross-modal compression** as introduced by OmniZip for Qwen2.5-Omni — currently the closest prior art to my own direction.
- **Architectural design choices** in the Qwen2.5/Qwen3 Thinker–Talker family, Video-LLaMA, and related omni-modal stacks.

I synthesized these findings into working notes and a mechanistic layer-level analysis of the Qwen2.5/Qwen3 Thinker–Talker architecture, which now grounds my experimental direction.

## Gap identified

OmniZip and most prior work treat **audio as the guide** for compressing video tokens. The inverse problem — using **video saliency to guide audio token compression** — is largely underexplored, despite audio often carrying significant redundancy in long-form video. This motivates **VideoZip**, my current research direction: a training-free, mechanistically grounded compression scheme for the audio stream of Qwen2.5-Omni, conditioned on video-derived saliency signals.

## Current contributions and progress

- **Mechanistic finding:** Identified that **Thinker layer 6 encodes question-invariant audio saliency**, allowing per-video caching of the importance signal. On VideoMME this matches OmniZip's accuracy (114/118) while delivering a **1.61× prefill speedup** — a concrete, reproducible result.
- **Prototype:** Implemented training-free pruning hooks against the Qwen2.5-Omni stack, evaluated on VideoMME via `lmms-eval`.
- **Working theory:** Drafted a layer-level account of *why* L6 (rather than later attention-heavy layers) carries the cleanest saliency signal — a story that the numbers alone don't tell.
- **Next steps:** Extending to hybrid audio+video scoring, adaptive ρ per modality, and two-stage pruning; targeting a workshop-scale write-up.

## Skills and research growth

PURS taught me how to systematically read and dissect papers, evaluate methodologies and benchmarks, identify open problems, and understand how research directions evolve. It strengthened my academic writing, technical communication, and critical-thinking skills, and gave me hands-on experience reproducing baselines, instrumenting large multimodal models, and interpreting layer-level behavior.

## Long-term impact

Most importantly, PURS crystallized my decision to pursue a PhD focused on **efficient ML systems and multimodal machine learning** — specifically the intersection of mechanistic interpretability and inference-time compression for foundation models. The combination of a concrete experimental result (the L6 finding) and a clear forward research agenda (VideoZip) is a direct product of this program.

Thank you for the support and mentorship throughout.

— Armaan
