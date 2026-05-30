# VideoZip docs (pointers, not duplicates)

These documents are the authoritative source. They live under `docs/` at the project root
and are NOT mirrored here to avoid drift.

| Doc | Purpose |
|---|---|
| [`docs/videozip_plan.md`](../../docs/videozip_plan.md) | Full functional spec — sections §3.1–§11 |
| [`docs/videozip_comparison.md`](../../docs/videozip_comparison.md) | NotebookLM 3-way comparison: OmniSIFT vs OmniZip vs VideoZip |
| [`docs/videozip_briefing.md`](../../docs/videozip_briefing.md) | NotebookLM briefing doc (auto-generated, OmniZip-focused — kept for archive) |
| [`docs/l6cache.md`](../../docs/l6cache.md) | L6 cache experimental record (audio-side; basis for video-side §3.6) |
| [`docs/method_l6_omnizip.md`](../../docs/method_l6_omnizip.md) | L6 method writeup |
| [`docs/findings.md`](../../docs/findings.md) | Cross-experiment findings record |
| [`docs/futureplanning.md`](../../docs/futureplanning.md) | Strategic directions; VideoZip extends Direction #4 |
| [`docs/PROJECT_BRIEFING.md`](../../docs/PROJECT_BRIEFING.md) | One-page project context |

## Section index inside `videozip_plan.md`

| Section | Topic |
|---|---|
| §1 | Literature gap |
| §2 | OmniZip baseline pipeline |
| §3 | VideoZip pipeline (full role swap) |
| §3.6 | L6-cached video saliency rationale |
| §4a | `omnizip_video_saliency()` — live attention |
| §4a-L6 | `omnizip_video_saliency_l6()` — cached |
| §4a-Sim | `omnizip_video_saliency_simonly()` — frozen cross-modal |
| §4b | `omnizip_audio_compress()` — per-group |
| §4c | `omnizip_istm_audio_anchored()` |
| §4d | `omnizip_videozip()` orchestrator |
| §5 | Config + CLI flags |
| §6 | Modeling dispatch |
| §7 | Ablations A–E |
| §10 | Implementation order |
| §11 | Risks |
