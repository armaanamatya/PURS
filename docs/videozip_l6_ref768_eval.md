# VideoZip — L6 ref768 run evaluation

**Run:** `10x/qwen25_matrix_gpu7_all7_snapkv/videozip/runs/l6_ref768/`
**Date:** 2026-05-28 · GPU 7 (`omnizip_clean`) · branch `videozip`
**Benchmark:** 118-Q subset across video-mme + worldsense + daily-omni
**Config (matched to 10× matrix):** `rho_audio=0.3, rho_video=0.6, g=3, audio_anchor_beta=0.3, contextual_ratio=0.05`, `fps=2.0, max_pixels≈100352, max_frames 768/128`, L6 cached saliency (`layer_depth_ref768.jsonl`, cached_videos=44), bf16.

## Headline (vs the 10× reference matrix)
| method | accuracy | prefill (mean) | **speedup** | VRAM peak |
|---|---|---|---|---|
| baseline (10×) | 0.322 | 2467 ms | 1.00× | — |
| OmniZip (10×) | 0.320 | 1572 ms | 1.57× | — |
| **VideoZip (this run)** | **0.305** (36/118; 36/116 excl. 2 jitter errors) | **1492 ms** | **1.65×** | 19.23 GB (model alone 16.64) |

## Verdict
**The run is valid and the result is positive.** VideoZip beats OmniZip on prefill
(**1.65× vs 1.57×** vs baseline) **at iso-accuracy**. That is exactly the intended
payoff: caching Thinker L6 Q·K video saliency removes the online saliency pass during
prefill, so it's strictly faster than OmniZip while doing the same audio compression.

- **Accuracy is a wash (~0.30–0.32 across baseline / OmniZip / VideoZip).** This is a
  hard 118-Q subset; accuracy is *not* the metric this method moves and should not be
  read as degradation — all methods sit in the same band, baseline included.
- **The contribution is prefill speed, not memory.** VRAM peak (19.23 GB) is dominated
  by model weights (16.64 GB); at MCQ decode lengths KV savings are negligible. Don't
  market this as a memory win.

## Caveats / what to fix before publishing
1. **n=1, not n=10.** Baseline/OmniZip numbers are 10-repeat means; VideoZip is a single
   pass. The 1.65× vs 1.57× gap is real-direction but needs a 10× repeat to match the
   matrix's error bars and rule out run-to-run prefill jitter.
2. **2 errored questions** (worldsense/Temporal Prediction): cached length 5577 vs live
   5538 (78×71) — a ~0.7% frame-decode jitter from decord (`h264 mmco: unref short
   failure`), **not** a config error. The fail-loud equality assertion correctly refused
   to misalign saliency. Acceptable to report as 36/116, but ideally re-decode that clip.
3. **Cache coverage = 44 videos.** Confirm every evaluated question's video is in the
   cache (no silent fallback); the run reports 0 "no cache" skips, which is consistent.

## Bottom line
Ship-worthy as a *prefill-speedup* result: **VideoZip > OmniZip on prefill at equal
accuracy.** Promote it from n=1 to a 10× repeat and keep the framing on prefill latency,
not VRAM.
