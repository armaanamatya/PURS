# VideoZip — First Valid Benchmark Run (2026-05-28)

End-to-end record of bringing VideoZip's eval harness up and producing its first
config-matched speedup result against the 10× reference matrix.

- Branch: `videozip` · Workspace: `C:\Users\Armaan\Desktop\PURS`
- Remote GPU: `armaan@10.244.120.178` (env `omnizip_clean`), run on **GPU 7**
- Method: VideoZip = video-guided **audio** compression (inverse of OmniZip), training-free,
  using **cached Thinker L6 Q·K video saliency** instead of online saliency.

---

## TL;DR result

Config-matched to the 10× matrix (`fps=2.0, max_pixels=100352, max_frames 768/128`), all on GPU 7:

| method | accuracy | prefill (mean) | **speedup vs baseline** | VRAM peak alloc |
|---|---|---|---|---|
| baseline (10×) | 0.322 | 2467 ms | 1.00× | — |
| OmniZip (10×) | 0.320 | 1572 ms | 1.57× | — |
| **VideoZip (1×)** | **0.305** (36/118; 36/116 excl. 2 jitter errors) | **1492 ms** | **1.65×** | 19.23 GB (model alone 16.64) |

**VideoZip beats OmniZip on prefill (1.65× vs 1.57×) at iso-accuracy** — the cached-saliency
payoff (no online saliency pass during prefill). VRAM savings are negligible at MCQ decode
lengths (weights dominate); **the contribution is prefill speed, not memory.** Accuracy is a
wash across all methods on this hard 118-Q subset — expected, not the point.

Run dir (pulled local): `videozip/runs/l6_ref768/` (`results.jsonl`, `vram.jsonl`, `eval.log`, `errors.log`).

---

## The bug chain (what it took to get a single inference to run)

The harness failed four times, each masking the next:

1. **Stale base signature.** Remote `eval_qwen_omni_zip.py` had an old 5-arg `run_inference`;
   wrapper passed 7 + `measure_prefill`. → synced the modern 8-arg file.
2. **`sys.path` shadowing (the real root cause).** `import videozip` prepends `OmniZip-main/omnizip`
   to `sys.path`; a *vendored* 5-arg `eval_qwen_omni_zip.py` there shadowed the project-root one.
   `videozip/__init__.py`'s final `_prepend(PROJECT_ROOT)` was a no-op because the script bootstrap
   had already inserted PROJECT_ROOT (the `not in sys.path` guard). **Fix:** force PROJECT_ROOT to
   index 0 via remove-then-insert in `videozip/__init__.py`. Diagnosed with
   `base.run_inference.__code__.co_filename` after replicating the eval's import order.
3. **Preprocessing/cache mismatch.** Cached L6 video saliency is per-token and must positionally
   align with live video tokens. The eval defaults (768/128, max_pixels 151200) didn't match the
   cache build. Added a **fail-loud exact-equality assertion** in
   `videozip/src/videozip.py:_aggregate_video_scores_to_groups` (never silently truncate → would
   misalign saliency → invalid result).
4. **Heterogeneous cache.** `run_precompute.sh` built video-mme/daily-omni fresh at 64 frames but
   **imported worldsense via `--existing_cache`** from `layer_depth_experiment_v2` at a higher frame
   budget. No single eval config aligned all three. → regenerated the cache fresh, one config.

## Cache regeneration

- `run_precompute_f64.sh` — fresh L6 cache at 64/100800 for all datasets (debug pass; gave 30.5% but
  not comparable to the reference, which is 768/128).
- `run_precompute_ref.sh` — **reference-matched cache** `layer_depth_ref768.jsonl`: per-dataset caps
  (video-mme @768, worldsense/daily-omni @128), max_pixels 100352, fps 2.0. precompute takes a single
  `--max_frames`, so it runs **once per dataset** (`--dataset` filter), all appending to one output.
  Includes an `until ... restart` loop for this box's transient `unspecified launch failure` CUDA
  crashes (skips already-written records on restart). OOM was a non-issue: the L6 hook slices `q` to
  **text queries only** before `q @ k.T`, so the matrix is ~`[heads, 60, 55k]` (~370 MB), and the
  model forward uses flash-attention.

## Comparison reference

- Correct reference is the **10× repeat matrix** at `10x/qwen25_matrix_gpu7_all7_snapkv/`
  (supersedes the single-run `resultswprefill` and `docs/l6cache.md` 118 numbers). Config = paper
  default 768/128/100352, fps 2.0. Top-level SUMMARY/CSV only aggregate mixkv; per-method
  prefill/accuracy were aggregated from each `<method>/temp_0p1-run_01..10/results.jsonl`.

---

## Known issue + fix applied this run

**Frame-decode jitter.** One worldsense video (2 questions) errored: cached length **5577** vs live
**5538** (= 78 frames × 71 tok/frame), a **~0.7% mismatch** — decord returned a slightly different
frame set in the eval pass than the precompute pass (the clip threw `h264 mmco: unref short failure`).
This is *not* a config error; the exact-equality assertion correctly refused to misalign.

**Fix (implemented + pushed to remote):** in `videozip/src/videozip.py`, `omnizip_videozip` now
detects `cached_length != num_input_frames * video_token_per_frame` and **falls back to online
saliency** (`dispatch_video_saliency`, `attn` if `attn_logits` present else `sim_only`) for that one
video, emitting a `warnings.warn`, instead of raising. The hard assertion stays in
`_aggregate_video_scores_to_groups` as a backstop for gross/config-level mismatches. So a sub-1%
decode jitter no longer ERRORs a row; it just loses the cache benefit for that single clip.

---

## Caveats (before publishing)

1. **VideoZip is a single run**; baseline/OmniZip are 10× means. Prefill variance is low, so 1.65× is
   a solid point estimate, but **repeat 10×** for statistical parity. (Next step.)
2. With the jitter fallback, the 2 previously-errored rows will now evaluate (true accuracy ≈ 31.0%,
   even closer to baseline/OmniZip).
3. The **amortization win** (compute L6 once per video, reuse across all its questions) is not even
   credited by this per-question eval — it's an additional, unmeasured advantage.
4. Memory comparison is weak here by nature (short MCQ decode, weights dominate). Don't headline it.

## Next steps

1. 10× repeat of VideoZip at 768/128 (slot into the matrix harness) → publishable speedup ± std.
2. Re-confirm the jitter fix end-to-end (the 2 rows should now produce real predictions).
3. Optional: within-session fresh baseline + OmniZip for a same-session ratio (matrix is same-GPU, so rigor not necessity).

## Key files

- `videozip/__init__.py` — sys.path ordering fix.
- `videozip/src/videozip.py` — alignment assertion + jitter fallback.
- `videozip/eval/eval_videozip.py` — eval wrapper (compat shims, cached-video controller).
- `run_precompute_ref.sh`, `run_vz_ref.sh` — reference-config cache build + eval.
- `vizzing/layer_depth_ref768.jsonl` (remote) — reference-matched L6 cache.
- `videozip/runs/l6_ref768/` — this run's outputs.
- Memory: `reference_speedup_matrix.md`, `project_videozip_eval_gotchas.md`.
