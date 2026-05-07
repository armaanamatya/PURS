# [Armaan Amatya] Research Log
Spring 2026

---

## Week 13 (4/27 – 5/03): ~10 hours
**Goals:**
- Reorganize all documentation into `docs/`
- Write professor update summarizing full results
- Draft VideoZip plan as next research direction

**Sunday 4/27 – Tuesday 4/29 (~4 hours):**
- Wrote up `professor_update.md`: full two-version summary of the Layer-6 cache replacement experiment, including the layer sweep, AUC result, and the cached L6 vs. OmniZip paired comparison
- Key numbers finalized: Baseline 29.7%, OmniZip 30.5%, OmniZip+L6-cache 28.8%; prefill times 1455 ms vs 1470 ms; both at 18.5 GB peak VRAM; 114/118 questions identical between OmniZip and cached-L6 versions

**Wednesday 4/30 – Thursday 5/01 (~4 hours):**
- Wrote `videozip_plan.md`: full algorithm design for VideoZip, the training-free inverse of OmniZip where video guides audio compression instead of audio guiding video
- Identified the literature gap: OmniSIFT (Feb 2026) already does video→audio but requires 4.85M parameter training; VideoZip would be training-free with audio-anchored ISTM
- Drafted four new functions: `omnizip_video_saliency()`, `omnizip_audio_compress()`, `omnizip_istm_audio_anchored()`, `omnizip_videozip()` (entry point)
- Ablation plan written: direction of guidance, audio anchor beta sweep, per-group vs. global compression, adaptive guide selection via attention entropy

**Saturday 5/03 (~2 hours):**
- Moved all `.md` docs and figures from root into `docs/` folder
- Updated configs and eval script paths to match new structure (commit: `move docs to docs/ and update configs`)

---

## Week 12 (4/19 – 4/26): ~14 hours
**Goals:**
- Run layer depth sweep: which Thinker layer has the best saliency signal?
- Add ROC AUC alignment metric (stronger than Gini alone)
- Write up findings, plan next experiments, and draft future directions doc

**Sunday 4/20 – Tuesday 4/22 (~6 hours):**
- Added `viz_attention_depth_curve.py`, `viz_attention_encoders.py`, `viz_early_layer_relevance.py`, `viz_early_layer_relevance_batch.py` — scripts to extract and plot cross-modal relevance from Thinker layers `[0, 1, 3, 6, 10, 14, 20, 27]`
- Ran over 39 videos (WorldSense ×21, Video-MME ×12, Daily-Omni ×6)
- First pass metrics: cross-question Spearman correlation, Gini concentration index, score spread
- Key finding: signal is question-invariant — cross-question Spearman never drops below 0.917 at any layer; this is closer to a video-level semantic saliency than a question-specific relevance map
- Gini curve: first strong jump at Layer 6 (0.033 → 0.327), second peak at Layer 14 (0.341)

**Wednesday 4/23 – Thursday 4/24 (~5 hours):**
- Added ROC AUC experiment: compared each layer's saliency against OmniZip's actual audio keep mask on 59 WorldSense questions
- **Decisive result**: Layer 6 AUC = 0.6528 ± 0.1105 vs. Layer 14 AUC = 0.5253 ± 0.0662 (near-random)
- Layer 6 beats Layer 14 on 58/59 questions; mean gap +0.1276
- Within-video AUC std for Layer 6: 0.0013 — extremely stable across questions on the same video; confirms caching makes sense
- Layer 6 gains over Layer 14 largest on audio-heavy tasks: audio source localization (+0.35 AUC), event recognition (+0.28), human-object interaction (+0.26)
- Committed: `add attention viz scripts and update eval pipelines`

**Friday 4/25 (~3 hours):**
- Wrote `findings.md`: full structured writeup of all layer sweep results and what they mean for the paper
- Wrote `futureplanning.md`: seven future directions ranked by expected impact — hybrid score (L6 + encoder), adaptive rho, two-stage pruning, video-side upgrade, diversity term, group-level retention swap, lightweight learned residual
- Also wrote `otherpoints.md`, `session.md`, and updated `method_l6_omnizip.md`

---

## Week 11 (4/12 – 4/18): ~10 hours
**Goals:**
- Measure prefill time end-to-end across methods
- Complete benchmarking PR and merge
- Update website with results

**Sunday 4/13 – Monday 4/14 (~4 hours):**
- Added prefill time measurement to eval scripts: timestamps around `model.generate()` to capture time-to-first-token separately from decode time
- Confirmed: OmniZip prefill time ~1455 ms; baseline Qwen2.5-Omni prefill ~5100 ms on same hardware; that's ~3.5x speedup at prefill
- Commit: `measureprefilltime`

**Tuesday 4/15 – Wednesday 4/16 (~3 hours):**
- Finalized Flash Attention 2 integration across all eval scripts
- Confirmed FA2 doesn't break OmniZip's custom attention logic (separate `return_logits` path uses explicit QK^T, not FA2)
- Commit: `flashattentionadded`

**Thursday 4/17 – Friday 4/18 (~3 hours):**
- Merged PR #6 (benchmarking branch)
- Updated website with current results table
- Minor fix to eval script argument parsing
- Commits: `update`, `Merge pull request #6 from armaanamatya/benchmarking`, `website`, `fix`

---

## Week 10 (4/05 – 4/11): ~12 hours
**Goals:**
- Add Flash Attention 2 support to accelerate evaluation
- Get MixKV and DivPrune running as comparison baselines
- Fix quantization-related FA2 bug

**Sunday 4/06 (~5 hours):**
- Added `flash_attention_2` as `attn_implementation` option in eval scripts
- Major bug: FA2 incompatible with GPTQ-quantized model weights — FA2 requires specific dtype alignment that quantized models break
- Fix: detect quantization via config and fall back to SDPA when GPTQ is active
- Got new results with FA2 enabled — slight improvement in VRAM efficiency
- Commits: `flashattention`, `update data`, `newresults`, `correctresults`, `FA2fixforquant`, `vramresults`, `updatetoruns`

**Monday 4/07 – Wednesday 4/09 (~4 hours):**
- Wrote `eval_qwen_omni_mixkv.py`: implements MixKV-style KV cache compression during prefill
  - `MixKVCompressor` class: per-head attention importance + key diversity + value norm scoring
  - Runtime monkeypatch of Qwen2.5-Omni's SDPA attention forward via `_make_mixkv_sdpa_forward()`
  - Defaults: `budget=256`, `window_size=32`, `select_method="snapkv"`
- Wrote `eval_qwen_omni_divprune.py`: implements DivPrune max-min cosine-distance diversity selection
  - `divprune_select()`: greedy farthest-point subset construction
  - Two modes: `frame` (prune decoded frames) and `token` (prune pixel_values_videos)
  - Default: `subset_ratio=0.5`, `prune_mode="frame"` (frame mode is safer for Qwen2.5-Omni)

**Thursday 4/10 – Friday 4/11 (~3 hours):**
- Ran MixKV and DivPrune evaluations, collected results
- Wrote `qwen_omni_mixkv_divprune_implementation.md`: detailed notes on what was ported from each paper vs. adapted for Qwen2.5-Omni
- Commit: `mixkvndivprunedata`; merged PR #4 and PR #5 from `mixkvndivprune` branch

---

## Week 9 (3/29 – 4/04): ~10 hours
**Goals:**
- Run full second round of baseline + OmniZip evaluations (run2)
- Harden eval scripts: fix edge cases, add shared MCQ parser, audio fallback

**Monday 3/31 (~3 hours):**
- Re-ran full eval with corrected video paths and audio handling
- run1 results committed: `qwen25_baseline_20260331_032724` and `qwen25_omnizip_20260331_033058`
- Commits: `oldresults`, `properresults`

**Wednesday 4/02 (~4 hours):**
- Ran `run2` experiments: baseline and OmniZip with updated pipeline
- Results committed in `run2/baseline/` and `run2/omnizip/`
- Commits: `run2results`, `newresults`, `updatestocmmits`, `update`

**Friday 4/04 (~3 hours):**
- Hardened eval scripts: enforced text-only generation throughout (no talker), added shared MCQ answer extraction function, improved audio fallback when video has no audio track
- Key fix: when `moviepy` reports no audio, gracefully set `use_audio_in_video=False` instead of crashing
- Commit: `Harden eval scripts: text-only generation, shared MCQ parser, audio fallback fixes`

---

## Week 8 (3/22 – 3/28): ~14 hours
**Goals:**
- Fix final OmniZip integration errors
- Run clean baseline vs. OmniZip evaluation
- Set up attention visualization pipeline

**Monday 3/23 – Wednesday 3/25 (~3 hours):**
- Reading and reviewing Qwen2.5-Omni architecture documentation in detail
- Worked through `modeling_qwen2_5_omni.py` line by line — especially the OmniZip injection at lines 2547-2584 and the M-RoPE position ID handling
- Wrote `Qwen2.5-Omni_Architecture_Documentation.md`

**Thursday 3/26 (~3 hours):**
- Renamed/reorganized evaluation scripts for clarity
- Commit: `renames`

**Friday 3/27 (~6 hours):**
- **Critical bug fix**: audio input needs to be resampled to 16 kHz WAV before passing to the processor; added `librosa.resample()` call and WAV conversion step — this was the root cause of most OmniZip failures
- Ran clean Qwen2.5-Omni baseline — results committed: `resultsforqwen25baseline`
- OmniZip finally running end-to-end without errors: commit `finalfixforfinalerrorsonomnizip`
- Added and documented attention extraction scripts, merged PR #1 from `attention` branch
- Wrote initial deep-dive doc: `omnizip_deep_dive.md`
- Commits: `audiotowav`, `resultsforqwen25baseline`, `updates`, `finalfixforfinalerrorsonomnizip`, `docs`, `Merge pull request #1 from armaanamatya/attention`

**Saturday 3/28 (~2 hours):**
- Ran several analysis scripts on the first clean results, started attention heatmap visualizations

---

## Week 7 (3/15 – 3/21): ~12 hours
**Goals:**
- Debug OmniZip integration errors
- Get attention visualization working
- Understand audio encoder modification in OmniZip source

**Sunday 3/16 (~3 hours):**
- Hit multiple OmniZip errors: shape mismatch in `attn_logits` downsampling, audio chunks not aligning with token positions
- Read through `omnizip_units.py` and `modeling_qwen2_5_omni.py` in detail to understand what changed vs. vanilla Qwen2.5-Omni
- Traced the `return_logits=True` path through the audio encoder: only the last encoder layer computes explicit QK^T for the saliency signal; Flash Attention runs normally for all other layers and the actual forward pass
- Commit: `omnizip errors`

**Monday 3/17 – Tuesday 3/18 (~5 hours):**
- Fixed `attn_logits` shape issues: the `[H, T, T]` logit tensor must be head-averaged and column-summed to `[T]`, then pairwise-averaged to `[T/2]` to match the `AvgPool1d` downsampling in the audio encoder
- Got first OmniZip results
- Added VRAM logging: track `torch.cuda.max_memory_allocated()` per sample
- Fixed VRAM logging bug
- Added `viz_attention_heatmap.py`, `viz_attention_qwen.py`, `viz_attention_omnizip.py` — attention heatmap over audio tokens
- Commits: `omnizip results`, `correctdata`, `attentionviz`, `vramfix`, `vramusage`, `vizscriptsnresults`

**Thursday 3/20 (~4 hours):**
- Worked on attention extraction: capturing Thinker self-attention for audio and video tokens at different layers
- Added `attentioncode` — scripts to extract and store per-layer attention weights
- Fixed markdown docs
- Commits: `attentioncode`, `fixmd`

---

## Week 6 (3/08 – 3/14): ~12 hours
**Goals:**
- Set up OmniZip repo and lmms-eval
- Write first eval script for Qwen2.5-Omni
- Run initial inference tests

**Thursday 3/13 (~12 hours):**
- Initial commit: set up project repo, added video assets, configured lmms-eval integration
- Wrote `eval_qwen_omni.py`: full evaluation script for baseline Qwen2.5-Omni over VideoMME / WorldSense / AVUT
  - Model loading with `Qwen2_5OmniForConditionalGeneration` and `Qwen2_5OmniProcessor`
  - `process_mm_info()` for audio/video extraction
  - `text-only` generation mode (`return_audio=False`, talker disabled)
  - JSONL result logging, VRAM log, per-domain accuracy breakdown
- Wrote `eval_qwen_omni_zip.py`: same structure but with `WRAPPER=OmniZip` env var enabling patched model class
- Set up project website
- Commits: `Initial commit`, `vids`, `results&eval`, `website`, `newevalscript`

---

## Week 5 (3/01 – 3/07): ~8 hours
**Goals:**
- Finish reading OmniZip paper in detail
- Clone and understand OmniZip source code structure
- Set up WSL2 environment for GPU inference

**~8 hours across the week:**
- Read OmniZip paper (arXiv:2511.14582, Tao et al., Nov 2025) end to end
- Key takeaways:
  - Training-free, inference-time token compression for Qwen2.5-Omni
  - Uses audio encoder's last-layer self-attention as a "free" saliency signal
  - Jointly compresses audio and video; audio information density dynamically sets per-window video budget
  - Video compressed via interleaved spatial (DPC-KNN) + temporal (novelty detection) scheme
  - 3.42x inference speedup, 1.4x memory reduction, near-zero accuracy degradation (paper claims)
- Cloned OmniZip repo and lmms-eval, explored file structure
- Wrote `qwen2.5-omni-setup-wsl2.md`: WSL2 setup instructions, Flash Attention 2 build, CUDA environment
- Wrote `setupinstructions.md`: step-by-step for running baseline and OmniZip evals
- Understood the two-path design in `omnizip()` (Path A: nframes%4==0, Path B: general fallback)
- Opened questions going into week 6: how does the audio saliency score align with actual token importance? Is last-layer attention actually the best layer?

---

## Week 4 (2/22 – 2/28): 10 hours
**Goals:**
- Continue exploring author's papers and the subsequent lab's paper/works
- Caught up on KV cache, quantization methods

## Week 3 (2/22 – 2/28): 12 hours
**Goals:**
- Understand Qwen architecture better — finish tech report
- Finish baseline paper and explore author's past papers on same topic

## Week 2 (2/15 – 2/21): 8 hours
**Goals:**
- Understand Qwen architecture better
- Explore some black box testing methods

**Sunday 2/15 (1 hour):**
- Researched Qwen architecture
- Some testing done on HF: https://huggingface.co/spaces/Qwen/Qwen2.5-Omni-7B-Demo

## Week 1 (2/08 – 2/14): 8 hours
**Goals:**
- Set up research log
- Answer reflection questions below
- Briefly go over baseline paper from professor

**Sunday 2/09 (1 hour):**
- Set up research log
- Answered initial reflection questions:

  **Q: What are you most excited about in UR2PhD, and why?**
  A: I'm most excited about UR2PhD because it makes the path from undergrad to a PhD feel real and accessible. The structured mentorship and early research exposure really stand out to me. I'm especially drawn to exploring computer science at the intersection of deep math and systems or architecture, and the program feels like a great place to build strong research foundations while figuring out that direction.

  **Q: What are you most nervous about in UR2PhD, and why?**
  A: I'm most nervous about figuring out whether the PhD path truly fits me long term. I really enjoy building things every day across both hardware and software, and I know PhD research can be more narrow and specialized. I'm excited to explore research deeply, but I also want to make sure that research as a lifestyle aligns with how I naturally like to build and think.

  **Q: What did you think about the two sample research logs? What were their strengths? What (if any) aspects will you borrow for your own research logs?**
  A: I liked the sample research logs because they showed how reflecting on a paper can make reading way more useful. The biggest strengths were the clear summaries and how they focused on key contributions. I also liked that they included personal questions and critiques instead of just repeating what the paper said. For my own research logs, I want to copy the concise summaries and the habit of writing takeaways and open questions. That seems really helpful for reviewing papers later and actually building a deeper understanding over time.

**Tuesday 2/10 – 2/14 (5-7 hours):**
- What does OmniZip do
- Why it works
- How it works
- Explored HF demos

---

## Key Results Summary (as of 5/07/2026)

### Authoritative results — 10× repeat matrix (`10x/qwen25_matrix_gpu7_all7_snapkv/`)

10 runs × 2 temperatures (0.1 / 0.9) per method on the full benchmark with measured prefill + per-question VRAM. Eval config: `fps=2.0`, `max_pixels=100352`, `max_frames_videomme=768`, `max_frames_other=128`, `dtype=bfloat16`.

| Method | Acc T=0.1 (mean ± std) | Acc T=0.9 (mean ± std) | Prefill ms | E2E ms | Process Peak GB | Frame keep |
|---|---|---|---:|---:|---:|---:|
| baseline | **0.311 ± 0.013** | 0.333 ± 0.030 | 2481 | 2592 | 34.1 | 1.000 |
| omnizip (ρ_a=0.3, ρ_v=0.6, g=3) | **0.312 ± 0.013** | 0.328 ± 0.034 | 1614 | 1802 | 21.0 | 1.000 |
| mixkv (budget=256, snapkv) | 0.200 ± 0.011 | 0.220 ± 0.034 | 2304 | 2391 | 24.1 | 1.000 |
| divprune (subset=0.5, frame mode) | 0.203 ± 0.012 | 0.245 ± 0.034 | 1200 | 1295 | 21.4 | 0.500 |
| rediprune (α=0.5, subset=0.5, frame) | 0.218 ± 0.009 | 0.264 ± 0.017 | 1257 | 1356 | 21.4 | 0.500 |

**Three findings that reshape the story:**

1. **OmniZip and baseline are statistically tied on accuracy** (0.312 vs 0.311 at T=0.1 — within 0.1σ). The earlier "OmniZip 30.5% > baseline 29.7%" gap on the 118-Q snapshot was single-run noise. OmniZip's real contribution is **35% prefill speedup + 38% VRAM reduction at parity accuracy**, not an accuracy win. The "beat OmniZip" target is therefore a Pareto target on speed/memory/compression-ratio, not on accuracy.
2. **MixKV at budget=256 is broken on Qwen-Omni.** −11 accuracy points vs baseline with only 7% prefill speedup. Audio+video context overflows the 256-token KV cache. Need a budget sweep (512 / 1024 / 2048) before MixKV is usable as a comparator or a stacking partner.
3. **ReDiPrune beats DivPrune at matched compression** (0.218 vs 0.203 at T=0.1 with the same 0.5 keep ratio and frame mode). The +1.5 points come from adding text-relevance over pure visual diversity. Confirms text-query relevance as a useful signal worth incorporating into a fusion.

### Earlier 118-Q snapshot (now superseded — different config)

These numbers used an older eval config (different max_pixels / max_frames defaults) and a 118-question subset; kept for historical reference only.

| Method | Accuracy (118-Q) | Prefill | Peak VRAM |
|---|---|---|---|
| Baseline | 35/118 = 29.7% | ~5100 ms | ~18.5 GB |
| OmniZip (ρ_a=0.3, ρ_v=0.6) | 36/118 = 30.5% | 1455 ms | 18.5 GB |
| OmniZip + cached L6 | 34/118 = 28.8% | 1470 ms | 18.5 GB |

The 1-question gaps in this snapshot are within run-to-run noise on the 10× matrix. The L6-cache study should be re-run on the 10× harness before any conclusions about its acc-delta vs OmniZip.

### Layer sweep (Thinker layers [0,1,3,6,10,14,20,27], 39 videos, 59 questions)

- Signal is question-invariant: cross-question Spearman ≥ 0.917 at all layers
- Gini first strong jump at Layer 6 (0.033 → 0.327); second peak at Layer 14 (0.341)
- Layer 6 AUC vs OmniZip keep mask: 0.653 ± 0.111
- Layer 14 AUC vs OmniZip keep mask: 0.525 ± 0.066 (near-random)
- L6 beats L14 on 58/59 questions; within-video AUC std = 0.0013

### Next planned experiment

**Audio-guidance-OFF ablation.** Modify `eval_qwen_omni_zip.py` to replace OmniZip's audio-derived per-group video budget with random / uniform selection at fixed ρ_v=0.6, all other settings identical. Question being decided: does OmniZip's compute saving come from the audio-guided *selection*, or just from compressing to the budget? If random ≈ OmniZip → audio guidance is a story not a contribution; pivot to adaptive ρ + diversity. If random ≈ baseline → audio signal carries the gain; pursue 3-signal fusion (L6 + encoder + ReDiPrune-style query relevance). Either result decides the rest of the project's direction.

VideoZip and the MixKV budget sweep are queued behind this ablation.
