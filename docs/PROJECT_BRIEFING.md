# PURS Project Briefing

**Owner:** Armaan Amatya · **Last refreshed:** 2026-05-07 · **Branch:** `mds`

A short, citation-anchored snapshot of what this repo is, what has been done, and where it is going. Use as the entry point for new sessions; deeper notes are linked per section.

---

## 1. Research Goal

Build **training-free, mechanistically-grounded token compression** for omnimodal LLMs (Qwen2.5-Omni, Qwen3-Omni) so audio-video reasoning is faster and cheaper without retraining the model.

Two intertwined questions drive the work:

1. **Where does cross-modal saliency live?** — Layer-by-layer mechanistic interpretability of the Qwen Thinker, looking for an early Thinker layer whose attention already ranks audio/video tokens by relevance.
2. **Can that signal replace the runtime cost of existing compressors?** — Specifically OmniZip's audio-encoder saliency: can a single early-layer signal be **cached once per video** and reused across questions, matching OmniZip's quality at lower bookkeeping cost, and can the same machinery be **inverted (VideoZip)** for video-primary tasks?

Long-term framing: tie the empirical compression result to a **mechanistic story** about Qwen's Thinker–Talker architecture (see `Mechanistic Layer-Level Analysis... .md`), so the final paper has both numbers and an explanation.

---

## 2. Headline Results

### 2a. 10× repeat benchmark matrix (authoritative as of 2026-05-07)

`10x/qwen25_matrix_gpu7_all7_snapkv/` — 10 runs × 2 temperatures × 5 methods, full benchmark, prefill + per-question VRAM logged. Eval config: `fps=2.0`, `max_pixels=100352`, `max_frames_videomme=768`, `max_frames_other=128`, `dtype=bfloat16`.

| Method | Acc T=0.1 (mean ± std) | Acc T=0.9 (mean ± std) | Prefill ms | E2E ms | Process Peak GB | Frame keep |
|---|---|---|---:|---:|---:|---:|
| baseline | **0.311 ± 0.013** | 0.333 ± 0.030 | 2481 | 2592 | 34.1 | 1.000 |
| omnizip (ρ_a=0.3, ρ_v=0.6, g=3) | **0.312 ± 0.013** | 0.328 ± 0.034 | 1614 | 1802 | 21.0 | 1.000 |
| mixkv (budget=256, snapkv) | 0.200 ± 0.011 | 0.220 ± 0.034 | 2304 | 2391 | 24.1 | 1.000 |
| divprune (subset=0.5, frame mode) | 0.203 ± 0.012 | 0.245 ± 0.034 | 1200 | 1295 | 21.4 | 0.500 |
| rediprune (α=0.5, subset=0.5, frame) | 0.218 ± 0.009 | 0.264 ± 0.017 | 1257 | 1356 | 21.4 | 0.500 |

Three findings reshape the strategy:

1. **OmniZip and baseline are statistically tied on accuracy** (0.312 vs 0.311 at T=0.1, within 0.1σ). The earlier "OmniZip 30.5% > baseline 29.7%" gap from the 118-Q snapshot was single-run noise. **OmniZip's real contribution is 35% prefill speedup + 38% VRAM reduction at parity accuracy, not an accuracy win.** The "beat OmniZip" target is a Pareto target on speed/memory/compression-ratio, not on accuracy.
2. **MixKV at budget=256 is broken on Qwen-Omni.** −11 accuracy points vs baseline with only 7% prefill speedup. Audio+video context overflows the 256-token KV cache. Need a budget sweep (512 / 1024 / 2048) before MixKV is usable as a comparator or stacking partner.
3. **ReDiPrune > DivPrune at matched compression** (0.218 vs 0.203 at T=0.1, same 0.5 keep ratio, frame mode). +1.5 pts from text-relevance over pure visual diversity. Validates text-query relevance as a useful signal for any future fusion.

### 2b. Layer-6 cache result (still valid; reframed)

The L6 signal of Qwen2.5-Omni's Thinker is a video-level, question-invariant proxy for OmniZip's audio keep mask. See `docs/method_l6_omnizip.md`, `docs/findings.md`, `docs/professor_update.md`, `vizzing/analysis_v2/v2_analysis.md`.

| Metric | Value |
|---|---|
| Cross-question Spearman floor at L6 | **0.9992** (signal is video-level, not query-specific) |
| Gini jump L3 → L6 | **6.1×** concentration |
| L6 AUC vs OmniZip keep mask | **0.6528 ± 0.1105** (59 WorldSense questions) |
| L14 AUC vs OmniZip keep mask | **0.5253 ± 0.0662** (near-random) |
| L6 beats L14 | **58 / 59** questions; mean AUC gap +0.1276 |
| Within-video L6 AUC std | **0.0013** — caching is justified |
| 118-Q snapshot acc (Baseline / OmniZip / OmniZip+L6) | 29.7% / 30.5% / 28.8% (superseded; within 10× matrix noise) |
| Exact prediction agreement (L6-cache vs OmniZip) | **114/118 = 96.6%** |
| Prefill speedup on 118-Q snapshot | 1.61× (vs 1.62× OmniZip; cached path +15.9 ms median) |

**Reframed read:** an early Thinker layer encodes a usable audio-pruning signal *before* the question is seen — but in light of the 10× matrix showing OmniZip ≈ baseline on accuracy, the L6-cache's value is *not* "preserves OmniZip's accuracy gain" (there is no such gain). It is **a question-invariant cacheable surrogate for OmniZip's audio mask** that enables one-shot offline precomputation and amortization across queries on the same video. The accuracy claim must be re-validated on the 10× harness before any standalone publishable result. The mechanistic interpretation (L6 = modality-input geometry, L14+ = reasoning) is unchanged.

---

## 3. Method Stack

```
Qwen2.5-Omni (Thinker–Talker, TMRoPE) [arXiv 2503.20215]
        │
        ├── OmniZip (audio→video, training-free, CVPR 2026)         [arXiv 2511.14582]
        │       └─ replaces audio-encoder saliency with cached L6 Thinker attention  ← our contribution
        │
        ├── VideoZip (proposed, training-free, video→audio inverse) docs/videozip_plan.md
        │       └─ closes the gap left by OmniSIFT (which trains 4.85M params) [arXiv 2602.04804]
        │
        └── Mechanistic layer-level analysis (Thinker depth sweep)   docs/mech_interp_feasibility.md
                · 28-layer forward-hook attention capture (`viz_attention_depth_curve.py`)
                · cross-modal Gini, last-token modality split, Spearman, AUC, Jaccard, separation, autocorr
```

Comparator methods that have been vendored as references for benchmarking and ablation: SnapKV / PyramidKV / AdaKV (KV pruning), FastKV (token-selective propagation), DivPrune / ReDiPrune (visual token pruning), AngelSlim (Tencent compression toolkit), MixKV (importance + diversity for LVLM KV), inspectus (attention viz library).

---

## 4. What's in the Repo

### `docs/` — narrative + planning
| Doc | Purpose |
|---|---|
| `research_log.md` | Week-by-week log, Weeks 1–13 |
| `findings.md` | Layer sweep results writeup |
| `method_l6_omnizip.md` | The L6-cache method, key numbers, deep-dive |
| `futureplanning.md` | 7 ranked future directions (hybrid score, adaptive ρ, two-stage, video upgrade, diversity, group-retention swap, learned residual) |
| `videozip_plan.md` | Full algorithm design for video-guided audio compression |
| `mech_interp_feasibility.md` | What mechanistic tools (logit lens, patching, probing) apply to Qwen-Omni |
| `qwen2.5-omni-layerexp.md` | Layer experiment plan |
| `Qwen2.5-Omni_Architecture_Documentation.md` | Layer-by-layer architecture reference |
| `dataflowomnizip.md` | Data flow through the OmniZip-modified Thinker |
| `OmniZip-Style Token Compression Meets Speculative Decoding.md` | Future direction: combine compression with spec-decoding |
| `professor_update.md` | Two-version writeup sent to advisors |
| `l6cache.md`, `otherpoitns.md`, `session.md` | Working notes |
| `commands.md`, `commando.md`, `sweep_commands.md`, `EVAL_SCRIPTS_DOCUMENTATION.md`, `vizscripts.md`, `viz_scripts_guide.md` | Run/eval recipes |
| `setupinstructions.md`, `qwen2.5-omni-setup-wsl2.md`, `fixes.md` | Environment, HPC bring-up, fixes |
| `qwen_omni_mixkv_divprune_implementation.md` | Notes on integrating MixKV / DivPrune |
| `omnizip_deep_dive.md`, `dataset_schemas.md` | Reference |

### `scripts/` — analysis & viz drivers
Output mapping in `scripts/outputs.txt`. Notable:
- **Attention probes:** `viz_attention_qwen.py`, `viz_attention_omnizip.py`, `viz_attention_heatmap*.py` (3 variants), `viz_attention_depth_curve.py` (28-layer streaming), `viz_attention_encoders.py` (Q/K hooks on audio+vision encoders).
- **Cross-modal/depth:** `viz_crossmodal_spearman.py`, `viz_layer_depth_experiment.py`, `viz_early_layer_relevance.py` + `_batch.py`.
- **Display only:** `viz_encoders.py`, `viz_omnizip_math.py`, `viz_tmrope.py`, `qwen_omni_pipeline_viz.py`, `generate_qwen_arch_figures.py`.
- **Score analysis:** `analyze_existing_scores.py`, `analyze_v2_scores.py`, `analyze_crossmodal_vs_omnizip.py`.
- **Bench drivers:** `run_qwen_omni_benchmark_matrix.py`, `run_all_attention_viz.py`, `plot_prefill_score_memory_time.py`, `build_10x_prefill_line_xlsx.py`.
- **Sync:** `sync-to-data-armaan.{bash,cmd,ps1}` — push to `/data/armaan/purs` HPC, with progressive excludes.

### `vizzing/` — generated artifacts
Top-level: `all/`, `analysis/` (`scores_analysis.md`), `analysis_v2/` (`v2_analysis.md`), `auc_vs_omnizip/`, `crossmodal_spearman/`, `depth_curve/`, `early_layer_relevance/`, `encoder_attention/`, `layer_depth_experiment/`, `layer_depth_experiment_v2/`, `omnizip_auc_l6_worldsense/`, `omnizip_auc_l14_worldsense/`, plus per-method JSONL/log result dumps for the cached-L6 sweep across Daily-Omni / Video-MME / WorldSense.

### `mechinterp/` — current home of the depth-curve probe
`viz_attention_depth_curve.py` (the 28-layer hook script lives here too) and `outputs/depth_curve/` containing the canonical 5 PNGs: cross-modal curves, self-vs-cross, Gini-by-depth, cross-modal heatmap, last-token attention by depth. Historical copies remain at `vizzing/depth_curve/` for reference.

### Vendored upstream paper trees (read-only reference — do not edit; per memory `feedback_never_edit_paper_source_trees.md`)
`OmniZip-main/`, `Qwen2.5-Omni/`, `FastKV-main/`, `MixKV-main/`, `divprune-main/`, `ReDiPrune-main/`, `AngelSlim-main/`, `inspectus-main/`. To extend any of them, copy into `<project>/vendored/<paper>/` and patch the copy.

---

## 5. Papers Collected (Citations)

**Backbone**
- Qwen Team. *Qwen2.5-Omni Technical Report.* [arXiv:2503.20215](https://arxiv.org/abs/2503.20215). — Thinker–Talker, TMRoPE, block-wise streaming.
- Qwen Team. *Qwen3-Omni Technical Report.* [arXiv:2509.17765](https://arxiv.org/abs/2509.17765). — MoE Thinker–Talker, multi-codebook causal-ConvNet Talker.

**Direct method targets**
- Tao et al. *OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal LLMs.* CVPR 2026. [arXiv:2511.14582](https://arxiv.org/abs/2511.14582), [code](https://github.com/KD-TAO/OmniZip). — 3.42× speedup, 1.4× memory; baseline we replace.
- Ding et al. *OmniSIFT: Modality-Asymmetric Token Compression for Efficient Omni-modal LLMs.* [arXiv:2602.04804](https://arxiv.org/abs/2602.04804). — Trains 4.85 M params for video→audio; **VideoZip is the training-free counterpart**.

**KV / token compression comparators**
- Jo et al. *FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration.* ACL Findings 2026. [arXiv:2502.01068](https://arxiv.org/abs/2502.01068). — Token-Selective Propagation; 1.82× prefill / 2.87× decode speedup.
- Li et al. *SnapKV: LLM Knows What You Are Looking for Before Generation.* NeurIPS 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469).
- Cai et al. *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* [arXiv:2406.02069](https://arxiv.org/abs/2406.02069).
- Feng et al. *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation.* NeurIPS 2025. [arXiv:2407.11550](https://arxiv.org/abs/2407.11550).
- *MixKV: Mixing Importance with Diversity — Joint Optimization for KV Cache Compression in LVLMs.* [arXiv:2510.20707](https://arxiv.org/abs/2510.20707). — Plug-and-play head-wise importance + diversity weighting.
- Alvar et al. *DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models.* CVPR 2025. [arXiv:2503.02175](https://arxiv.org/abs/2503.02175). — Max-Min diversity selection; SOTA across 16 datasets.
- *ReDiPrune: Relevance-Diversity Pre-Projection Token Pruning for Efficient Multimodal LLMs.* [arXiv:2603.24680](https://arxiv.org/abs/2603.24680). — Prunes before the V-L projector.
- Tencent Hunyuan. *AngelSlim: A More Accessible, Comprehensive, Efficient LLM Compression Toolkit.* [arXiv:2602.21233](https://arxiv.org/abs/2602.21233), [code](https://github.com/Tencent/AngelSlim). — FP8/INT8 PTQ + spec-decoding + token pruning toolkit.

**Benchmarks used**
- Hong et al. *WorldSense: Evaluating Real-world Omnimodal Understanding for Multimodal LLMs.* ICLR 2026. [arXiv:2502.04326](https://arxiv.org/abs/2502.04326). — 1,662 AV-synced videos, 3,172 MCQs across 8 categories.
- Zhou et al. *Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment Across Modalities.* [arXiv:2505.17862](https://arxiv.org/abs/2505.17862). — 684 videos / 1,197 cross-modal-temporal MCQs.
- Video-MME — used via OmniZip's `lmms-eval` integration for video-language QA.

---

## 6. Current Direction (post-10× matrix)

The 10× matrix changed the priority stack. The audio-guidance-OFF ablation is now the gating experiment for everything else.

**Immediate (this week):**

1. **Audio-guidance-OFF ablation in OmniZip** — replace audio-derived per-group video budget with random / uniform selection at fixed ρ_v=0.6, all else identical. Two outcomes, both decisive:
   - random ≈ OmniZip → audio guidance is a story not a contribution; pivot to adaptive ρ + diversity.
   - random ≈ baseline → audio signal carries the gain; pursue 3-signal fusion (L6 + encoder + ReDiPrune-style query relevance).
2. **MixKV budget sweep** — 256 / 512 / 1024 / 2048. The current 256 setting crashes accuracy (0.20 vs baseline 0.31) and makes MixKV unusable as a comparator or stacking partner.
3. **ReDiPrune α sweep** at fixed subset=0.5 — quantifies the text-relevance contribution and pre-validates the query branch of any future fusion.

**Next (post-ablation):**

4. **OmniZip + (working) MixKV stacking** — once MixKV's budget is fixed. Pre-LLM × post-LLM compression on orthogonal axes; expected Pareto improvement over OmniZip alone at parity accuracy.
5. **3-signal fusion with L6-variance routing** — L6 cache + encoder attention + ReDiPrune-style text relevance, fused submodularly, with per-query routing driven by L6 within-video AUC variance as a free uncertainty oracle.

**Queued:**

6. **VideoZip** — training-free video→audio inverse of OmniZip. Plan in `docs/videozip_plan.md`. Reframe baseline: target OmniSIFT (which trains 4.85M params), not OmniZip. Functions to add to `omnizip_units.py`: `omnizip_video_saliency`, `omnizip_audio_compress`, `omnizip_istm_audio_anchored`, `omnizip_videozip`. Dispatch via `guide_mode: "audio" | "video" | "adaptive"`.
7. **Hybrid + adaptive extensions** — `docs/futureplanning.md`: hybrid `α·L6 + (1−α)·encoder` (now lower priority since L6 and encoder are correlated and OmniZip ≈ baseline anyway), saliency-entropy-driven adaptive ρ (higher priority — see audio-OFF outcome H1), importance+similarity on video side.
8. **Compression × speculative decoding** — `docs/OmniZip-Style Token Compression Meets Speculative Decoding.md`. Audio-guided dual masks (drafter vs verifier), modality-aware acceptance, vs MSD baseline.

Underneath all of this: keep the **mechanistic story** consistent with `Mechanistic Layer-Level Analysis for Multimodal LLMs in the Qwen2.5/Qwen3 Thinker–Talker Architecture.md` — the L6 result is one data point in that broader layer-level analysis (cross-modal suppression, expert usage in Qwen3-Omni MoE, temporal circuits during chunked-prefill).

---

## 7. Quick Pointers

- **Key memory entries** (auto-loaded across sessions): `MEMORY.md` indexes `project_videozip.md`, `project_futureplanning.md`, `feedback_never_edit_paper_source_trees.md`.
- **HPC sync:** `scripts/sync-to-data-armaan.bash` → `/data/armaan/purs`. Set `SYNC_SLIM=1` to also exclude `OmniZip-main/lmms-eval/`.
- **Bench output layout:** `runs/qwen25_matrix_gpu7_all7_snapkv/<method>/temp_<t>-run_<NN>/{results,vram_log,gpu_samples,run_summary}.{jsonl,json}` plus top-level `summary_by_config.{json,csv}` and `SUMMARY.md`.
- **L6-cache result dumps (vizzing/):** `results_zip_cached_l6_all{,_timed,_vram}.jsonl`, `results_zip_cached_l6_worldsense{,_vram}.jsonl`, `omnizip_auc_l{6,14}_worldsense/omnizip_auc_report.md`.
