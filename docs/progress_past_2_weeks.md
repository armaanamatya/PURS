# Past 2 Weeks — Progress Summary (2026-04-19 → 2026-05-03)

Auto-derived from `git log` + file mtimes on branch `mds`. Use as a stocktake / weekly-log entry / pre-meeting brief.

> **Status update 2026-05-07:** the 118-question accuracy numbers in this doc (28.8% vs 30.5% etc.) are **superseded** by the 10× repeat matrix at `10x/qwen25_matrix_gpu7_all7_snapkv/`. See `docs/research_log.md §Key Results Summary` and `docs/PROJECT_BRIEFING.md §2a` for authoritative numbers. Headline change: OmniZip and baseline are statistically tied on accuracy (0.312 vs 0.311 at T=0.1, within 0.1σ), so the L6-cache "regression" is within run noise. Operational details below (what was done, when, in which commits) remain accurate.

---

## Short Version

The past two weeks are the period where the Layer-6 cache idea moved from a half-formed hypothesis to a closed empirical loop with a mechanistic frame around it, plus a queued-up next direction.

Concretely, five things shipped:

1. **Layer-depth sweep + AUC alignment experiment** ran end-to-end. Added four viz/probing scripts in one commit on **4/24** (`viz_attention_depth_curve`, `viz_attention_encoders`, `viz_early_layer_relevance`, `viz_early_layer_relevance_batch`), ran on 39 videos across WorldSense / Video-MME / Daily-Omni, and wrote the analysis. The decisive output: Layer 6 AUC vs OmniZip's audio keep mask = **0.6528 ± 0.1105**, Layer 14 = **0.5253 ± 0.0662** (near-random); L6 wins on 58/59 questions. Artifacts under `vizzing/omnizip_auc_l6_worldsense/`, `vizzing/omnizip_auc_l14_worldsense/`, `vizzing/analysis/`, `vizzing/analysis_v2/`.
2. **The actual cached-L6 replacement benchmark.** Precomputed Layer 6 saliency offline once per video, patched OmniZip to read from the cache, ran the full 118-question pool with timing + VRAM logging. Result: **28.8% vs OmniZip's 30.5%**, 114/118 identical predictions, prefill within +15.9 ms median (1470 ms vs 1455 ms), identical 18.5 GB peak VRAM. Result dumps in `vizzing/results_zip_cached_l6_*.jsonl` (5 variants: all, all_timed, all_timed_vram, worldsense, worldsense_vram).
3. **Wrote the layer-by-layer mechanistic interpretability analysis** for the Qwen2.5/Qwen3 Thinker–Talker stack (`Mechanistic Layer-Level Analysis... .md`). It frames the L6 finding as one observation inside a larger map: which Thinker depths carry low-level perceptual structure, fusion, vs reasoning; what TransformerLens-style probes (linear probes, activation patching, causal tracing along TMRoPE) apply to omni models; and how to use compression itself as a mechanistic probe. Turns the empirical result into a "why."
4. **VideoZip plan** (`docs/videozip_plan.md`, 4/27): full algorithm spec for the training-free inverse of OmniZip — video saliency drives audio compression ratios, audio embeddings anchor video-token selection (β-weighted into ISTM dpcknn). Four functions designed at code level, four ablations written, literature gap explicit (OmniSIFT trains 4.85M params; this wouldn't).
5. **Repo and docs consolidated** (5/03): all `.md` notes moved into `docs/`, `mechinterp/` created as the canonical home for the depth-curve probe with the 5 PNGs, `docs/PROJECT_BRIEFING.md` written as a single-page entry point with verified arXiv citations, `docs/professor_update.md (Update 2 section, post-merge)` drafted as the follow-up to the prior advisor email.

**New today (5/03), not yet reported anywhere:** started a `fastkv_omni/` subproject porting FastKV's Token-Selective Propagation onto Qwen2.5-Omni — `plan/PLAN.md`, `src/qwen25omni_fastkv.py`, `analysis/a3_tsp_sweep.py`, `analysis/kl_vs_baseline.py`, vendored copies of FastKV's `llama_model.py` and `utils.py`. This adds a third comparator to the compression line of work alongside OmniZip and (planned) VideoZip.

---

## Detailed Version

### Commits on branch `mds` in window

| SHA | Date | Subject | Scope |
|---|---|---|---|
| `09aaf67c` | 2026-04-24 | add attention viz scripts and update eval pipelines | 8 files, +1954 — the 4 new probes + eval-pipeline updates for base/divprune/mixkv/zip |
| `6d9e3eae` | 2026-04-25 | whatimworkignon | 3387 files, +286764 — bulk import of vendored paper trees + run dumps |
| `a52852e2` | 2026-04-25 | mds | 56 files, +8705 — mechanistic analysis doc + adjacent |
| `488fd614` | 2026-04-25 | txt | +83 |
| `20234991` | 2026-04-25 | others | +146 |
| `a765448f` | 2026-05-03 | move docs to docs/ and update configs | 18 files, ~±4900 |
| `3ca81787` | 2026-05-03 | cleanup | 60 files, ~±9000 — final reorg |

(Branch `whatimworkingon` is a parallel snapshot of `6d9e3eae`.)

### Files written / touched, by area

**Documentation (`docs/`)** — by mtime:
- 4/20 `viz_scripts_guide.md`
- 4/21 `session.md`
- 4/24 `findings.md` — full structured layer-sweep writeup
- 4/25 `method_l6_omnizip.md`, `l6cache.md`, `professor_update.md`, `futureplanning.md` (7 ranked future directions), `otherpoitns.md`
- 4/27 `videozip_plan.md` — full VideoZip algorithm + ablation grid
- 5/03 `research_log.md`, `mech_interp_feasibility.md`, `PROJECT_BRIEFING.md`, `professor_update.md (Update 2 section, post-merge)`

**Probes / scripts (all under `scripts/` after 5/03 reorg):**
`viz_attention_depth_curve.py`, `viz_attention_encoders.py`, `viz_attention_heatmap{,_omnizip,_qwen}.py`, `viz_attention_omnizip.py`, `viz_attention_qwen.py`, `viz_crossmodal_spearman.py`, `viz_early_layer_relevance.py`, `viz_early_layer_relevance_batch.py`, `viz_layer_depth_experiment.py`, `analyze_crossmodal_vs_omnizip.py`, `analyze_existing_scores.py`, `analyze_v2_scores.py`.

**Generated artifacts (`vizzing/`)** — 4/24–4/25:
- `analysis/scores_analysis.md` + `.jsonl` + `.png` — L0/L1 metric writeup
- `analysis_v2/v2_analysis.md` + `.jsonl` + `.png` — L6/L14 metric writeup
- `depth_curve/` — 5 canonical PNGs (cross-modal curves, self-vs-cross, Gini-by-depth, cross-modal heatmap, last-token attention)
- `encoder_attention/` — 5 PNGs (audio + vision encoder entropy and heatmaps + comparison)
- `layer_depth_experiment/` — `layer_depth_stats.jsonl`, `layer_depth_summary.png`
- `omnizip_auc_l6_worldsense/` and `omnizip_auc_l14_worldsense/` — `omnizip_auc_report.md`, `_questions.jsonl`, `_summary.json`
- `early_layer_relevance/scores.jsonl`
- `auc_vs_omnizip/auc_vs_omnizip.jsonl` + `.png`, `crossmodal_spearman/` (6 plots + stats)
- `results_zip_cached_l6_{all,all_timed,all_timed_vram,all_vram,worldsense,worldsense_vram}.jsonl/log`

**`mechinterp/` (new today, 5/03):** `viz_attention_depth_curve.py`, `README.md`, and `outputs/depth_curve/` containing the canonical 5 PNGs. New canonical home for the depth probe.

**`fastkv_omni/` (new today, 5/03):**
- `plan/PLAN.md` — the implementation plan for porting FastKV onto Qwen2.5-Omni
- `src/qwen25omni_fastkv.py` — Qwen-Omni FastKV adapter
- `analysis/a3_tsp_sweep.py` — Token-Selective Propagation layer sweep
- `analysis/kl_vs_baseline.py` — KL divergence vs full-cache baseline
- `vendored/fastkv/llama_model.py`, `utils.py` — vendored from FastKV-main per the never-edit-upstream rule

**Figures** added (uncommitted at root): `prefill.png`, `qwen_omni_pipeline.png`, `relevance_pruning_*.png` (5 variants), `task_relevance_*.png` (6 variants), `viz_encoders_arch.png`, `viz_omnizip_math.png`, `viz_tmrope.png`, `run3results.png`, `utcssummer2026.png`. Plus poster files `URD Poster.pptx` and `URD Poster Templates (4).pptx` — looks like a poster session was prepped in window.

### Headline arc

- **Week 12 (4/19–4/26)** was an empirical week: probes built, sweep run, AUC experiment added when Gini turned out to be ambiguous between L6 and L14, decisive result locked in (L6 wins 58/59), full benchmark run, professor update written.
- **Week 13 (4/27–5/03)** was a synthesis week: VideoZip plan written 4/27, mechanistic interpretability writeup, research log updated, repo reorganized into `docs/` + `mechinterp/`, `PROJECT_BRIEFING.md` and `professor_update.md (Update 2 section, post-merge)` drafted, and a new `fastkv_omni/` port started today.

### Methodological observation worth keeping

Gini concentration was ambiguous between L6 and L14 — both showed jumps. AUC against the *actual* downstream keep mask was the metric that broke the tie (0.65 vs 0.53). Without that second pass, the project would have shipped the wrong layer. Keeping this lesson for any future layer-probing (VideoZip's video-saliency layer pick, FastKV's TSP layer choice, Qwen3-Omni MoE expert routing).

### State of the working tree

Last commit (`3ca81787 cleanup`) on 5/03 finished the docs/ reorg, but the following are still uncommitted:
- `docs/PROJECT_BRIEFING.md`, `docs/professor_update.md (Update 2 section, post-merge)`
- `Mechanistic Layer-Level Analysis... .md` at repo root
- All vendored paper trees as untracked: `OmniZip-main` (was tracked), `FastKV-main`, `MixKV-main`, `divprune-main`, `ReDiPrune-main`, `AngelSlim-main`, `inspectus-main`, `Qwen2.5-Omni`
- All of `fastkv_omni/`
- `figures/` PNGs and the URD Poster files
- New `vizzing/auc_vs_omnizip/` and `vizzing/crossmodal_spearman/` subfolders
- Uncommitted deletes for the original-location viz scripts that were copied into `scripts/` (the reorg moved them but the old paths still register as `D` until staged)

A consolidating commit (e.g. "consolidate docs and mech-interp; add fastkv_omni port; vendor reference paper trees") would close the loop on the reorg.

### What this lines up for the next two weeks

The pieces that are now infrastructure-ready:
- A reproducible layer-probing harness (`viz_early_layer_relevance_batch.py` + analysis scripts).
- A working cached-L6 OmniZip variant with timing + VRAM logging.
- A drafted VideoZip with four function specs and a four-ablation grid.
- A FastKV port scaffold with TSP sweep + KL-vs-baseline harness.

The natural next experiments are: (a) the multi-turn caching demonstration with the L6 cache (the missing experiment from the prior professor update), (b) implementing the four VideoZip functions and running ablation A (direction of guidance), and (c) running the FastKV TSP sweep on Qwen2.5-Omni and comparing against L6-cached OmniZip on the same 118-question pool.
