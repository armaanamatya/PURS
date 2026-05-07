# Session: VideoZip scaffolding + L6 cache integration + remote eval bring-up

**Date:** 2026-05-06 → 2026-05-07
**Branch:** `mds`
**Outcome:** `videozip/` package created; cache loader + validator running; full
VideoZip orchestrator implemented; `eval_videozip.py` wired and ready to run
remotely once the path-shadowing fix is synced.

---

## 1. OmniSIFT discussion

User opened with `omnisift`. Aurelle (prior agent) returned a write-up of
**OmniSIFT** (arXiv 2602.04804, Feb 2026) — a learned modality-asymmetric
compression method using STVP (spatial+temporal video pruning) and VGAS
(vision-guided audio selection), 4.85M trainable params via STE.

User followed up: *"is there a way we can make this training-free like omnizip?
also does this method do the pruning before the LLM sees them or after?"*

Aurelle's answer:
- **Pre-LLM** placement (encoder output, before Thinker layers).
- STVP is already cosine-only → near-training-free.
- VGAS could be replaced with frozen cross-modal similarity heuristics.

User asked: *"does this plan seem grounded and viable?"*

My critique:
- Aurelle independently re-derived the user's existing `videozip_plan.md`.
- Aurelle's plan is a **strict subset** — misses bidirectional anchoring,
  per-group ratios, attention-logit saliency, adaptive guide selection.
- Worth grafting only the **frozen cross-modal similarity baseline** as one
  ablation row.
- Both plans missed the **L6 cache integration** — that's the headline
  speedup angle (free 1.6× prefill on top of token reduction).

---

## 2. Plan updates (`docs/videozip_plan.md`)

Edits applied:
- §3.6 — L6-cached video saliency rationale and pipeline.
- §4a-L6 — `omnizip_video_saliency_l6()` spec.
- §4a-Sim — `omnizip_video_saliency_simonly()` (Aurelle's frozen baseline).
- Ablation A — added sim-only column.
- Ablation E — live attn vs L6-cached vs L6-no-cache.
- Config dict — added `video_saliency_source`, `l6_cache_dir`.
- §10 implementation order — extended 7 → 10 steps (~400 LOC total).

User confirmed plan superiority: *"so is our @docs/videozip_plan.md superior to
aurelle's suggestion?"* Verdict: **strictly superior on every axis hers
touched, plus three things hers doesn't have** (bidirectional anchoring,
adaptive guide-mode dispatch, per-group ratio mapping).

---

## 3. NotebookLM 3-way comparison

User invoked `/notebooklm` with three sources:
1. `omnisift.pdf`
2. `omnizip.pdf`
3. `docs/videozip_plan.md`

Process:
- Located `notebooklm` CLI at `PURS\venv\Scripts\notebooklm.exe`.
- Created notebook `1d2870bd-64b9-4710-91ef-9353fc840137`.
- Uploaded all three sources (READY in seconds).
- First attempt: `generate report --format briefing-doc` → produced
  **OmniZip-only** writeup (template ignored append instructions).
- Fix: `notebooklm ask` with all three source IDs forced and the question
  read from `_scratch/videozip_question.txt` (PowerShell quote-stripping
  required the file workaround).
- Cleanly extracted balanced JSON (status line had contaminated the file).

Outputs:
- `docs/videozip_briefing.md` — original briefing (kept for archive).
- `docs/videozip_comparison.md` — clean 3-way comparison, 4,898 chars,
  1,191 source references.
- Notebook URL:
  https://notebooklm.google.com/notebook/1d2870bd-64b9-4710-91ef-9353fc840137

Key cited findings:
- **OmniSIFT:** video→audio, learned (4.85M), uni-directional, pre-LLM,
  >40% inference reduction at 25% retention.
- **OmniZip:** audio→video, training-free, uni-directional, pre-LLM,
  2.51–3.42× prefill at 35% retention.
- **VideoZip:** video→audio, training-free, **bi-directional**, pre-LLM,
  free 1.6× prefill from L6 cache.

---

## 4. Codebase audit ("do we have anything similar?")

User asked whether prior code overlapped with this session's discussion.

Findings:
- **L6 cache pipeline already exists for AUDIO only:**
  - `precompute_l6_saliency.py` (project root)
  - `eval_qwen_omni_zip_cached.py` (project root)
  - `viz_layer_depth_experiment.py`, `compute_omnizip_auc.py`
  - `vizzing/layer_depth_all_full.jsonl` — 44 videos × 3 datasets
- **`omnizip_units.py`:** only 3 functions — `omnizip_audio_attn`,
  `omnizip_istm`, `omnizip`. No video-saliency variant.
- **17 ablation scripts** under `ablationscripts/`, all knob-tweaks of the
  audio-guided path. None video-guided.
- **Vendored work:** `divprune-main/`, `ReDiPrune-main/`, `MixKV-main/`,
  `FastKV-main/` — none relevant to video-guided audio compression.
- **`docs/futureplanning.md` Direction #4** ("Upgrade the video side") and
  Direction #2 ("Adaptive rho") prefigured VideoZip but framed differently.

Recommended path: **reuse existing infra, write only the new orchestrator
and the evaluation glue**. No new precompute script needed (next section).

---

## 5. `videozip/` folder scaffolding

User: *"lets get started, but lets create a folder called 'videozip' to
store all of our code/copied code, new code, docs, etc. do not delete any
files or folders, just import or something"*

Folder structure created:
```
videozip/
├── README.md
├── __init__.py                 sys.path bootstrap (probes OmniZip-main / OmniZip-OG)
├── src/
│   ├── _utils.py               _map_retention_to_ratios, _build_audio_groups
│   ├── video_saliency.py       three saliency variants + dispatcher
│   ├── audio_compress.py       per-group wrapper around omnizip_audio_attn
│   ├── istm_audio_anchored.py  dpcknn_audio_guided + ISTM loop
│   └── videozip.py             orchestrator
├── docs/README.md              pointers to authoritative docs (no duplicates)
├── precompute/                 (validator only — no new precompute needed)
├── eval/
│   └── eval_videozip.py        monkey-patch eval, mirrors eval_qwen_omni_zip_cached.py
├── ablations/.gitkeep
├── cache/
│   ├── __init__.py
│   └── loader.py               reads existing JSONL cache
└── tests/test_video_saliency.py
```

**Constraints honored:**
- No deletions.
- No copies of upstream code — `__init__.py` adds `OmniZip-main/omnizip` to
  `sys.path`, so existing `omnizip_units.py` is imported live.
- No edits to upstream paper trees (per memory rule).

---

## 6. Major discovery: precompute already done

While writing the precompute fork I realized `precompute_l6_saliency.py`
**already dumps both audio AND video L6 saliency** in the same JSONL
(see lines 276–277 and 351 of the script). No new precompute needed.

Rather than fork the script, I wrote:
- `videozip/cache/loader.py` — reads `vizzing/layer_depth_all_full.jsonl`,
  returns `{video_key: 1D_score_array}` for the requested layer.
- `videozip/precompute/validate_l6_cache.py` — CPU-only validator that
  confirms cache coverage against metadata.

Also corrected a bug in my `omnizip_video_saliency_l6`: the cache stores
**1D Q·K saliency vectors**, not hidden states. Function signature changed
from `l6_hidden_states: torch.Tensor [T, D]` to `l6_video_scores: torch.Tensor [N_v]`.

---

## 7. Remote bring-up

User on `armaan@lambda-scalar:/data/armaan/purs` with 8× RTX 6000 Ada GPUs.
Active conda env: `omnizip_clean`.

### 7a. Validator output

```
Cache: vizzing/layer_depth_all_full.jsonl
Layer: 6
Total records: 44

Dataset      Records  WithAudio  WithVideo  AudioMin AudioMax VideoMin VideoMax
daily-omni         6          6          6      776     2246     4320     4608
video-mme         12         12         12      773     2219     4320     4608
worldsense        26         26         26      600     2242     3456     9216

Metadata cross-check (all datasets):
  Relevant entries (with questions): 44
  Present in cache:                  44
  Missing from cache:                0
```

44/44 covered. Skipped precompute, went straight to eval.

### 7b. Smoke tests

```
$ python -m videozip.tests.test_video_saliency
all smoke tests passed
```

### 7c. Eval errors and fixes

**Error 1:** `python videozip/eval/eval_videozip.py ...`
→ `ModuleNotFoundError: No module named 'videozip'`

**Cause:** Path-based execution puts the script's dir on `sys.path`, not
the project root.

**Fix:** Added self-bootstrap to top of `eval_videozip.py`:
```python
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
```
Module form (`python -m videozip.eval.eval_videozip ...`) also works.

**Error 2:** `AttributeError: module 'eval_qwen_omni_zip' has no attribute
'OMNIZIP_DEFAULT_RHO_AUDIO'`

**Cause:** sys.path ambiguity — `OmniZip-main/` (or its `omnizip/` subdir)
contains a vendored `eval_qwen_omni_zip.py` shadowing the customized
project-root version. My bootstrap order put OmniZip dirs ahead of
PROJECT_ROOT.

**Fix:** Reordered `videozip/__init__.py` so `_PROJECT_ROOT` is prepended
**last**, ending at `sys.path[0]`:
```python
# OmniZip dirs first
_prepend(_cand)
_prepend(_cand.parent)
# qwen-omni-utils
_prepend(_qwen_root)
# PROJECT_ROOT last → sys.path[0]
_prepend(_PROJECT_ROOT)
```

### 7d. Verification command (pending user run)

```bash
python -c "
import videozip
import eval_qwen_omni_zip as base
print('module file:', base.__file__)
print('has OMNIZIP_DEFAULT_RHO_AUDIO:', hasattr(base, 'OMNIZIP_DEFAULT_RHO_AUDIO'))
"
```

Expected: `/data/armaan/purs/eval_qwen_omni_zip.py` and `True`.

---

## 8. Files written this session

### New
- `videozip/__init__.py`
- `videozip/README.md`
- `videozip/src/__init__.py`
- `videozip/src/_utils.py`
- `videozip/src/video_saliency.py`
- `videozip/src/audio_compress.py`
- `videozip/src/istm_audio_anchored.py`
- `videozip/src/videozip.py`
- `videozip/cache/__init__.py`
- `videozip/cache/loader.py`
- `videozip/precompute/validate_l6_cache.py`
- `videozip/eval/__init__.py`
- `videozip/eval/eval_videozip.py`
- `videozip/tests/__init__.py`
- `videozip/tests/test_video_saliency.py`
- `videozip/docs/README.md`
- `videozip/{ablations,cache,eval,precompute}/.gitkeep` (placeholders)
- `docs/videozip_briefing.md` (NotebookLM auto-generated; OmniZip-focused archive)
- `docs/videozip_comparison.md` (clean 3-way comparison)
- `docs/videozip_comparison_raw.json` (raw chat response with refs)
- `_scratch/videozip_question.txt`

### Modified
- `docs/videozip_plan.md` — §3.6, §4a-L6, §4a-Sim, Ablation A/E, config dict, §10.

### Untouched (per "no deletions" constraint)
- All existing `OmniZip-OG/` / `OmniZip-main/` files.
- All `ablationscripts/*.py`.
- All `vizzing/` cache files.
- All other `docs/` files.

---

## 9. What runs next

```bash
cd /data/armaan/purs
conda activate omnizip_clean

# Verify path resolution
python -c "
import videozip
import eval_qwen_omni_zip as base
print(base.__file__, hasattr(base, 'OMNIZIP_DEFAULT_RHO_AUDIO'))
"

# Quick sanity (daily-omni only, 6 videos / ~18 questions / ~5 min)
CUDA_VISIBLE_DEVICES=0 python -m videozip.eval.eval_videozip \
    --cache vizzing/layer_depth_all_full.jsonl \
    --layer 6 \
    --metadata videos/metadata.json \
    --videos videos \
    --category daily-omni \
    --output vizzing/results_videozip_l6_dailyomni.jsonl \
    --log vizzing/eval_videozip_l6_dailyomni.log \
    --vram_log vizzing/vram_videozip_l6_dailyomni.jsonl \
    --device cuda:0 \
    --measure_prefill

# Full eval (118 questions / ~25-30 min)
CUDA_VISIBLE_DEVICES=0 python -m videozip.eval.eval_videozip \
    --cache vizzing/layer_depth_all_full.jsonl \
    --layer 6 \
    --metadata videos/metadata.json \
    --videos videos \
    --output vizzing/results_videozip_l6.jsonl \
    --log vizzing/results_videozip_l6.log \
    --vram_log vizzing/vram_videozip_l6.jsonl \
    --device cuda:0 \
    --measure_prefill
```

### Useful flags
| Flag | Purpose |
|---|---|
| `--video_saliency_source attn` | Ablation A — live attn instead of cache |
| `--video_saliency_source sim_only` | Aurelle's frozen-cosine baseline |
| `--audio_anchor_beta 0.0` | Ablation B floor — recovers OmniZip's dpcknn |
| `--category daily-omni` | Restrict to one dataset for fast iteration |

---

## 10. Open questions / next session

- Will VideoZip accuracy match or beat OmniZip+L6 (28.8% / 30.5% baseline)?
  Hypothesis: ties on Daily-Omni, slight edge on VideoMME, slight loss on
  AIR-Bench-like audio-primary tasks.
- Audio-anchor-beta sweep (Ablation B) — `{0.0, 0.1, 0.3, 0.5, 1.0}`.
- Live-attn vs L6-cached saliency comparison (Ablation E) — measures
  cache fidelity directly.
- Adaptive guide-mode dispatch (Ablation D) — entropy-based selection.
- Implement non-divisible-by-4 frame branch (currently falls back to
  OmniZip — limits VideoZip's coverage on a small subset of videos).
