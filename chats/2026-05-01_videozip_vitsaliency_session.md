# Session: VideoZip & ViT Saliency for OmniZip

**Date:** 2026-05-01
**Topic:** Inverting OmniZip's audio→video guidance and adding ViT feature-norm saliency to video pruning

---

## 1. Initial Question — Can we swap audio/video roles in OmniZip?

User asked whether the OmniZip paper's audio-guided-video paradigm could be inverted to video-guided-audio.

**Research findings (web search + context7):**
- OmniZip (CVPR 2026, arXiv 2511.14582): training-free, audio guides video pruning
- OmniSIFT (arXiv 2602.04804, Feb 2026): does video-guided audio compression but **requires training** (4.85M params, straight-through estimator)
- **Gap identified:** training-free + video-guided + bidirectional cross-modal anchoring → VideoZip

---

## 2. VideoZip Design (Full Plan)

Plan written to `videozip_plan.md`.

**Algorithm differences from OmniZip:**

| Component | OmniZip | VideoZip |
|---|---|---|
| Saliency target | Audio tokens | Video tokens (frame-aggregated) |
| Temporal guide | audio retention → video ratios | video retention → audio ratios |
| Cross-modal anchor | video → audio anchors | + audio → video anchors (in dpcknn) |
| Compression order | audio first | video saliency first, then audio, then video ISTM |
| Fixed ratio modality | Audio (rho_audio) | Video (rho_v, uniform) |

**5-stage pipeline:**
1. `omnizip_video_saliency()` — per-token attention → per-frame mean → per-group mean, normalized to [0,1]
2. Map video_group_retention → audio_merging_ratios via `mapped = max_ratio + (min_ratio-max_ratio)*ret`
3. Per-group audio compression (re-uses `omnizip_audio_attn()`)
4. ISTM with audio-anchored dpcknn: `score = diversity + beta*max_sim(patch, audio_dominant)`
5. Build global mask

---

## 3. Files Created

### Implementation files (no official OmniZip files modified)

- **`OmniZip-main/omnizip/videozip_units.py`** — VideoZip core algorithm
  - `omnizip_video_saliency()` — frame-aggregated attention saliency
  - `omnizip_istm_audio_anchored()` — ISTM with audio-guided dpcknn
  - `videozip()` — drop-in replacement for `omnizip()`

- **`OmniZip-main/demo_videozip.py`** — runtime monkey-patch demo
  - Patches `omnizip_units.omnizip` before model loads
  - Model's lazy `from omnizip.omnizip_units import omnizip` (line 2554) picks up dispatcher
  - Flags: `--guide_mode {video,audio}`, `--beta`, `--rho_audio`, `--rho_video`

### Plan and memory

- **`videozip_plan.md`** — 11-section ultra-plan with function specs, ablations, paper section mapping
- **`~/.claude/projects/.../memory/MEMORY.md`** — index updated with VideoZip pointer

---

## 4. Pivot — ViT Feature-Norm Saliency for Video Pruning

User pointed out: OmniZip's video pruning has AUC ~0.52 (random). The ViT encoder already runs and computed self-attention. **Use ViT's own signal for video token selection** — surgical change, no guidance direction swap.

**Key clarification — VideoZip vs ViT Saliency:**

| | OmniZip | VideoZip | ViT Saliency |
|---|---|---|---|
| What guides audio | Audio attention | Same | Same (unchanged) |
| What guides video ratios | Audio retention | Video saliency | Audio retention (unchanged) |
| **How patches are picked** | dpcknn diversity | dpcknn diversity + audio sim | **dpcknn diversity + ViT feature norm** |
| Extra compute | — | — | `video_feature.norm(dim=-1)` ≈ 0 |

**The signal:** `vit_saliency[i] = ||video_feature[i]||₂` normalized to [0,1]. In transformer encoders, patches with heavy attention updates accumulate higher-norm representations.

**Combined dpcknn score:**
```
score = (1 - gamma) * diversity + gamma * vit_saliency
```
- gamma=0 → pure OmniZip
- gamma=1 → equal blend (default)
- gamma>1 → saliency-dominant

### Files Created

- **`OmniZip-main/omnizip/vitsaliency_units.py`**
  - `compute_vit_saliency()` — global L2 norm normalization
  - `compute_vit_saliency_per_frame()` — per-frame normalization variant
  - `omnizip_istm_vit_guided()` — modified ISTM with feature-norm term
  - `omnizip_vit()` — drop-in replacement for `omnizip()`

- **`OmniZip-main/demo_vitsaliency.py`**
  - Flags: `--gamma`, `--per_frame_norm`

---

## 5. Remote Execution (armaan@10.244.120.178)

**Path:** `/data/armaan/purs/OmniZip-main/`

### Issues encountered + fixes

1. **`ModuleNotFoundError: qwen_omni_utils`**
   - Package is in `qwen-omni-utils/src/` (src layout)
   - Fix: `export PYTHONPATH=/data/armaan/purs/OmniZip-main/qwen-omni-utils/src:$PYTHONPATH`
   - `pip install -e qwen-omni-utils/` failed due to `.gitignore` UTF-8 error in hatchling

2. **`No space left on device` during HF download**
   - Model exists locally at `/data/armaan/models/Qwen2.5-Omni-7B`
   - Fix: `sed -i 's|model_path = "Qwen/Qwen2.5-Omni-7B"|model_path = "/data/armaan/models/Qwen2.5-Omni-7B"|g' demo_vitsaliency.py demo_videozip.py`
   - Plus: `export HF_HOME=/data/armaan/hf-cache; export HF_HUB_OFFLINE=1`

### Successful runs

```bash
python demo_vitsaliency.py --gamma 1.0   # ViT saliency enabled
python demo_vitsaliency.py --gamma 0.0   # OmniZip baseline (pure dpcknn)
```

Both produced coherent video descriptions of the example clip (woman at desk showing MagSafe accessories). Qualitative outputs nearly identical — meaningful comparison requires benchmark eval.

---

## 6. What We're Measuring

**Core question:** Does keeping semantically important video patches (by ViT feature norm) preserve more task-relevant information than keeping spatially diverse patches (by dpcknn), at the same token budget?

**Failure mode addressed:** OmniZip's dpcknn picks maximally spread-out patches — blind to which patches matter for the question. A boring background corner gets kept if it's spatially isolated; the main subject gets dropped if surrounded by similar patches.

**Hypothesis:** ViT feature norm is a proxy for "how much information did the encoder write into this patch." High-norm patches should be more likely to contain answer-relevant content.

**Ablation that matters:**

| gamma | Selection criterion | Expected vs baseline |
|---|---|---|
| 0.0 | dpcknn only (OmniZip) | baseline |
| 0.5 | mild saliency bias | +? |
| 1.0 | equal blend | +? |
| 2.0 | saliency-dominant | +? or overfit |

**Positive result:** VideoMME accuracy goes up at gamma=1.0 vs gamma=0.0, no token budget change.

---

## 7. Next Step (not yet done)

User asked whether to write `eval_qwen_omni_vitsaliency.py` that mirrors the existing `eval_qwen_omni_zip.py` benchmark pipeline — paused before proceeding with that.

---

## Key Technical Decisions Log

- **Why feature norm vs ViT attention hooks:** ViT in Qwen2.5-Omni doesn't expose attention weights (forward returns only `torch.Tensor`). Feature norms work without hooks, with zero extra compute, and are already in memory when `omnizip()` runs.
- **Why monkey-patch instead of editing model file:** User explicitly said "don't touch official files." Lazy `from omnizip.omnizip_units import omnizip` inside model's `forward()` (line 2554) re-resolves every call, so patching `_units.omnizip` before generate() is a clean intercept point.
- **Why per-group audio (VideoZip) vs global audio (OmniZip):** Video saliency varies temporally — uniform audio compression would waste tokens in low-saliency windows.
- **Why beta=0.3 default for audio anchor in ISTM:** Diversity should remain primary signal; audio is secondary cross-modal nudge.
