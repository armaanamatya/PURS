# Mechanistic Interpretability — Tier-1 Depth Curve Session

**Dates:** 2026-05-03 → 2026-05-07
**Project:** Qwen2.5-Omni Thinker mechanistic interpretability (PURS)
**Status at session end:** `depth_curve_batch.py` written locally, ready to sync + run on remote for the N=10 audio + N=10 video robustness check.

---

## 1. Premise

User asked whether the experiments in `Mechanistic Layer-Level Analysis for Multimodal LLMs in the Qwen2.5 Qwen3 Thinker–Talker Architecture.md` are actually doable, given the local repo state and Lambda instance hardware.

**Verdict:** Yes, much more so than the source doc implies — most of the hardest infra (instrumented Thinker, lmms-eval harness, multi-GPU box) already exists in the repo.

Wrote `docs/mech_interp_feasibility.md` with a 3-tier roadmap (11 experiments). Tiers summarized:

- **Tier 1 (1–2 weeks, single GPU):** quick wins on existing infra
  - #1 per-layer modality attention depth curves
  - #2 layer ablation curves (4 benchmarks)
  - #3 layer-wise linear probes
  - #4 OmniZip-importance vs probe-importance overlay
- **Tier 2 (3–5 weeks):** causal claims via patching (AV-conflict dataset, cross-modal patching, TMRoPE temporal patching, attention knockout)
- **Tier 3 (6+ weeks, cluster):** Talker dissection, Qwen3-Omni MoE router probes, MechZip (compression coupled with mech interp)

---

## 2. Discovered existing state

User has substantial existing infrastructure that subsumes some Tier-1 work:

| existing file (repo root on remote) | Tier-1 step covered |
|---|---|
| `viz_attention_depth_curve.py` | #1 (cross-modal flow + Gini + last-token across all 28 layers) |
| `viz_attention_omnizip.py` | #1 variant (with OmniZip drop overlay) |
| `viz_layer_depth_experiment.py` | partial #3 (Cross-Q Spearman + Gini at 8 depth checkpoints, multi-GPU batch driver) |
| `viz_early_layer_relevance_batch.py` | partial #3 (layers 0–1 only, batched over metadata.json) |
| `viz_crossmodal_spearman.py` + `analyze_crossmodal_vs_omnizip.py` | #4 done |
| `eval_qwen_omni.py` | importable `run_inference()` and `load_model()` — usable harness for #2 |

**Key lesson learned (twice):** check existing scripts before writing infrastructure from scratch. User added a feedback memory: vendored `*-main/` paper trees are read-only; treat repo's own `scripts/` (and root) similarly — read first.

---

## 3. Work done in `mechinterp/`

### 3.1 First attempt — deleted

Wrote `mechinterp/modality_attention.py` as a fresh Tier-1 #1 script. After looking around, discovered `viz_attention_depth_curve.py` already exists and is more comprehensive (5 plots vs my 4). **Deleted.**

### 3.2 Move viz_attention_depth_curve.py into `mechinterp/`

- Recovered the file from git (`HEAD` commit `09aaf67c` — was deleted in working tree on `mds` branch).
- Patched paths: `SCRIPT_DIR = dirname(__file__)`, `REPO_ROOT = dirname(SCRIPT_DIR)`, so `OMNIZIP_DIR` and `videos/` resolve via the parent directory.
- Default `OUT_DIR` retargeted to `mechinterp/outputs/depth_curve/`.
- Added a `run_meta.json` sidecar dump so each output folder is self-describing (video, question, fps, max_frames, max_pixels, timestamp).
- Mirrored the 5 historical PNGs from `vizzing/depth_curve/` into `mechinterp/outputs/depth_curve/`.

Final layout in `mechinterp/`:

```
mechinterp/
├── viz_attention_depth_curve.py
├── depth_curve_batch.py        # written but not yet run
├── README.md                    # documents script + outputs + Tier-1 status table
└── outputs/
    ├── depth_curve/             # vision-leaning (worldsense / attribute_reasoning)
    │   ├── 1_crossmodal_attention_curves.png
    │   ├── 2_self_vs_cross_attention.png
    │   ├── 3_attention_gini_depth.png
    │   ├── 4_crossmodal_heatmap.png
    │   └── 5_last_token_attention_depth.png
    └── depth_curve_audio_clip/  # audio-leaning (daily-omni / av_event_alignment)
        └── (same 5 PNGs)
```

### 3.3 Single-clip runs + analysis

**Run 1 — vision-leaning (worldsense / "man with beard wearing suit"):**

| layer | Audio | Video | Text |
|---|---|---|---|
| 0 | 20% | 10% | 70% |
| 4 | 4% | 4% | 92% |
| 5–25 | 2–5% | 4–10% | 85–95% |
| 26–27 | 3% | 8% (slight uptick) | ~89% |

**Run 2 — audio-leaning (daily-omni / "audio event synced with Obsidian Fury static shot"):**

OOM'd twice on the vision encoder before fixing:
1. First crash: 118 frames at default sampling → 8496 video tokens → encoder OOM allocating 34 GB.
2. Second crash: capped frames but ghost processes from earlier sessions held 21 GB on GPU 0 (leftover from interrupted runs). Pinned to free GPU 5 via `CUDA_VISIBLE_DEVICES=5`.
3. Successful run: `--max_frames 32 --fps 2 --max_pixels 200704`, T=3880 (1500 audio + 2304 video + 76 text).

| layer | Audio | Video | Text |
|---|---|---|---|
| 0 | **28%** | 4% | 68% |
| 1–3 | 21–26% | 4–7% | varied |
| 4 | 5% | 3% | 92% |
| 5–25 | 2–8% | 1–5% | ~92% |
| 26–27 | **10%** | **13%** | ~78% |

### 3.4 Three findings from N=2

1. **Layer-0 modality routing is question-conditional.** Audio share at L0 jumps from 20% → 28% (+40% rel) when the question is audio-leaning; video drops 10% → 4%. The model is *already* prompt-aware at the first layer — not blind perceptual integration.

2. **Layer-4 collapse is robust to question type.** Both clips drop to <10% combined modality share by L4. Whatever fusion mechanism Qwen-Omni uses, it's structurally locked to layers 0–3.

3. **NEW — late-layer re-engagement specific to audio-temporal-sync question.** L26–27 audio bumps to 10% and video to 13%, totaling ~22% non-text share — a >2× jump from L25. Vision-leaning clip showed only a modest video-only late bump. **Hypothesis:** tasks requiring precise temporal binding trigger a "verification re-read" at the deepest layers. Worth chasing — distinct circuit from the early-integration zone.

Cross-checks against the other 4 plots (cross-modal flow, self-vs-cross, Gini, heatmap) confirmed:
- `audio→text` dominates `video→text` on the audio-leaning clip; reverse on vision-leaning. Routing is genuinely question-conditional.
- Audio's special L0 self-attention phase (~40% self) is structural, identical across clips.
- Audio attention is more concentrated than video (Gini ~0.90 vs ~0.85) at every depth — model finds discrete events in audio while spreading attention across video frames.

### 3.5 Deleted attempts (prune list)

- `mechinterp/modality_attention.py` — redundant with `viz_attention_depth_curve.py`. Deleted.
- First version of `mechinterp/depth_curve_batch.py` — written without checking existing batch infrastructure (`viz_layer_depth_experiment.py`). Deleted, rewrote second version using its multi-GPU sharded-model-load pattern.

### 3.6 Final batch script written

`mechinterp/depth_curve_batch.py` — batch depth-curve over N audio-leaning + N video-leaning clips:

- **Multi-GPU model load** via `device_map="auto"` + `max_memory={i: "14GiB"}` (lifted from `viz_layer_depth_experiment.py`).
- **`resolve_video_path`** with Windows-backslash + recursive-glob fallback (lifted).
- **Streaming hook** capturing compact per-layer stats and immediately returning `None` for the attention tensor (lifted from `viz_attention_depth_curve.py` lines 187–211 — bounds memory at one layer's attention at a time).
- **Eager attention** (`attn_implementation="eager"`) — required for `output_attentions=True`. We deliberately did **not** adopt the Q/K-projection trick from `viz_layer_depth_experiment.py` because that bypasses RoPE and produces a different metric than the existing depth-curve plots; metric-compatibility matters here.
- **Per-clip JSON + aggregate-at-end** (lifted pattern).
- **Side-by-side group plots:** `group_last_token.png` (stacked area, audio vs video group), `group_crossmodal.png` (audio→text & video→text overlaid, mean ± 1σ).

Defaults: `--gpus 2,3 --n_audio 10 --n_video 10 --audio_datasets daily-omni --video_datasets video-mme --max_frames 32 --fps 2 --max_pixels 200704`.

---

## 4. Important environment notes

### 4.1 Model checkpoint path
**Local Linux path:** `/data/armaan/models/Qwen2.5-Omni-7B`. Not in HF cache by default — pointing scripts at `Qwen/Qwen2.5-Omni-7B` triggers a 5-shard download. Always pass `--model_path /data/armaan/models/Qwen2.5-Omni-7B` (or accept it as default in the new scripts).

### 4.2 GPU contention
Lambda box has 8× RTX 6000 Ada (48 GB each). Recurring issue: ghost Python processes from interrupted runs hold 20–30 GB on GPUs 0/1. Always `nvidia-smi` first; pin to free GPUs via `CUDA_VISIBLE_DEVICES=N` or `--gpus a,b`.

### 4.3 Vision encoder OOM
At default frame sampling, long videos (>60 s) yield 100+ frames → 8000+ video tokens → encoder full-attention pass OOMs at 34 GB. Mitigation: `--max_frames 32 --fps 2 --max_pixels 200704` (=448²). The Thinker forward at T~3900 with eager attention is fine; the encoder is the bottleneck.

### 4.4 Local vs remote sync state
- Local Windows working tree (branch `mds`) has many `D` (deleted) files relative to `HEAD`, including the `viz_*` scripts. They live at the repo root on the remote, were "moved" into `scripts/` locally.
- `mechinterp/modality_attention.py` was deleted locally but still present on remote at session end — needs `rm` after next sync.
- `mechinterp/outputs/per_clip/` is leftover from the deleted modality_attention.py's dry-run — also needs cleanup on remote.

### 4.5 Datasets in `videos/metadata.json`
Only three: `daily-omni`, `video-mme`, `worldsense`. `omnivideobench/` directories are present but empty — videos not downloaded.

---

## 5. Open questions / decisions

1. **Layer-ablation script (`layer_ablation.py`, Tier-1 #2)** — pending the batch robustness check. If ≥2 of the 3 hypotheses (L4-collapse-robust, L0-question-conditional, L26-27-audio-specific) hold, write it; otherwise refine hypothesis first.
2. **Late-layer re-engagement N=1.** The L26-27 uptick is from one audio-temporal-sync clip. The batch run will tell us whether it generalizes across other audio-leaning questions or was idiosyncratic.
3. **Other `scripts/` candidates to move into `mechinterp/`** (deferred until first ablation lands):
   - `viz_attention_omnizip.py` — Tier-1 #1b (canonical with OmniZip drop overlay)
   - `viz_attention_encoders.py` — perception-stack mech interp
   - `viz_crossmodal_spearman.py` + `analyze_crossmodal_vs_omnizip.py` — Tier-1 #4 pair
   - `viz_early_layer_relevance.py`, `viz_early_layer_relevance_batch.py` — partial #3
   - `viz_layer_depth_experiment.py` — keep at root for now (it's the batch driver template)
   - `viz_tmrope.py` — structural, useful for Tier-2 temporal patching design
4. **Whether to pull the freshest `vizzing/depth_curve/` outputs from remote** rather than relying on the local copies that we mirrored at session start. Existing single-clip outputs in `mechinterp/outputs/depth_curve/` came from local Windows; remote may have regenerated them since.

---

## 6. Resume checklist (next session)

```bash
# 1. Push the batch script to remote
scp mechinterp/depth_curve_batch.py armaan@10.244.120.178:/data/armaan/purs/mechinterp/

# 2. Clean stale state on remote
ssh armaan@10.244.120.178 "cd /data/armaan/purs && \
    rm -f mechinterp/modality_attention.py && \
    rm -rf mechinterp/outputs/per_clip"

# 3. Check GPUs and run
ssh armaan@10.244.120.178
cd /data/armaan/purs
nvidia-smi   # find free GPUs (typically 2-7)
python mechinterp/depth_curve_batch.py \
    --gpus 2,3 --n_audio 10 --n_video 10 \
    --audio_datasets daily-omni \
    --video_datasets video-mme

# 4. Pull results when done
exit
scp -r armaan@10.244.120.178:/data/armaan/purs/mechinterp/outputs/depth_curve_batch \
       mechinterp/outputs/

# 5. Read figures/group_last_token.png, figures/group_crossmodal.png, summary.json
#    Decide: write layer_ablation.py, or refine hypothesis.
```

Estimated time: 15–20 min on 2 free GPUs (one model load + 20 forward passes at ~30–60s each).

---

## 7. Files written / modified this session

- `docs/mech_interp_feasibility.md` (new — 3-tier plan + per-section feasibility table)
- `mechinterp/viz_attention_depth_curve.py` (recovered from git + path-patched + run_meta sidecar)
- `mechinterp/README.md` (Tier-1 status table, output layout, what-to-look-for)
- `mechinterp/depth_curve_batch.py` (new — batch robustness check)
- `mechinterp/outputs/depth_curve/` + `mechinterp/outputs/depth_curve_audio_clip/` (5 PNGs each)
- `mechinterp/modality_attention.py` (created and deleted — redundant)
- `chats/mechinterp_tier1_depth_curve_session.md` (this file)

End of session.
