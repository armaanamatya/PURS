# mechinterp/

Mechanistic-interpretability experiments on the Qwen2.5-Omni Thinker, scoped
per the Tier-1 → Tier-3 plan in `docs/mech_interp_feasibility.md`.

## Status

| Tier-1 step | status | script |
|---|---|---|
| #1 per-(layer) cross-modal attention depth curves | **done** | `viz_attention_depth_curve.py` (moved here from repo root) |
| #1b per-(layer) modality attention with OmniZip drop overlay | done (lives at repo root) | `../viz_attention_omnizip.py` |
| #2 layer ablation curves on text/audio/video/AV benchmarks | **TODO — next** | `layer_ablation.py` |
| #3 layer-wise linear probes | TODO; partial overlap with `../viz_early_layer_relevance*.py` | — |
| #4 OmniZip-importance vs probe-importance | done (lives at repo root) | `../analyze_crossmodal_vs_omnizip.py` |

## Scripts

### `viz_attention_depth_curve.py` — Tier-1 #1 (done)

Single-video diagnostic: hooks every Thinker `self_attn`, streams compact
modality statistics layer-by-layer, and produces five plots in
`outputs/depth_curve/`:

1. `1_crossmodal_attention_curves.png` — 6 directional cross-modal curves
   (audio↔video, audio↔text, video↔text) across all 28 decoder layers.
2. `2_self_vs_cross_attention.png` — per-modality self-vs-cross ratio per layer.
3. `3_attention_gini_depth.png` — attention concentration (Gini) per modality.
4. `4_crossmodal_heatmap.png` — layer × modality-pair heatmap (max per row marked `*`).
5. `5_last_token_attention_depth.png` — stacked-area share of the last token's
   attention across audio/video/text per layer (the generation-relevant view).

#### Run

```bash
# default video (worldsense / attribute_reasoning) and question
python mechinterp/viz_attention_depth_curve.py

# custom video + question + lighter sampling
python mechinterp/viz_attention_depth_curve.py \
    --video_path videos/video-mme/action_reasoning/video.mp4 \
    --question  "Which of the following reasons motivated the archaeologists to excavate the tomb?" \
    --fps 1 --max_pixels 200704 --max_frames 16

# pin output dir
python mechinterp/viz_attention_depth_curve.py --out_dir mechinterp/outputs/depth_curve
```

#### Defaults assume

- Repo layout `<repo_root>/mechinterp/viz_attention_depth_curve.py`, with
  `OmniZip-main/`, `videos/`, etc. as siblings of `mechinterp/`.
- `MODEL_PATH = /data/armaan/models/Qwen2.5-Omni-7B` (override with `--model_path`).
- Uses **stock HF** `Qwen2_5OmniForConditionalGeneration` (not OmniZip-instrumented),
  so no token pruning interferes with the raw attention analysis.
- `attn_implementation="eager"` because FlashAttention-2 doesn't materialize
  attention weights.
- Single GPU (`cuda:0`); ~16–20 GB VRAM at the default frame budget.

#### What to look for

The headline plot is **`5_last_token_attention_depth.png`** — it answers
"what is the model attending to as it prepares to generate?" If audio share
collapses with depth while video/text rise, that replicates the audio-suppression
finding from arXiv 2604.02605. If audio share holds, Qwen-Omni is a counter-example
and Tier-2 patching should chase *why* (TMRoPE? AuT-style audio pretraining?).

`1_crossmodal_attention_curves.png` is the per-layer mechanism (which directional
flows carry the change).

## Output folder: `outputs/`

```
mechinterp/outputs/
└── depth_curve/
    ├── 1_crossmodal_attention_curves.png
    ├── 2_self_vs_cross_attention.png
    ├── 3_attention_gini_depth.png
    ├── 4_crossmodal_heatmap.png
    └── 5_last_token_attention_depth.png
```

The five PNGs from the original run (when this script lived at the repo root)
have been copied here. The historical copies remain at `../vizzing/depth_curve/`
for reference.

## Caveats

- **Single-video**: this run is for one (video, question) pair. To get a
  population-level picture, sweep over `videos/metadata.json` items (planned
  follow-up; the loop logic from the deleted `modality_attention.py` is the
  template).
- **Attention mass ≠ causal importance.** A modality can carry high attention
  mass without being load-bearing for the answer. Tier-2 cross-modal patching
  (planned in `cross_modal_patching.py`) is the causal version.

## Next: `layer_ablation.py`

Tier-1 #2. Zero each Thinker layer's residual contribution one at a time;
measure ΔAccuracy on text-only (MMLU subset), audio-only (LibriSpeech-clean),
video-QA (Video-MME subset), and audio-video (OmniBench / WorldSense). The
existing `eval_qwen_omni*.py` harnesses give the benchmark loop; the hook
adds a layer-wise residual zero. Estimated 1–2 days to a first plot.
