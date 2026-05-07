# VideoZip

Training-free, video-guided audio token compression for Qwen2.5-Omni, with bidirectional
cross-modal anchoring and L6-cached video saliency. The inverse of OmniZip's
audio-guided-video paradigm.

This folder is the canonical home for all VideoZip work. **Existing OmniZip code is imported,
not copied** — `videozip/__init__.py` injects `OmniZip-OG/omnizip/` onto `sys.path` so
modules like `omnizip_units` are importable directly.

## Folder layout

```
videozip/
├── README.md                  this file
├── __init__.py                sys.path bootstrap; do not skip importing this first
├── src/                       new VideoZip code (no copies of OmniZip)
│   ├── _utils.py              ratio mapping, audio group builder
│   ├── video_saliency.py      §4a / §4a-L6 / §4a-Sim — three saliency variants
│   ├── audio_compress.py      §4b — per-group audio compression
│   ├── istm_audio_anchored.py §4c — audio-guided ISTM (dpcknn + audio anchor)
│   └── videozip.py            §4d — main orchestrator
├── docs/                      pointers to authoritative docs (no duplication)
├── precompute/                video-side L6 cache generation (mirrors existing audio script)
├── eval/                      benchmark eval harness (mirrors eval_qwen_omni_zip_cached.py)
├── ablations/                 Ablation A–E runners (one script per ablation)
├── cache/                     output JSONL files for L6 video saliency (gitignored content)
└── tests/                     unit tests for saliency + utils
```

## Authoritative references (NOT duplicated here)

| Doc | Location |
|---|---|
| Plan (functional spec) | `docs/videozip_plan.md` |
| 3-way comparison | `docs/videozip_comparison.md` |
| L6 cache experimental record | `docs/l6cache.md` |
| Future planning context | `docs/futureplanning.md` |
| Project briefing | `docs/PROJECT_BRIEFING.md` |
| Original OmniZip code | `OmniZip-OG/omnizip/` |
| L6 audio cache script | `OmniZip-OG/omnizip/precompute_l6_saliency.py` |
| L6 audio cache eval | `OmniZip-OG/omnizip/eval_qwen_omni_zip_cached.py` |
| Cached audio saliency JSONL | `vizzing/layer_depth_all_full.jsonl` |

## How to import from this package

```python
# bootstraps sys.path so OmniZip imports resolve
import videozip

from videozip.src.videozip import omnizip_videozip
from videozip.src.video_saliency import (
    omnizip_video_saliency,
    omnizip_video_saliency_l6,
    omnizip_video_saliency_simonly,
)
```

## Implementation order (revised given existing infra)

**Discovery during scaffolding:** the existing `precompute_l6_saliency.py` (project root)
already dumps BOTH audio and video L6 scores into `vizzing/layer_depth_all_full.jsonl`.
The `_cross()` hook is called for audio AND video positions in the same forward pass,
and both vectors are written under `raw_audio_scores` / `raw_video_scores` per record.
**No new precompute is needed** — we just need a consumer.

1. **`cache/loader.py`** — DONE. Reads the existing JSONL, returns per-video score
   vectors. Pure CPU.
2. **`precompute/validate_l6_cache.py`** — DONE. Remote-runnable validator that
   confirms every metadata video has a populated `raw_video_scores[layer]` entry
   before any model run. Run this first on remote.
3. **`src/video_saliency.py`** — DONE. Three saliency variants + dispatcher. The
   `omnizip_video_saliency_l6` consumes the 1D Q*K score vector from the cache
   directly (NOT hidden states, NOT L2 norms).
4. **`src/_utils.py`**, **`src/audio_compress.py`**, **`src/istm_audio_anchored.py`** — DONE.
5. **`src/videozip.py`** orchestrator — DONE except for the `merge_plan` application
   step (TODO marker; lives in modeling_qwen2_5_omni.py).
6. **`eval/eval_videozip.py`** — NEXT. Monkey-patch `omnizip()` to dispatch to
   `omnizip_videozip()` when `omnizip_config["guide_mode"] == "video"`. Pattern:
   `eval_qwen_omni_zip_cached.py`. This is also where the merge_plan integration
   lands.
7. **`ablations/ablation_*.py`** — Final. One script per Ablation A–E from the plan.

## Constraints honored

- **No deletions.** All existing files in `OmniZip-OG/`, `ablationscripts/`, `vizzing/`,
  `docs/`, etc. remain untouched.
- **No copies of upstream code.** OmniZip is imported, not duplicated.
- **No edits to upstream paper trees.** Per memory rule
  `feedback_never_edit_paper_source_trees.md`.
