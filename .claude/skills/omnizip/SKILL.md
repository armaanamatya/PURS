---
name: omnizip
description: |
  Training-free audio-guided token compression for Qwen2.5-Omni omnimodal LLMs.
  Use when: (1) running faster inference on audio-video inputs with OmniZip enabled,
  (2) evaluating OmniZip on benchmarks (VideoMME, etc.) via lmms-eval,
  (3) tuning rho_audio/rho_video compression ratios for speed vs. quality tradeoffs,
  (4) setting up the OmniZip environment from scratch.
  Achieves 3.42x inference speedup and 1.4x memory reduction over baseline Qwen2.5-Omni.
---

## Installation

```bash
git clone https://github.com/KD-TAO/OmniZip.git
cd OmniZip
conda create -n omnizip python=3.10 -y
conda activate omnizip
pip install --upgrade pip
bash setup.sh
cd lmms-eval && pip install -e . && cd ..
pip install flash-attn --no-build-isolation
# Recommended: pip install torch==2.6.0 torchvision==0.21.0
```

## Quick start

```bash
# Basic demo (uses assets/example.mp4)
python demo.py --omnizip

# With custom compression ratios
python demo.py --omnizip --rho_audio 0.4 --rho_video 0.7

# Custom video and prompt
python demo.py --omnizip --video path/to/video.mp4 --describe "What happens in this video?"
```

## Key parameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `--rho_audio` | 0.4 | 0–1 | Audio token merging ratio — higher = more compression |
| `--rho_video` | 0.7 | 0–1 | Video token pruning ratio — higher = more compression |
| `--g` | 3 | int | Temporal group size for interleaved spatio-temporal compression |
| `--contextual_ratio` | 0.05 | 0–1 | Fraction of tokens kept for cross-modal context anchoring |

**Tradeoff guide:**
- Paper settings: `rho_audio=0.4, rho_video=0.7` — balanced speed/quality
- Eval.sh defaults: `rho_audio=0.3, rho_video=0.6` — slightly more conservative
- Higher `rho_video` → faster inference, possible quality drop on dense visual scenes
- `contextual_ratio` below 0.03 risks losing cross-modal alignment cues

## Programmatic usage (Python)

```python
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

omnizip_config = {
    "rho_audio": 0.4,
    "rho_video": 0.7,
    "g": 3,
    "contextual_ratio": 0.05,
}

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    torch_dtype="auto",
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)
model.thinker.omnizip_config = omnizip_config
model.thinker.nframes = num_input_frames  # set after processing video
```

Key: import from `omnizip.modeling_qwen2_5_omni`, not from `transformers`, to enable compression.

## Video input settings

Tunable in the source (affects VRAM and throughput):

```python
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28  # paper uses 128*28*28 for lower VRAM
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
```

Lower `VIDEO_MAX_PIXELS` (e.g., `128*28*28`) reduces VRAM significantly with minor quality trade-off.

## Evaluation

For benchmark evaluation via lmms-eval, see [references/eval-benchmarks.md](references/eval-benchmarks.md).

## How it works (brief)

1. **Audio saliency** — identifies salient audio tokens via token-level scoring
2. **Retention scoring** — computes per-time-group audio retention score (information density)
3. **Video pruning** — audio scores dynamically guide which video tokens to prune
4. **Cross-modal anchoring** — preserves audio-anchor-enhanced video tokens via cross-modal similarity
5. **Spatio-temporal compression** — remaining video tokens compressed with interleaved scheme per window
