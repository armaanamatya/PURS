# OmniDrop Qwen2.5-Omni Integration Notes

This document records the practical integration work done after the minimal
paper2code reproduction. The original `omnidrop/` package remains a pure NumPy
implementation of the paper mechanisms; the GPU runner wires those mechanisms
into the local Qwen2.5-Omni evaluation stack.

## What Was Added

- `omnidrop/omnidrop/torch_pruner.py`
  - Torch runtime helper for live Qwen prefill.
  - Implements Sec. 3.4 pre-LLM intra-modality pruning:
    - audio: top-saliency keep ratio, default `0.7`
    - video: Dycoke-style TTM, default group size `4`, prune rate `0.8`
  - Implements decoder-layer OmniDrop state:
    - identifies audio/video token positions from `input_ids`
    - keeps non-AV text/system tokens
    - computes text-query to AV attention importance
    - applies PLP schedule and TDS
    - records per-layer trace rows

- `OmniZip-main/omnizip/modeling_qwen2_5_omni.py`
  - Patched local copied Qwen model used by the existing OmniZip eval path.
  - Adds `model.thinker.omnidrop_config`.
  - Runs pre-prune after audio/video embeddings are scattered into
    `inputs_embeds` and before decoder prefill.
  - Infers video pre-prune frame groups from Qwen's `video_grid_thw` temporal
    axis, not from raw decoded frame count.
  - Runs layer-wise OmniDrop inside `Qwen2_5OmniThinkerTextModel.forward`.
  - Slices `hidden_states`, `position_ids`, `cache_position`, causal mask, and
    RoPE embeddings after each pruning step.
  - Stores summaries in:
    - `model.thinker.omnidrop_pre_prune_stats`
    - `model.thinker.model.omnidrop_last_trace`

- `eval_qwen_omni_omnidrop.py`
  - Dedicated eval runner matching the existing full-token and OmniZip runners.
  - Outputs normal prediction JSONL plus OmniDrop pre-prune and layer-prune
    diagnostics.

## Remote Baseline Already Run

Remote machine:

```text
armaan@10.244.120.178
/data/armaan/purs
conda env: omnizip_clean
GPU: 8x RTX 6000 Ada, ~49 GB each
```

Sanity tests:

```bash
PYTHONPATH=/data/armaan/purs/omnidrop python -m pytest /data/armaan/purs/omnidrop/tests/ -q
```

Result:

```text
23 passed
```

Baseline slice:

```text
metadata: videos/metadata.json
videos: videos
category: "Inference"
questions: 19
```

Results:

| Method | Accuracy | Mean Peak Alloc | Mean Peak Reserved | Mean Prefill | Mean E2E |
|---|---:|---:|---:|---:|---:|
| Full Qwen2.5-Omni | 12/19 = 63.16% | 19.75 GB | 28.55 GB | 1885.68 ms | 1921.05 ms |
| OmniZip | 13/19 = 68.42% | 18.21 GB | 20.84 GB | 1224.59 ms | 1246.29 ms |
| OmniDrop decoder trace run | 13/19 = 68.42% | 19.62 GB | 24.31 GB | 1170.70 ms | 1155.22 ms |

These are controls for the OmniDrop run.

The first OmniDrop trace run showed decoder-layer pruning working, but the
`pre_*` fields were absent. That exposed an over-strict pre-prune guard and a
raw-frame/video-grid mismatch. The current code fixes both issues; rerun after
copying the files below and verify `pre_prune_applied` plus the `pre_*` counts.

## Copy Changed Files To Remote

From local Windows PowerShell:

```powershell
scp "C:\Users\Armaan\Desktop\PURS\eval_qwen_omni_omnidrop.py" armaan@10.244.120.178:/data/armaan/purs/
scp "C:\Users\Armaan\Desktop\PURS\omnidrop\omnidrop\torch_pruner.py" armaan@10.244.120.178:/data/armaan/purs/omnidrop/omnidrop/
scp "C:\Users\Armaan\Desktop\PURS\OmniZip-main\omnizip\modeling_qwen2_5_omni.py" armaan@10.244.120.178:/data/armaan/purs/OmniZip-main/omnizip/
```

Quick import check on remote:

```bash
cd /data/armaan/purs
python - <<'PY'
import eval_qwen_omni_omnidrop
from omnidrop.torch_pruner import OmniDropTorchState, omnidrop_pre_prune
print("omnidrop runner import ok")
PY
```

## Run Full OmniDrop Pipeline

Pre-prune is enabled by default.

```bash
cd /data/armaan/purs

CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_omnidrop.py \
  --model /data/armaan/models/Qwen2.5-Omni-7B \
  --metadata videos/metadata.json \
  --videos videos \
  --category "Inference" \
  --output runs/omnidrop_prep/omnidrop_daily_inference.jsonl \
  --log runs/omnidrop_prep/omnidrop_daily_inference.log \
  --vram_log runs/omnidrop_prep/omnidrop_daily_inference_vram.jsonl \
  --stderr_log runs/omnidrop_prep/omnidrop_daily_inference_stderr.log \
  --errors_log runs/omnidrop_prep/omnidrop_daily_inference_errors.log \
  --dtype bfloat16 \
  --max_new_tokens 16 \
  --measure_prefill
```

Decoder-only ablation:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_omnidrop.py \
  --model /data/armaan/models/Qwen2.5-Omni-7B \
  --metadata videos/metadata.json \
  --videos videos \
  --category "Inference" \
  --output runs/omnidrop_prep/omnidrop_decoder_only_daily_inference.jsonl \
  --log runs/omnidrop_prep/omnidrop_decoder_only_daily_inference.log \
  --vram_log runs/omnidrop_prep/omnidrop_decoder_only_daily_inference_vram.jsonl \
  --stderr_log runs/omnidrop_prep/omnidrop_decoder_only_daily_inference_stderr.log \
  --errors_log runs/omnidrop_prep/omnidrop_decoder_only_daily_inference_errors.log \
  --dtype bfloat16 \
  --max_new_tokens 16 \
  --measure_prefill \
  --no_pre_prune
```

## Output Files

- `*_inference.jsonl`
  - prediction records
  - correctness
  - frame counts
  - timings
  - OmniDrop trace summaries

- `*_vram.jsonl`
  - per-question VRAM
  - timings
  - pre-prune stats
  - decoder pruning summaries

- `*.log`
  - human-readable progress and final accuracy

- `*_errors.log`
  - Python tracebacks for failed questions

- `*_stderr.log`
  - backend warnings from video/audio/model libraries

## Important Logged Fields

Pre-LLM Sec. 3.4:

- `pre_prune_applied`
- `pre_audio_has_attention`
- `pre_audio_before`
- `pre_audio_after`
- `pre_video_before`
- `pre_video_after`
- `pre_video_frame_count`
- `pre_video_tokens_per_frame`
- `pre_video_skip_reason` when video TTM fails open
- `pre_seq_before`
- `pre_seq_after`

Decoder-layer OmniDrop:

- `omnidrop_layers`
- `omnidrop_pruned_total`
- `omnidrop_seq_start`
- `omnidrop_seq_end`
- `omnidrop_av_start`
- `omnidrop_av_end`
- `omnidrop_mean_av_retained`
- `omnidrop_final_av_retained`
- `omnidrop_trace`

Quick field check:

```bash
head -n 1 runs/omnidrop_prep/omnidrop_daily_inference_trace_vram.jsonl \
  | python -m json.tool \
  | grep -E "pre_|omnidrop_mean|omnidrop_final|omnidrop_av"
```

## Current Caveats

- This is an experimental Qwen integration, not a validated reproduction of the
  paper's benchmark numbers yet.
- Pre-prune audio saliency uses the audio encoder attention tensor exposed by
  the local OmniZip model copy, matching the reproduction assumption rather than
  an official OmniDrop implementation.
- Video TTM assumes video tokens can be evenly divided by Qwen's temporal video
  grid count. If not, it fails open for video and logs
  `pre_video_skip_reason`.
- Decoder pruning forces attention weights to be materialized during prefill so
  the score can be computed. This may reduce some of the raw FlashAttention
  memory/speed benefit, but it is necessary for query-guided pruning.
- KV cache is created after the progressively pruned prefill, so generation
  should continue on the compressed sequence. This has not yet been validated on
  the remote GPU run at the time this note was written.
- Batching is intentionally unsupported for OmniDrop pruning; the eval runner
  processes one sample/question at a time.

## Local Verification

After the integration edits:

```text
python -m py_compile eval_qwen_omni_omnidrop.py omnidrop/omnidrop/torch_pruner.py OmniZip-main/omnizip/modeling_qwen2_5_omni.py
PYTHONPATH=omnidrop python -m pytest omnidrop/tests -q
```

Result:

```text
23 passed
```
