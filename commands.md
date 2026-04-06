# Evaluation Run Commands

All commands assume you are in `/data/armaan/purs/` using a Python env with the same dependencies as the old `omnizip_pip` setup (recommended: **`omnizip_clean`** venv below).

---

## Python environment (`omnizip_clean`)

Create or activate the venv, then install pins. PyTorch CUDA wheels are **not** on PyPI alone—you must pass PyTorch’s wheel index:

```bash
source /data/armaan/venvs/omnizip_clean/bin/activate
pip install -r omnizip_pip_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

If the requirements file only exists next to your data dir: `pip install -r /data/armaan/omnizip_pip_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`

After `pip freeze` from conda, edit the file before install: remove `+cu124` from `torch` / `torchvision` / `torchaudio` lines; replace any `packaging @ file://...` line with `packaging==<version>` from `pip show packaging`.

---

## Sync eval code from your laptop to the instance

From the **PURS repo root** on your machine (Git Bash, WSL, or Linux). Replace the host if needed.

```bash
export REMOTE=armaan@10.244.120.178
export RDIR=/data/armaan/purs

rsync -avz --progress \
  eval_qwen_omni.py eval_qwen_omni_zip.py commands.md omnizip_pip_requirements.txt \
  "${REMOTE}:${RDIR}/"

rsync -avz --progress OmniZip-main/omnizip/ "${REMOTE}:${RDIR}/OmniZip-main/omnizip/"
rsync -avz --progress OmniZip-main/qwen-omni-utils/ "${REMOTE}:${RDIR}/OmniZip-main/qwen-omni-utils/"
```

Only these paths are required to run the two eval scripts (not the full `lmms-eval` tree).

**Windows (PowerShell)** if `rsync` is unavailable, use **scp**:

```powershell
scp eval_qwen_omni.py eval_qwen_omni_zip.py commands.md omnizip_pip_requirements.txt armaan@10.244.120.178:/data/armaan/purs/
scp -r OmniZip-main\omnizip armaan@10.244.120.178:/data/armaan/purs/OmniZip-main/
scp -r OmniZip-main\qwen-omni-utils armaan@10.244.120.178:/data/armaan/purs/OmniZip-main/
```

Prefer **`rsync` from Git Bash** (same flags as above); it preserves partial transfers and is clearer for trees.

---

## Run 2 — fixed folder layout (`run2/baseline` vs `run2/omnizip`)

Same outputs as timestamped `runs/…` folders, but under a single **`run2`** directory with separate logs for baseline vs OmniZip.

The eval scripts create missing parent directories for `--log` / `--output` / etc. **`--stderr_log`** captures stderr inside Python after those dirs exist. Avoid **`2>&1 | tee run2/.../stderr.log`** unless you **`mkdir -p run2/baseline`** first—`tee` opens its file before Python runs, so a missing directory makes `tee` fail.

```bash
source /data/armaan/venvs/omnizip_clean/bin/activate
cd /data/armaan/purs

# Baseline (pick GPU)
CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni.py \
  --metadata videos/metadata.json \
  --videos videos \
  --output run2/baseline/results.jsonl \
  --log run2/baseline/console.log \
  --vram_log run2/baseline/vram_log.jsonl \
  --errors_log run2/baseline/errors.log \
  --stderr_log run2/baseline/stderr.log \
  --model_variant qwen2.5-omni \
  --fps 0.5 --max_pixels 50176 --max_new_tokens 256

# OmniZip (use another GPU in parallel if you want)
CUDA_VISIBLE_DEVICES=1 python eval_qwen_omni_zip.py \
  --metadata videos/metadata.json \
  --videos videos \
  --output run2/omnizip/results.jsonl \
  --log run2/omnizip/console.log \
  --vram_log run2/omnizip/vram_log.jsonl \
  --errors_log run2/omnizip/errors.log \
  --stderr_log run2/omnizip/stderr.log \
  --fps 0.5 --max_pixels 50176 \
  --rho_audio 0.3 --rho_video 0.6 --g 3 --contextual_ratio 0.05
```

---

## Recommended Full-Set Runs (`runs/...`, strict audio, clean stderr)

These commands run the full dataset in `videos/metadata.json`:

- **44** video entries
- **118** total questions

They use:

- the recommended **`omnizip_clean`** venv
- timestamped output folders under `runs/`
- **`--stderr_log`** instead of `2>&1 | tee ...`
- the patched eval scripts, which keep **video + audio + text** as input by default and only generate **text** output

Important:

- If a video has an audio stream and audio decoding fails, the run now errors instead of silently falling back to video-only.
- If a video genuinely has **no audio stream**, the script logs that and uses **video + text** for that item because no audio exists to consume.
- `qwen-vl-utils` / `decord` logs the source file's native FPS (for example `video_fps=30.0`); that is **not** your requested `--fps` sampling rate.

Set common inputs once, then launch any of the runs below:

```bash
source /data/armaan/venvs/omnizip_clean/bin/activate
cd /data/armaan/purs

export META=videos/metadata.json
export VIDS=videos
export FPS=0.5
export MAX_PIXELS=50176
export MAX_NEW_TOKENS=256
```

### 1. Qwen2.5-Omni-7B Baseline (full 44 / 118)

```bash
TS=$(date +%Y%m%d_%H%M%S)
DIR=/data/armaan/purs/runs/qwen25_baseline_${TS}
mkdir -p "$DIR"

CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni.py \
  --metadata "$META" \
  --videos "$VIDS" \
  --output "$DIR/results.jsonl" \
  --log "$DIR/console.log" \
  --vram_log "$DIR/vram_log.jsonl" \
  --errors_log "$DIR/errors.log" \
  --stderr_log "$DIR/stderr.log" \
  --model_variant qwen2.5-omni \
  --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS"
```

**Model**: `/data/armaan/models/Qwen2.5-Omni-7B` (auto-detected)

### 2. Qwen2.5-Omni-7B + OmniZip (full 44 / 118)

```bash
TS=$(date +%Y%m%d_%H%M%S)
DIR=/data/armaan/purs/runs/qwen25_omnizip_${TS}
mkdir -p "$DIR"

CUDA_VISIBLE_DEVICES=1 python eval_qwen_omni_zip.py \
  --metadata "$META" \
  --videos "$VIDS" \
  --output "$DIR/results.jsonl" \
  --log "$DIR/console.log" \
  --vram_log "$DIR/vram_log.jsonl" \
  --errors_log "$DIR/errors.log" \
  --stderr_log "$DIR/stderr.log" \
  --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS" \
  --rho_audio 0.3 --rho_video 0.6 --g 3 --contextual_ratio 0.05
```

**Model**: `/data/armaan/models/Qwen2.5-Omni-7B` (auto-detected)

### 3. Qwen2.5-Omni-7B GPTQ (full 44 / 118)

```bash
TS=$(date +%Y%m%d_%H%M%S)
DIR=/data/armaan/purs/runs/qwen25_gptq_${TS}
mkdir -p "$DIR"

CUDA_VISIBLE_DEVICES=2 python eval_qwen_omni.py \
  --metadata "$META" \
  --videos "$VIDS" \
  --output "$DIR/results.jsonl" \
  --log "$DIR/console.log" \
  --vram_log "$DIR/vram_log.jsonl" \
  --errors_log "$DIR/errors.log" \
  --stderr_log "$DIR/stderr.log" \
  --model_variant qwen2.5-omni \
  --quantization gptq \
  --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS"
```

**Model**: `/data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4` (auto-detected)

### 4. Qwen2.5-Omni-7B AWQ (full 44 / 118)

```bash
TS=$(date +%Y%m%d_%H%M%S)
DIR=/data/armaan/purs/runs/qwen25_awq_${TS}
mkdir -p "$DIR"

CUDA_VISIBLE_DEVICES=3 python eval_qwen_omni.py \
  --metadata "$META" \
  --videos "$VIDS" \
  --output "$DIR/results.jsonl" \
  --log "$DIR/console.log" \
  --vram_log "$DIR/vram_log.jsonl" \
  --errors_log "$DIR/errors.log" \
  --stderr_log "$DIR/stderr.log" \
  --model_variant qwen2.5-omni \
  --quantization awq \
  --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS"
```

**Model**: `/data/armaan/models/Qwen2.5-Omni-7B-AWQ` (auto-detected)

### Notes for GPTQ / AWQ

- These quantized commands run through `eval_qwen_omni.py`, **not** `eval_qwen_omni_zip.py`.
- Quantized + OmniZip is **not** wired up yet in `eval_qwen_omni_zip.py`.
- GPTQ / AWQ force `torch_dtype=float16` internally.
- Start with the same conservative `FPS`, `MAX_PIXELS`, and `MAX_NEW_TOKENS` as baseline; increase only after a clean full run.

---

## Run 3 — FlashAttention-2, paper-default resolution + frame caps (`run3/`)

Now using `flash_attention_2` (flash-attn 2.7.4.post1 built from source) with the OmniZip paper's default settings:
- `FPS=2.0` (paper default, was 0.5)
- `MAX_PIXELS=100352` (128×28×28 = `VIDEO_MAX_PIXELS` default, was 50176)
- `MAX_NEW_TOKENS=256`
- `--max_frames_videomme 768` / `--max_frames_other 128` (paper defaults, auto-applied per dataset)

Frame caps are applied automatically: VideoMME gets up to 768 frames, all other datasets (worldsense, daily-omni) get up to 128 frames. This matches the paper exactly and prevents OOM on long videos.

All 4 runs can launch **simultaneously** on different GPUs.

```bash
source /data/armaan/venvs/omnizip_clean/bin/activate
cd /data/armaan/purs

export META=videos/metadata.json
export VIDS=videos
export FPS=2.0
export MAX_PIXELS=100352
export MAX_NEW_TOKENS=256
```

### 3.1 Baseline

```bash
CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni.py --metadata "$META" --videos "$VIDS" --output run3/baseline/results.jsonl --log run3/baseline/console.log --vram_log run3/baseline/vram_log.jsonl --errors_log run3/baseline/errors.log --stderr_log run3/baseline/stderr.log --model_variant qwen2.5-omni --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS" &
```

### 3.2 OmniZip

```bash
CUDA_VISIBLE_DEVICES=1 python eval_qwen_omni_zip.py --metadata "$META" --videos "$VIDS" --output run3/omnizip/results.jsonl --log run3/omnizip/console.log --vram_log run3/omnizip/vram_log.jsonl --errors_log run3/omnizip/errors.log --stderr_log run3/omnizip/stderr.log --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS" --rho_audio 0.3 --rho_video 0.6 --g 3 --contextual_ratio 0.05 &
```

### 3.3 GPTQ

```bash
CUDA_VISIBLE_DEVICES=2 python eval_qwen_omni.py --metadata "$META" --videos "$VIDS" --output run3/gptq/results.jsonl --log run3/gptq/console.log --vram_log run3/gptq/vram_log.jsonl --errors_log run3/gptq/errors.log --stderr_log run3/gptq/stderr.log --model_variant qwen2.5-omni --quantization gptq --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS" &
```

### 3.4 AWQ

```bash
CUDA_VISIBLE_DEVICES=3 python eval_qwen_omni.py --metadata "$META" --videos "$VIDS" --output run3/awq/results.jsonl --log run3/awq/console.log --vram_log run3/awq/vram_log.jsonl --errors_log run3/awq/errors.log --stderr_log run3/awq/stderr.log --model_variant qwen2.5-omni --quantization awq --fps "$FPS" --max_pixels "$MAX_PIXELS" --max_new_tokens "$MAX_NEW_TOKENS" &
```

**Notes**:
- Frame caps (`--max_frames_videomme 768 --max_frames_other 128`) are the defaults — no need to pass them explicitly.
- GPTQ/AWQ still use `sdpa` (quantization libraries may not support FA2). Baseline and OmniZip use `flash_attention_2`.
- All commands are single-line to avoid line-continuation issues when pasting into terminal.

---

## Pre-flight Checks

```bash
# Verify metadata and videos exist
ls videos/metadata.json
ls videos/

# Check free GPUs (need GPUs with ~2MiB usage)
nvidia-smi

# Verify python env works
/data/armaan/mamba/envs/omnizip_pip/bin/python -c "import torch; print(torch.cuda.device_count(), 'GPUs')"
```

---

## 1. Qwen2.5-Omni Baseline (single GPU)

```bash
export PATH="/data/armaan/mamba/envs/omnizip_pip/bin:$PATH"
cd /data/armaan/purs && TS=$(date +%Y%m%d_%H%M%S) && \
DIR=/data/armaan/purs/runs/qwen25_baseline_${TS} && mkdir -p $DIR && \
CUDA_VISIBLE_DEVICES=6 /data/armaan/mamba/envs/omnizip_pip/bin/python eval_qwen_omni.py \
  --metadata videos/metadata.json \
  --videos videos \
  --output $DIR/results.jsonl \
  --log $DIR/console.log \
  --vram_log $DIR/vram_log.jsonl \
  --errors_log $DIR/errors.log \
  --model_variant qwen2.5-omni \
  --fps 0.5 --max_pixels 50176 --max_new_tokens 256 \
  2>&1 | tee $DIR/stderr.log
```

**Model**: `/data/armaan/models/Qwen2.5-Omni-7B` (auto-detected by script)
**VRAM**: ~25-35GB on 1x RTX 6000 Ada (49GB)
**Note**: fps=2.0/max_pixels=151200 OOMs on single GPU (needs 60GB+ for attention). fps=0.5/max_pixels=50176 fits.

---

## 2. Qwen2.5-Omni + OmniZip (single GPU)

```bash
export PATH="/data/armaan/mamba/envs/omnizip_pip/bin:$PATH"
cd /data/armaan/purs && TS=$(date +%Y%m%d_%H%M%S) && \
DIR=/data/armaan/purs/runs/qwen25_omnizip_${TS} && mkdir -p $DIR && \
CUDA_VISIBLE_DEVICES=7 /data/armaan/mamba/envs/omnizip_pip/bin/python eval_qwen_omni_zip.py \
  --metadata videos/metadata.json \
  --videos videos \
  --output $DIR/results.jsonl \
  --log $DIR/console.log \
  --vram_log $DIR/vram_log.jsonl \
  --errors_log $DIR/errors.log \
  --fps 0.5 --max_pixels 50176 \
  --rho_audio 0.3 --rho_video 0.6 --g 3 --contextual_ratio 0.05 \
  2>&1 | tee $DIR/stderr.log
```

**Model**: `/data/armaan/models/Qwen2.5-Omni-7B` (auto-detected by script)
**OmniZip** (same as `OmniZip-main` / `lmms_eval/models/simple/qwen2_5_omni.py` defaults): rho_audio=0.3, rho_video=0.6, g=3, contextual_ratio=0.05
**VRAM**: ~18-25GB expected (1.4x reduction from OmniZip)

> Run 1 and 2 simultaneously on different GPUs.

---

## 3. Qwen3-Omni Baseline (multi-GPU)

### 3a. Download the model first

```bash
# Install latest transformers from source (required for Qwen3-Omni)
/data/armaan/mamba/envs/omnizip_pip/bin/pip install git+https://github.com/huggingface/transformers
/data/armaan/mamba/envs/omnizip_pip/bin/pip install accelerate qwen-omni-utils -U

# Download model (~60GB)
/data/armaan/mamba/envs/omnizip/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct', local_dir='/data/armaan/models/Qwen3-Omni-30B-A3B-Instruct')
"
```

### 3b. Run evaluation (needs 2-3 GPUs, ~78-145GB VRAM in bf16)

```bash
TS=$(date +%Y%m%d_%H%M%S) && \
DIR=/data/armaan/purs/runs/qwen3_baseline_${TS} && mkdir -p $DIR && \
CUDA_VISIBLE_DEVICES=6,7 /data/armaan/mamba/envs/omnizip_pip/bin/python eval_qwen_omni.py \
  --model /data/armaan/models/Qwen3-Omni-30B-A3B-Instruct \
  --metadata videos/metadata.json \
  --videos videos \
  --output $DIR/results.jsonl \
  --log $DIR/console.log \
  --vram_log $DIR/vram_log.jsonl \
  --errors_log $DIR/errors.log \
  --model_variant qwen3-omni \
  --fps 0.5 --max_pixels 50176 --max_new_tokens 256 \
  2>&1 | tee $DIR/stderr.log
```

**Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (30B MoE, 3B active params)
**VRAM**: 78-145GB bf16 for model weights alone -> needs 2-3x RTX 6000 Ada (49GB each)
**Note**: Run AFTER runs 1 & 2 finish to free up GPUs. Use `CUDA_VISIBLE_DEVICES=6,7` (or more GPUs if needed). May need even lower fps for long videos.

---

## Output Structure

Each run creates a timestamped folder under `runs/`:

```
runs/
  qwen25_baseline_20260331_024700/
    results.jsonl       # per-question predictions
    console.log         # stdout log with accuracy summary
    vram_log.jsonl      # per-question VRAM usage
    errors.log          # tracebacks for failed questions
    stderr.log          # stderr capture
  qwen25_omnizip_20260331_024700/
    ...
  qwen3_baseline_20260331_034500/
    ...
```

---

## Qwen3-Omni Model Info

- **HuggingFace ID**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- **Architecture**: Thinker-Talker MoE (30B total, 3B active)
- **Variants**: Instruct, Thinking, Captioner
- **Requires**: transformers from source (not yet on PyPI)
