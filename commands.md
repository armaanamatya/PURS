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

```bash
source /data/armaan/venvs/omnizip_clean/bin/activate
cd /data/armaan/purs
mkdir -p run2/baseline run2/omnizip

# Baseline (pick GPU)
CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni.py \
  --metadata videos/metadata.json \
  --videos videos \
  --output run2/baseline/results.jsonl \
  --log run2/baseline/console.log \
  --vram_log run2/baseline/vram_log.jsonl \
  --errors_log run2/baseline/errors.log \
  --model_variant qwen2.5-omni \
  --fps 0.5 --max_pixels 50176 --max_new_tokens 256 \
  2>&1 | tee run2/baseline/stderr.log

# OmniZip (use another GPU in parallel if you want)
CUDA_VISIBLE_DEVICES=1 python eval_qwen_omni_zip.py \
  --metadata videos/metadata.json \
  --videos videos \
  --output run2/omnizip/results.jsonl \
  --log run2/omnizip/console.log \
  --vram_log run2/omnizip/vram_log.jsonl \
  --errors_log run2/omnizip/errors.log \
  --fps 0.5 --max_pixels 50176 \
  --rho_audio 0.3 --rho_video 0.6 --g 3 --contextual_ratio 0.05 \
  2>&1 | tee run2/omnizip/stderr.log
```

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
