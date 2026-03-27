# Fixes / HPC bring-up notes

This file documents the issues we hit while moving PURS experiments to the GPU cluster and the exact working setup/commands.

## What we were trying to do

- Run 3 comparisons:
  - **Qwen2.5-Omni baseline** (no OmniZip) via `eval_qwen_omni.py`
  - **Qwen2.5-Omni + OmniZip** via `eval_qwen_omni_zip.py`
  - **Qwen3 baseline** via `eval_qwen_omni.py --model_variant qwen3-omni` (optional; depends on `transformers` support + weights)
- Store everything under **`/data/armaan/`** (models, caches, runs, logs). Avoid `/workspace` since it’s not writable on HPC.

## Remote layout (cluster)

- Code: `/data/armaan/purs`
- Videos: `/data/armaan/purs/videos` (includes `videos/metadata.json` and `.mp4`s)
- Models: `/data/armaan/models`
- HF cache: `/data/armaan/hf-cache`
- Run outputs: `/data/armaan/runs/<run_name>/`

## Copy/sync behavior (local → cluster)

### Goal: only send experiment necessities

We used an rsync script (`scripts/sync-to-data-armaan.bash`) to push the repo to `/data/armaan/purs` and progressively excluded:

- Junk/dev: `.git/`, `__pycache__/`, `*.py[cod]`, venvs, `.env`, logs, caches, `node_modules/`, `website/`, `.next/`, `.cursor/`, `.claude/`, etc.
- Docs/artifacts: `docs/`, `*.md`, `*.pdf`, `*.ipynb`
- Unneeded trees: `OmniZip-OG/`, `Qwen2.5-Omni/` demo materials, `omnizipresults/`
- Optional heavy dependency: `OmniZip-main/lmms-eval/` (via `SYNC_SLIM=1`)

### Common gotcha (Windows)

- PowerShell cannot execute `.bash` scripts directly; use Git Bash/WSL, or call wrappers (`scripts/sync-to-data-armaan.ps1` / `.cmd`).

## Model download location

### Download to `/data/armaan/models/Qwen2.5-Omni-7B`

Recommended (Hugging Face CLI):

```bash
export HF_HOME=/data/armaan/hf-cache
mkdir -p /data/armaan/models
huggingface-cli download Qwen/Qwen2.5-Omni-7B \
  --local-dir /data/armaan/models/Qwen2.5-Omni-7B \
  --local-dir-use-symlinks False
```

Alternative (Python):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="Qwen/Qwen2.5-Omni-7B",
  local_dir="/data/armaan/models/Qwen2.5-Omni-7B",
  local_dir_use_symlinks=False,
)
PY
```

## Environment management on HPC (what worked)

### Problems we hit

- `conda` not available on the node.
- `python -m venv` failed: `ensurepip is not available` (missing `python3.10-venv` system package; we avoided sudo).
- Attempting to install `torchcodec` via micromamba pulled a partial **PyTorch 2.10 CUDA 13** upgrade and corrupted the env:
  - `ImportError: libtorch_cuda.so: undefined symbol: ncclCommWindowDeregister`
- A micromamba env created with `pytorch=2.6` ended up **CPU-only**:
  - `torch.version.cuda None`, `torch.cuda.is_available() False`
- Solver conflicts when pinning the GPU stack via micromamba channels on this cluster.

### Working solution: micromamba env + pip CUDA wheels (cu124)

Create env with Python + ffmpeg, then install CUDA-enabled PyTorch wheels via pip:

```bash
micromamba create -y -n omnizip_pip -c conda-forge python=3.10 ffmpeg
micromamba activate omnizip_pip

pip install -U pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
PY
```

Then install experiment deps:

```bash
pip install transformers==4.52.3 accelerate moviepy huggingface_hub

# Audio/video decode deps required by qwen_omni_utils:
pip install audioread av librosa
```

## Required exports (cluster)

These were used repeatedly during setup/runs:

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/armaan/hf-cache

# Use repo code without editable install:
export PYTHONPATH="/data/armaan/purs/OmniZip-main:/data/armaan/purs/OmniZip-main/qwen-omni-utils/src:${PYTHONPATH}"
```

Optional stability knobs (only if you see decode resource errors):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Video backend selection in `qwen_omni_utils`:

- `qwen-omni-utils` will try `torchcodec` if available, else `decord`, else `torchvision`.
- If a backend is flaky on a node, you can force it:

```bash
export FORCE_QWENVL_VIDEO_READER=torchvision
```

## Persistent video decode failures (what finally fixed it)

### Symptom

On some nodes / for some `video-mme` clips, `torchvision`/PyAV would intermittently fail to open decoders:

- `avcodec_open2("h264", {})` → `[Errno 11] Resource temporarily unavailable`

Some clips were likely **AV1** (still inside `.mp4`), triggering:

- `avcodec_open2("libdav1d", {})` → `[Errno 12] Cannot allocate memory`

### Fix strategy

1) **Force `decord`** for video reading (more stable than `torchvision` on this node):

```bash
export FORCE_QWENVL_VIDEO_READER=decord
```

2) Patch `qwen_omni_utils` video reading to reduce decode pressure and avoid backend flip-flopping:

- `OmniZip-main/qwen-omni-utils/src/qwen_omni_utils/v2_5/vision_process.py`
  - `decord.VideoReader(..., num_threads=1, fault_tol=1)`
  - `torchvision.VideoReader(..., num_threads=1)` fallback path
  - If `FORCE_QWENVL_VIDEO_READER` is set, do **not** silently switch to another backend.

3) For the few clips that still failed (codec-level), **transcode in-place to H.264/AAC** so `metadata.json` does not need editing:

Install ffmpeg inside the micromamba env (no sudo):

```bash
micromamba activate omnizip_pip
micromamba install -y -c conda-forge ffmpeg
which ffmpeg
ffmpeg -version
```

Transcode the specific problematic files:

```bash
for f in \
  /data/armaan/purs/videos/video-mme/object_recognition/video.mp4 \
  /data/armaan/purs/videos/video-mme/spatial_perception/video.mp4 \
  /data/armaan/purs/videos/video-mme/spatial_reasoning/video.mp4 \
  /data/armaan/purs/videos/video-mme/temporal_perception/video.mp4
do
  tmp="${f}.tmp_h264.mp4"
  ffmpeg -y -hide_banner -loglevel error -i "$f" \
    -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 23 \
    -c:a aac -b:a 128k \
    "$tmp" && mv -f "$tmp" "$f"
done
```

After transcoding, the previously-`ERROR` samples produced normal predictions (no decoder errors).

## qwen-omni-utils packaging gotcha

We originally excluded `*.md` from sync. That caused `pip install -e OmniZip-main/qwen-omni-utils` to fail because packaging expected `README.md`.

We avoided editable install entirely by using:

```bash
export PYTHONPATH=".../OmniZip-main:.../qwen-omni-utils/src:${PYTHONPATH}"
```

## Runtime errors we fixed in code

### 1) OmniZip `topk` crash (“selected index k out of range”)

Old log showed:

- `RuntimeError: selected index k out of range` from `omnizip_audio_attn` (`torch.topk(attn_logits, dominant_num)`).

Fix (in repo):

- `OmniZip-main/omnizip/omnizip_units.py`: clamp `dominant_num` to `attn_logits.numel()` before calling `topk`.

### 2) Transient video decode failures (PyAV/torchvision)

Old log showed:

- `av.error.BlockingIOError: [Errno 11] Resource temporarily unavailable ... swscaler`

Fix (in repo):

- `eval_qwen_omni_zip.py`: retry `process_mm_info(...)` up to 3 times with short backoff.

### 3) Writing to `/workspace/errors.log` on HPC

On HPC, `/workspace` isn’t writable.

Fix (in repo):

- `eval_qwen_omni_zip.py` now accepts `--errors_log` and otherwise writes `errors.log` alongside `--log`.

### 4) Hardcoded `/workspace/model`

On HPC, the model is stored under `/data/armaan/models/...`.

Fix (in repo):

- `eval_qwen_omni.py` and `eval_qwen_omni_zip.py` accept `--model` and/or env var `QWEN_OMNI_MODEL_PATH`.
- Default model path is `/data/armaan/models/Qwen2.5-Omni-7B`.
- Fallback to `/workspace/model` only if it exists.

### 5) Baseline script importing Qwen3 classes unconditionally

On the cluster with `transformers==4.52.3`, Qwen3-Omni classes were not importable:

- `ImportError: cannot import name 'Qwen3OmniMoeForConditionalGeneration' from transformers`

Fix (in repo):

- `eval_qwen_omni.py` now imports Qwen3 classes lazily only when `--model_variant qwen3-omni` is requested.

### 6) Baseline OOM on 48GB GPUs

Baseline (no OmniZip) was OOM for some inputs at the previous defaults.

Fix (in repo):

- `eval_qwen_omni.py` gained knobs:
  - `--fps` (default 0.5)
  - `--max_pixels` (default 256*256)
  - `--max_new_tokens` (default 256)

## Commands to run the 3 comparisons (separate output folders)

Run from `/data/armaan/purs` after activating `omnizip_pip` and setting exports above.

### A) Qwen2.5 baseline (no OmniZip)

```bash
RUN=/data/armaan/runs/qwen25_baseline_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"
python /data/armaan/purs/eval_qwen_omni.py \
  --model /data/armaan/models/Qwen2.5-Omni-7B \
  --model_variant qwen2.5-omni \
  --metadata /data/armaan/purs/videos/metadata.json \
  --videos /data/armaan/purs/videos \
  --output "$RUN/results.jsonl" \
  --log "$RUN/eval.log" \
  --fps 0.5 \
  --max_pixels $((256*256)) \
  --max_new_tokens 256
```

### B) Qwen2.5 + OmniZip

```bash
export FORCE_QWENVL_VIDEO_READER=decord
RUN=/data/armaan/runs/qwen25_omnizip_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"
python /data/armaan/purs/eval_qwen_omni_zip.py \
  --model /data/armaan/models/Qwen2.5-Omni-7B \
  --metadata /data/armaan/purs/videos/metadata.json \
  --videos /data/armaan/purs/videos \
  --output "$RUN/results.jsonl" \
  --log "$RUN/eval.log" \
  --vram_log "$RUN/vram_log.jsonl" \
  --errors_log "$RUN/errors.log" \
  --fps 0.25 \
  --max_pixels $((224*224))
```

### C) Qwen3 baseline (optional)

Requirements:

- Download Qwen3 weights to `/data/armaan/models/<QWEN3_DIR>`
- Install a `transformers` version that includes Qwen3-Omni support (cluster default `transformers==4.52.3` did not).

```bash
RUN=/data/armaan/runs/qwen3_baseline_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"
python /data/armaan/purs/eval_qwen_omni.py \
  --model /data/armaan/models/<QWEN3_DIR> \
  --model_variant qwen3-omni \
  --metadata /data/armaan/purs/videos/metadata.json \
  --videos /data/armaan/purs/videos \
  --output "$RUN/results.jsonl" \
  --log "$RUN/eval.log" \
  --fps 0.5 \
  --max_pixels $((256*256)) \
  --max_new_tokens 256
```

## Final “fresh instance” setup (copy/paste)

This is the minimal serial setup to run experiments on an HPC instance without sudo.

### 0) One-time directories

```bash
mkdir -p /data/armaan/{tools,mamba,models,hf-cache,purs,runs}
```

### 1) Install micromamba (user-local)

```bash
cd /data/armaan/tools
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=/data/armaan/mamba
eval "$(/data/armaan/tools/bin/micromamba shell hook -s bash)"
```

### 2) Create env (pinned Python) + install GPU stack

```bash
micromamba create -y -n omnizip_pip -c conda-forge python=3.10 ffmpeg
micromamba activate omnizip_pip

pip install -U pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install transformers==4.52.3 accelerate==1.13.0 moviepy huggingface_hub
pip install audioread av librosa
```

### 3) Download model weights

```bash
export HF_HOME=/data/armaan/hf-cache
mkdir -p /data/armaan/models
huggingface-cli download Qwen/Qwen2.5-Omni-7B \
  --local-dir /data/armaan/models/Qwen2.5-Omni-7B \
  --local-dir-use-symlinks False
```

### 4) Run (baseline + OmniZip)

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/data/armaan/purs/OmniZip-main:/data/armaan/purs/OmniZip-main/qwen-omni-utils/src:${PYTHONPATH}"
export FORCE_QWENVL_VIDEO_READER=decord
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Baseline
RUN=/data/armaan/runs/qwen25_baseline_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"
python /data/armaan/purs/eval_qwen_omni.py \
  --model /data/armaan/models/Qwen2.5-Omni-7B \
  --model_variant qwen2.5-omni \
  --metadata /data/armaan/purs/videos/metadata.json \
  --videos /data/armaan/purs/videos \
  --output "$RUN/results.jsonl" \
  --log "$RUN/eval.log" \
  --fps 0.5 \
  --max_pixels $((256*256)) \
  --max_new_tokens 256

# OmniZip
RUN=/data/armaan/runs/qwen25_omnizip_$(date +%Y%m%d_%H%M%S); mkdir -p "$RUN"
python /data/armaan/purs/eval_qwen_omni_zip.py \
  --model /data/armaan/models/Qwen2.5-Omni-7B \
  --metadata /data/armaan/purs/videos/metadata.json \
  --videos /data/armaan/purs/videos \
  --output "$RUN/results.jsonl" \
  --log "$RUN/eval.log" \
  --vram_log "$RUN/vram_log.jsonl" \
  --errors_log "$RUN/errors.log" \
  --fps 0.25 \
  --max_pixels $((224*224))
```

