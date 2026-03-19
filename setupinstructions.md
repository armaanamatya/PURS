# Instance Setup Instructions — Qwen2.5-Omni + OmniZip Viz

## Docker workflow (recommended for repeat experiments)

### One-time setup

**1. Build and push image** (run locally from `C:/Users/Armaan/Desktop/PURS/`):
```bash
docker build -t <your-dockerhub-username>/qwen-omnizip:latest .
docker push <your-dockerhub-username>/qwen-omnizip:latest
```

**2. Create a persistent vast.ai volume** to store the model (one-time, ~15GB):
```bash
# On any vast.ai instance, download model to volume mount point
huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /workspace/model
```
Then in the vast.ai UI: Instances → Storage → create a volume snapshot from `/workspace/model`.

### Each new experiment
1. Rent a new instance using your custom image: `<dockerhub>/qwen-omnizip:latest`
2. Attach the model volume at `/workspace/model`
3. Run scripts immediately — no setup needed

### Updating code or scripts
```bash
# Edit scripts locally, then rebuild and push:
docker build -t <your-dockerhub-username>/qwen-omnizip:latest .
docker push <your-dockerhub-username>/qwen-omnizip:latest
```

---

# Current instance (manual setup)

## Instance details
| | |
|---|---|
| **GPU** | A100 SXM4 80GB |
| **Cost** | $1.161/hr |
| **Instance ID** | `33120862` |
| **SSH** | `ssh -p 10862 root@ssh7.vast.ai` |
| **Docker image** | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| **Disk** | 80 GB |

---

## 1. Copy files to instance

Only these files are needed (not the full OmniZip repo):

```bash
# Run from C:/Users/Armaan/Desktop/PURS on your local machine
scp -P 10862 viz_attention_qwen.py viz_attention_omnizip.py root@ssh7.vast.ai:/workspace/

scp -P 10862 -r OmniZip-main/omnizip root@ssh7.vast.ai:/workspace/OmniZip-main/
scp -P 10862 -r OmniZip-main/qwen-omni-utils root@ssh7.vast.ai:/workspace/OmniZip-main/

scp -P 10862 -r videos/worldsense/attribute_reasoning root@ssh7.vast.ai:/workspace/videos/worldsense/
```

Remote layout expected by scripts:
```
/workspace/
  viz_attention_qwen.py
  viz_attention_omnizip.py
  OmniZip-main/
    omnizip/
      modeling_qwen2_5_omni.py
      omnizip_units.py
    qwen-omni-utils/
      pyproject.toml
      src/qwen_omni_utils/
  videos/
    worldsense/
      attribute_reasoning/
        video.mp4
  model/          ← downloaded in step 3
```

---

## 2. Install dependencies

```bash
# PyTorch already in container — skip torch install

pip install transformers==4.52.3 accelerate moviepy matplotlib

# qwen-omni-utils (installs: requests, pillow, av, packaging, librosa)
pip install -e /workspace/OmniZip-main/qwen-omni-utils/

# PyAV v12+ renamed av.AVError → av.OSError, breaking torchvision's video reader
# Requires build deps before pip install
apt-get install -y pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
pip install "av<12"

# ffmpeg — required by librosa/audioread to decode audio from video files
apt-get install -y ffmpeg
```

---

## 3. Download model (~15 GB)

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /workspace/model
```

---

## 4. Run viz scripts

```bash
cd /workspace
python viz_attention_qwen.py       # outputs → attention_viz_qwen/
python viz_attention_omnizip.py    # outputs → attention_viz_omnizip/
```

---

## 5. Copy results back

```bash
# Run locally
scp -P 10862 -r root@ssh7.vast.ai:/workspace/attention_viz_qwen ./
scp -P 10862 -r root@ssh7.vast.ai:/workspace/attention_viz_omnizip ./
```

---

## GPU version compatibility

**No version changes needed between GPU types.** The same container
(`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`) and same pip deps work on:

| GPU | Compute Cap | CUDA 12.x support |
|---|---|---|
| A100 SXM4/PCIE | 8.0 | ✓ |
| H100 SXM/NVL | 9.0 | ✓ |
| A6000 | 8.6 | ✓ |

PyTorch 2.5.x supports compute capability 6.0–9.0 out of the box.

**Note on previous eval runs (`omnizipresults/`):** The `vram_log.jsonl` shows
peaks up to ~110 GB, which exceeds a single GPU. Those runs used
`device_map="auto"` (multi-GPU), so total VRAM is the sum across all GPUs.
The viz scripts use `device_map="cuda:0"` (single GPU) — the specific
attribute_reasoning video peaked at only **27.67 GB** in normal eval, so the
A100 80GB has plenty of headroom even with eager attention overhead.
