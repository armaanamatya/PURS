#!/bin/bash
# setup_instance.sh
# Run this once on a fresh vast.ai instance to prepare for eval_qwen_omni_zip.py
set -e

echo "=== Installing system packages ==="
apt-get update -qq
apt-get install -y -qq ffmpeg pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev

echo "=== Installing Python packages ==="
pip3 install -q transformers==4.52.3 accelerate moviepy decord soundfile
pip3 install -q qwen-vl-utils

echo "=== Installing qwen-omni-utils ==="
cd /workspace/qwen-omni-utils && pip3 install -q -e . && cd /workspace

echo "=== Patching av.AVError in torchvision ==="
TV_VIDEO=$(python3 -c "import torchvision; import os; print(os.path.join(os.path.dirname(torchvision.__file__), 'io', 'video.py'))" 2>/dev/null)
if [ -f "$TV_VIDEO" ]; then
    sed -i 's/FFmpegError = av.AVError/FFmpegError = Exception/' "$TV_VIDEO" && echo "Patched $TV_VIDEO"
else
    echo "torchvision video.py not found, skipping patch"
fi

echo "=== Downloading model (Qwen2.5-Omni-7B) ~15GB ==="
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Omni-7B', local_dir='/workspace/model')
"

echo "=== Verifying CUDA ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Torch:', torch.__version__)"

echo ""
echo "=== Setup complete. Run eval with: ==="
echo "python eval_qwen_omni_zip.py --metadata /workspace/videos/metadata.json --videos /workspace/videos --output /workspace/results_zip.jsonl --log /workspace/eval_zip.log"
