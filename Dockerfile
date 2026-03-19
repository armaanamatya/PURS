FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# System deps
RUN apt-get update && apt-get install -y \
    ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
    transformers==4.52.3 \
    accelerate \
    moviepy \
    matplotlib \
    numpy \
    huggingface_hub \
    requests \
    pillow \
    av \
    packaging \
    librosa \
    soundfile

WORKDIR /workspace

# Copy OmniZip (only what's needed — omnizip/ + qwen-omni-utils/)
COPY OmniZip-main/omnizip/ /workspace/OmniZip-main/omnizip/
COPY OmniZip-main/qwen-omni-utils/ /workspace/OmniZip-main/qwen-omni-utils/

# Install qwen-omni-utils from local copy
RUN pip install --no-cache-dir -e /workspace/OmniZip-main/qwen-omni-utils/

# Copy scripts
COPY viz_attention_qwen.py /workspace/
COPY viz_attention_omnizip.py /workspace/
COPY eval_qwen_omni.py /workspace/
COPY eval_qwen_omni_zip.py /workspace/

# Copy videos
COPY videos/ /workspace/videos/

# Model is NOT baked in — mount a vast.ai volume at /workspace/model
# or download on first run with:
#   huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir /workspace/model

CMD ["bash"]
