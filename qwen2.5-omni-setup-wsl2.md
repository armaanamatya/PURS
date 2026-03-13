# Qwen2.5-Omni Setup Guide (WSL2 + 5060 Ti 16GB)

## What Are the 3 Scripts in `low-VRAM-mode/`?

| Script | What It Does |
|--------|-------------|
| `modeling_qwen2_5_omni_low_VRAM_mode.py` | **Modified model architecture** — a patched version of the official Qwen2.5-Omni model code that enables the low-VRAM device map (splitting modules across CPU/GPU). Both demo scripts import from this. You never run this directly. |
| `low_VRAM_demo_awq.py` | **AWQ 4-bit quantized demo** — loads `Qwen2.5-Omni-7B-AWQ` (4-bit weights via AutoAWQ). Offloads visual encoder + audio tower to CPU, keeps the quantized thinker on GPU. Generates both text + audio output by default. |
| `low_VRAM_demo_gptq.py` | **GPTQ-Int4 quantized demo** — loads `Qwen2.5-Omni-7B-GPTQ-Int4` (4-bit via GPTQModel). Same CPU/GPU split strategy. Also generates text + audio by default. |

**AWQ vs GPTQ:** Both are 4-bit quantization. AWQ is generally slightly faster at inference; GPTQ is more widely supported. Either works for 16GB VRAM. This guide uses **AWQ**.

---

## What We're Setting Up

- **Model:** `Qwen/Qwen2.5-Omni-7B-AWQ` (~4-5 GB quantized weights)
- **Mode:** Text-only output (talker disabled, saves ~3-4 GB VRAM)
- **Input:** Full audio+video understanding (model sees AND hears the video)
- **Output:** Text only (no speech synthesis)
- **Video:** `OmniZip/assets/example.mp4` (up to 30 seconds)
- **Estimated peak VRAM:** ~9-12 GB (fits in 16GB)

---

## Step 1: Install WSL2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot your PC. After reboot, Ubuntu will launch and ask you to create a username and password.

Verify your GPU is visible inside WSL:

```bash
nvidia-smi
```

You should see your 5060 Ti listed. If not, make sure you have the latest NVIDIA Game Ready or Studio driver installed on Windows (WSL2 uses the Windows driver — you do NOT install a separate Linux driver).

---

## Step 2: Install Miniconda in WSL2

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

---

## Step 3: Create the Conda Environment

```bash
conda create -n omnizip python=3.10 -y
conda activate omnizip
```

---

## Step 4: Install PyTorch + Dependencies

```bash
# PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Core packages
pip install transformers==4.52.3
pip install accelerate
pip install autoawq==0.2.9
pip install moviepy
pip install soundfile

# Flash Attention 2 (this is why we need WSL2 — doesn't build on Windows)
# IMPORTANT: Use version 2.7.3 — it's the last version compatible with torch 2.6.0.
# Do NOT use flash-attn 2.8.x as it pulls in torch 2.10 and causes ABI mismatches.
# Must build from source (--no-cache-dir prevents using incompatible prebuilt wheels).
pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir

# Qwen omni utils (from the repo you already have)
# NOTE: Installing with -e from /mnt/c/ can fail with UnicodeDecodeError due to
# Windows filesystem encoding issues. Copy to WSL native filesystem first:
cp -r /mnt/c/Users/Armaan/Desktop/PURS/Qwen2.5-Omni/qwen-omni-utils ~/qwen-omni-utils
cd ~/qwen-omni-utils
pip install -e .
cd ~
```

> **Note:** `flash-attn` compilation takes 10-20 minutes. Be patient.
>
> **Warning:** Do NOT run `pip install flash-attn --force-reinstall` — it will upgrade
> torch to the latest version (2.10+) and break everything. If you need to reinstall
> flash-attn, always pin torch first: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`

---

## Step 5: Run the AWQ Demo (Text-Only, Your Video)

Navigate to the low-VRAM-mode directory:

```bash
cd /mnt/c/Users/Armaan/Desktop/PURS/Qwen2.5-Omni/low-VRAM-mode
```

### Option A: Quick modification to the existing script

Edit `low_VRAM_demo_awq.py` — change two things:

1. **Line 171** — change the video path:
```python
# Change this:
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
# To this:
video_path = "/mnt/c/Users/Armaan/Desktop/PURS/OmniZip/assets/example.mp4"
```

2. **Line 165** — disable audio output:
```python
# Change this:
output = model.generate(**inputs, use_audio_in_video=True, return_audio=True)
# To this:
output = model.generate(**inputs, use_audio_in_video=True, return_audio=False)
```

3. **After line 134** (`model.model.load_speakers(spk_path)`) — add:
```python
model.model.disable_talker()
```

4. **Comment out or remove the audio saving lines (181-185)** since there's no audio output:
```python
# audio_file_path = "./output_audio_awq.wav"
# sf.write(
#     audio_file_path,
#     audio.reshape(-1).detach().cpu().numpy(),
#     samplerate=24000,
# )
```

5. **Line 166-167** — fix the output unpacking:
```python
# Change this:
text = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
audio = output[2]
# To this:
text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 low_VRAM_demo_awq.py
```

### Option B: Standalone script (drop-in replacement)

Create a new file `run_video_textonly.py` in the `low-VRAM-mode/` directory:

```python
import torch
import time
import sys
import importlib.util
from awq.models.base import BaseAWQForCausalLM
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from huggingface_hub import hf_hub_download

from modeling_qwen2_5_omni_low_VRAM_mode import (
    Qwen2_5OmniDecoderLayer
)
from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration


def replace_transformers_module():
    original_mod_name = 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni'
    new_mod_path = 'modeling_qwen2_5_omni_low_VRAM_mode.py'
    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]
    spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)
    sys.modules[original_mod_name] = new_mod

replace_transformers_module()


class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "Qwen2_5OmniDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]

    @staticmethod
    def get_model_layers(model):
        return model.thinker.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device):
        model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
        model.thinker.visual = model.thinker.visual.to(device)
        model.thinker.audio_tower = model.thinker.audio_tower.to(device)
        model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(device)
        model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)
        for layer in model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        ))
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            ))
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        ))
        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        ))
        return layers


# ---- Config ----
VIDEO_PATH = "/mnt/c/Users/Armaan/Desktop/PURS/OmniZip/assets/example.mp4"
PROMPT = "Describe this video in detail."
MODEL_PATH = "Qwen/Qwen2.5-Omni-7B-AWQ"
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# ---- Load model (AWQ quantized, low VRAM) ----
print("Loading model...")
model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
    MODEL_PATH,
    model_type="qwen2_5_omni",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)

# Move encoders to GPU
device = "cuda"
model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
model.model.thinker.visual = model.model.thinker.visual.to(device)
model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
model.model.thinker.visual.rotary_pos_emb = model.model.thinker.visual.rotary_pos_emb.to(device)
model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(device)
for layer in model.model.thinker.model.layers:
    layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

# Disable talker to save ~3-4 GB VRAM (text-only output)
model.model.disable_talker()

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
print("Model loaded. Talker disabled (text-only mode).")

# ---- Build input ----
messages = [
    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    {"role": "user", "content": [
        {"type": "video", "video": VIDEO_PATH},
        {"type": "text", "text": PROMPT},
    ]},
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True)
inputs = inputs.to("cuda")

# ---- Generate (text only) ----
print(f"Processing video: {VIDEO_PATH}")
torch.cuda.reset_peak_memory_stats()
start = time.time()

output = model.generate(
    **inputs,
    use_audio_in_video=True,
    return_audio=False,  # text only — no speech synthesis
)

end = time.time()
peak_memory = torch.cuda.max_memory_allocated()

response = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("\n" + "=" * 60)
print("RESPONSE:")
print("=" * 60)
print(response[0])
print("=" * 60)
print(f"Inference time: {end - start:.2f}s")
print(f"Peak GPU memory: {peak_memory / 1024 / 1024:.0f} MB ({peak_memory / 1024 / 1024 / 1024:.1f} GB)")
```

Run it:

```bash
CUDA_VISIBLE_DEVICES=0 python3 run_video_textonly.py
```

---

## Step 6: Verify It Works

Expected output:

```
Loading model...
Model loaded. Talker disabled (text-only mode).
Processing video: /mnt/c/Users/Armaan/Desktop/PURS/OmniZip/assets/example.mp4
============================================================
RESPONSE:
============================================================
[Model's description of the video...]
============================================================
Inference time: XX.XXs
Peak GPU memory: XXXX MB (X.X GB)
```

Peak VRAM should be **~9-12 GB** — well within your 16 GB.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `nvidia-smi` not found in WSL | Update your Windows NVIDIA driver to latest version |
| `flash-attn` fails to build | Make sure you have `torch==2.6.0` installed first. Try `pip install ninja` then retry. Use `flash-attn==2.7.3` specifically |
| `flash-attn` `undefined symbol` error | The prebuilt wheel was built against a different torch version. Fix: `pip uninstall flash-attn -y && pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir` |
| `pip install flash-attn --force-reinstall` upgraded torch | This is a known pitfall. Reinstall torch: `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` |
| `qwen-omni-utils` editable install fails with `UnicodeDecodeError` | Copy to WSL native fs first: `cp -r /mnt/c/.../qwen-omni-utils ~/qwen-omni-utils && cd ~/qwen-omni-utils && pip install -e .` |
| CUDA out of memory | Reduce video resolution: set env var `VIDEO_MAX_PIXELS=100352` (128*28*28) before running |
| `autoawq` import error | Make sure you installed `autoawq==0.2.9` exactly |
| Slow file access from `/mnt/c/` | Copy the files into WSL's native filesystem (`~/`) for faster I/O |
| Model download is slow | First run downloads ~4-5 GB from HuggingFace. Use `huggingface-cli login` if you have a token for faster CDN |

---

## Optional: Reduce Video Resolution (If Still OOM)

Set these environment variables before running to cap token count:

```bash
export VIDEO_MAX_PIXELS=$((128 * 28 * 28))   # 100352 pixels per frame
export VIDEO_TOTAL_PIXELS=$((128 * 28 * 28 * 64))  # total budget
CUDA_VISIBLE_DEVICES=0 python3 run_video_textonly.py
```

---

## Summary

```
WSL2 Ubuntu 22.04
  └── conda env: omnizip (python 3.10)
        ├── torch 2.6.0 + CUDA 12.4
        ├── transformers 4.52.3
        ├── autoawq 0.2.9
        ├── flash-attn 2.7.3 (built from source)
        └── qwen-omni-utils (local install)

Model: Qwen2.5-Omni-7B-AWQ (4-bit quantized)
Mode:  Text-only (talker disabled)
Input: Video + audio (up to 30s)
VRAM:  ~9-12 GB peak / 16 GB available
```
