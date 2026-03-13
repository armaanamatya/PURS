import os
# Limit video total pixel budget — must be set before importing processors
os.environ["VIDEO_MAX_PIXELS"] = str(128 * 28 * 28)  # controls VIDEO_TOTAL_PIXELS in qwen-omni-utils

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
    attn_implementation="sdpa"
)

# Keep visual + audio_tower on CPU (low VRAM mode handles device movement)
# Only move the essential embedding/rotary layers to GPU
device = "cuda"
model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
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
        {"type": "video", "video": VIDEO_PATH,
         "fps": 1.0,                # 1 frame per second (30s video = 30 frames)
         "resized_height": 168,      # low resolution to fit 16GB VRAM
         "resized_width": 168},
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
