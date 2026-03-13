"""
Qwen2.5-Omni-7B-GPTQ-Int4 Demo Script for RTX 5060 Ti (16GB VRAM)
Uses the pre-quantized GPTQ-Int4 model with manual loading to bypass optimum issues.
"""

import torch
import soundfile as sf
from huggingface_hub import snapshot_download

print("="*60)
print("Qwen2.5-Omni-7B-GPTQ-Int4 Setup for RTX 5060 Ti")
print("="*60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print("="*60)

MODEL_ID = "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4"

print(f"\nDownloading model: {MODEL_ID}")
print("This will download ~10GB on first run...")

model_path = snapshot_download(repo_id=MODEL_ID)
print(f"Model downloaded to: {model_path}")

print("\nLoading processor...")
from transformers import Qwen2_5OmniProcessor
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
print("Processor loaded!")

print("\nLoading model with GPTQ quantization...")
print("Note: Using auto_gptq for loading...")

from auto_gptq import AutoGPTQForCausalLM

# Load the quantized model
model = AutoGPTQForCausalLM.from_quantized(
    model_path,
    device_map="auto",
    use_safetensors=True,
    trust_remote_code=True,
)

print("Model loaded successfully!")
print(f"Peak GPU Memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Test inference
print("\n" + "="*60)
print("Testing text generation...")
print("="*60)

test_prompt = "Hello! What can you help me with today?"
inputs = processor(text=test_prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
    
response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"\nPrompt: {test_prompt}")
print(f"Response: {response}")
print(f"\nPeak GPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
