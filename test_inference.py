import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

print("Loading model...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    '/workspace/model',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='sdpa',
)
processor = Qwen2_5OmniProcessor.from_pretrained('/workspace/model')
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")

video_path = '/workspace/videos/video-mme/action_reasoning/video.mp4'
messages = [
    {'role': 'system', 'content': [{'type': 'text', 'text': 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'}]},
    {'role': 'user', 'content': [
        {'type': 'video', 'video': video_path, 'fps': 1.0, 'max_pixels': 360*420},
        {'type': 'text', 'text': 'What is happening in this video? Answer in one sentence.'},
    ]},
]

print("Processing inputs...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
inputs = processor(
    text=text, audio=audios, images=images, videos=videos,
    return_tensors='pt', padding=True, use_audio_in_video=True,
).to(model.device).to(model.dtype)

print("Running inference...")
with torch.no_grad():
    out = model.generate(**inputs, use_audio_in_video=True, return_audio=False, max_new_tokens=128)

decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
print("\n=== OUTPUT ===")
print(decoded)
