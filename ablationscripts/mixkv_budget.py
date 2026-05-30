"""
mixkv_budget.py
Ablation: sweep MixKV budget in {64, 128, 256, 512} with select_method="snapkv".

Budget controls the total KV tokens kept per head after prefill compression:
  budget = selected_tokens + window_tokens
  capacity = max(budget - window_size, 1)  (tokens actively scored and selected)

Compression levels at window_size=32:
  budget=512  -> capacity=480 selected + 32 window  (mild compression)
  budget=256  -> capacity=224 selected + 32 window  (moderate, paper default)
  budget=128  -> capacity= 96 selected + 32 window  (aggressive)
  budget= 64  -> capacity= 32 selected + 32 window  (very aggressive;
                  50% of kept tokens are the recency window, 50% are scored)

All use snapkv (attention-only scoring) to isolate the budget effect cleanly.

Usage (single value):
    python mixkv_budget.py --value 128 --metadata metadata.json \\
        --videos /data/armaan/purs/videos --model /data/armaan/models/Qwen2.5-Omni-7B

Usage (all values):
    python mixkv_budget.py --metadata metadata.json \\
        --videos /data/armaan/purs/videos --model /data/armaan/models/Qwen2.5-Omni-7B

Output:
    ablation_outputs/mixkv_budget/budget_64/
    ablation_outputs/mixkv_budget/budget_128/
    ablation_outputs/mixkv_budget/budget_256/
    ablation_outputs/mixkv_budget/budget_512/
"""

import argparse
import json
import math
import os
import glob
import random
import shutil
import sys
import time
import torch
import torch.nn.functional as F
import types
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_REPO_ROOT, "..")
OMNIZIP_DIR = os.path.join(_PROJECT_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from mcq_answer_parse import parse_answer

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
FALLBACK_MODEL_PATH = "/workspace/model"

DEFAULT_FPS = 2.0
DEFAULT_MAX_PIXELS = 100352
DEFAULT_MAX_FRAMES_VIDEOMME = 768
DEFAULT_MAX_FRAMES_OTHER = 128
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1

FIXED_METHOD = "snapkv"
FIXED_WINDOW_SIZE = 32

ALL_BUDGETS = [64, 128, 256, 512]

SYSTEM_PROMPT_DEFAULT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
SYSTEM_MCQ_SUFFIX = (
    "For multiple-choice questions, reply with only one letter: A, B, C, or D. "
    "Do not explain, do not ask follow-up questions, and do not add text after the letter."
)

_VIDEO_MME_OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)
_VIDEO_MME_POST_PROMPT = "The best answer is:"
_WORLD_SENSE_SYS = (
    "Carefully watch this video and pay attention to every detail. "
    "Based on your observations, select the best option that accurately addresses the question."
)
_WORLD_SENSE_FRAMES_AUDIO = (
    "\nThese are the frames of a video and the corresponding audio. "
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
)

MODEL_LOADED_ALLOC_GB: Optional[float] = None
MODEL_LOADED_RESERVED_GB: Optional[float] = None

# ── Prompt builders ───────────────────────────────────────────────────────────

def _format_choice_lines(choices):
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))

def _build_user_prompt_video_mme(question, choices):
    return _VIDEO_MME_OPTION_PROMPT + "\n" + question + "\n" + _format_choice_lines(choices) + "\n" + _VIDEO_MME_POST_PROMPT

def _build_user_prompt_worldsense(question, choices):
    parts = [_WORLD_SENSE_SYS, _WORLD_SENSE_FRAMES_AUDIO, question + "\n"]
    for op in choices:
        parts.append(op + "\n")
    return "".join(parts)

def _build_user_prompt_daily_omni(question, choices):
    head = (
        "Listen and watch the video carefully. "
        "Select the best answer to the following multiple-choice question. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    return head + "\n" + question + "\n" + _format_choice_lines(choices) + "\n" + _VIDEO_MME_POST_PROMPT

def _build_user_prompt_default(question, choices):
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        + question + "\n" + _format_choice_lines(choices) + "\n" + _VIDEO_MME_POST_PROMPT
    )

def _canonicalize(ds):
    return (ds or "").strip().lower().replace("_", "-").replace(" ", "-")

def build_user_prompt_for_dataset(dataset, question, choices):
    n = _canonicalize(dataset)
    if n in {"video-mme", "videomme"}:
        return _build_user_prompt_video_mme(question, choices)
    if n == "worldsense":
        return _build_user_prompt_worldsense(question, choices)
    if n in {"daily-omni", "dailyomni"}:
        return _build_user_prompt_daily_omni(question, choices)
    return _build_user_prompt_default(question, choices)

# ── Utilities ─────────────────────────────────────────────────────────────────

def cuda_time_ms(fn):
    """Run fn() with CUDA event timing. Returns (elapsed_ms, result)."""
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    r = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), r

def set_run_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _capture_current_vram_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )

def _record_model_loaded_vram():
    global MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB
    MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB = _capture_current_vram_gb()
    print(
        f"Model loaded. VRAM: {MODEL_LOADED_ALLOC_GB:.1f} GB allocated, "
        f"{MODEL_LOADED_RESERVED_GB:.1f} GB reserved"
    )

def check_video_has_audio(path):
    try:
        import av
        c = av.open(path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False

def resolve_video_path(file_field, videos_dir):
    if os.path.exists(file_field):
        return file_field
    normalized = file_field.replace("\\", "/")
    filename = normalized.split("/")[-1]
    stem = filename.rsplit(".", 1)[0]
    candidate = os.path.join(videos_dir, filename)
    if os.path.exists(candidate):
        return candidate
    for ext in ("mp4", "mkv", "webm", "avi"):
        matches = glob.glob(os.path.join(videos_dir, "**", f"{stem}.{ext}"), recursive=True)
        if matches:
            return matches[0]
    return None

def resolve_model_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

# ── Tee loggers ───────────────────────────────────────────────────────────────

class Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")
        self.log.write(f"\n{'='*60}\nRUN (mixkv_budget): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
        self.log.flush()

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.log.close()

class StderrTee:
    def __init__(self, log_file, terminal):
        self.log = log_file
        self.terminal = terminal

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

# ══════════════════════════════════════════════════════════════════════════════
# MixKV: KV Cache Compression
# ══════════════════════════════════════════════════════════════════════════════

def _repeat_kv(hidden_states, n_rep):
    """Expand KV heads for GQA: (B, num_kv_heads, S, D) -> (B, num_heads, S, D)."""
    if n_rep == 1:
        return hidden_states
    B, H, S, D = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


class MixKVCompressor:
    """
    KV cache compressor using SnapKV (attention-only) scoring.

    This ablation uses select_method="snapkv" across all budget values to isolate
    the effect of the budget parameter independent of scoring strategy.

    Budget math (window_size=32):
      budget=512 -> capacity = max(512-32, 1) = 480  (mild: keeps most tokens)
      budget=256 -> capacity = max(256-32, 1) = 224  (moderate, paper default)
      budget=128 -> capacity = max(128-32, 1) =  96  (aggressive)
      budget= 64 -> capacity = max( 64-32, 1) =  32  (very aggressive;
                    equal split between scored selection and recency window)

    Final cached KV per head: [selected_capacity_tokens | window_size_tokens]
    """

    def __init__(self, budget=256, window_size=32, kernel_size=5, layer_idx=0,
                 num_kv_heads=4, num_kv_groups=7):
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups
        self.capacity = max(budget - window_size, 1)

    def _attn_scores(self, Q, K, head_dim):
        """
        Pooled attention scores from the last window_size queries over all keys.

        1. window queries Q_w = Q[:, :, -window_size:, :]
        2. attn = softmax(Q_w @ K^T / sqrt(d))    shape: (B, H, W, kv_len)
        3. causal mask applied within the window
        4. mean over window dim, exclude window tokens as candidates -> (B, H_kv, kv_len-W)
        5. avg_pool1d smoothing with kernel_size=5

        Returns: (B, num_kv_heads, kv_len - window_size)
        """
        K_exp = _repeat_kv(K, self.num_kv_groups)
        Q_w = Q[:, :, -self.window_size:, :]
        ws = self.window_size

        attn_w = torch.matmul(Q_w, K_exp.transpose(2, 3)) / math.sqrt(head_dim)
        # Causal mask for the window diagonal
        mask = torch.full((ws, ws), torch.finfo(attn_w.dtype).min, device=attn_w.device)
        mc = torch.arange(ws, device=attn_w.device)
        mask.masked_fill_(mc < (mc + 1).view(ws, 1), 0)
        attn_w[:, :, -ws:, -ws:] += mask[None, None, :, :]

        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(Q.dtype)
        # Mean over window queries; exclude window tokens from candidate pool
        attn_mean = attn_w[:, :, :, :-ws].mean(dim=-2)  # (B, num_heads, kv_len-W)
        # Collapse GQA groups -> (B, num_kv_heads, kv_len-W)
        attn_mean = attn_mean.view(
            attn_mean.shape[0], -1, self.num_kv_groups, attn_mean.shape[-1]
        ).mean(dim=2)
        return F.avg_pool1d(
            attn_mean, kernel_size=self.kernel_size,
            padding=self.kernel_size // 2, stride=1
        )

    def compress(self, K, Q, V):
        """
        Select top-capacity KV tokens using attention scores (snapkv).

        Args:
            K: (B, num_kv_heads, kv_len, D)
            Q: (B, num_heads, q_len, D)
            V: (B, num_kv_heads, kv_len, D)

        Returns:
            (K_compressed, V_compressed): (B, num_kv_heads, capacity+window_size, D)
            Unchanged if kv_len <= budget.
        """
        B, H_kv, N, D = K.shape
        if N <= self.budget:
            return K, V

        # snapkv: score = attn_pooled only
        combined = self._attn_scores(Q, K, D)

        _, indices = combined.sort(dim=-1, descending=True)
        sel_idx = indices[:, :, :self.capacity].unsqueeze(-1).expand(-1, -1, -1, D)
        # Gather top-capacity from non-window portion, append window
        Kc = torch.cat([
            K[:, :, :-self.window_size, :].gather(2, sel_idx),
            K[:, :, -self.window_size:, :]
        ], dim=2)
        Vc = torch.cat([
            V[:, :, :-self.window_size, :].gather(2, sel_idx),
            V[:, :, -self.window_size:, :]
        ], dim=2)
        return Kc, Vc


def _make_mixkv_forward(original_fwd):
    """Create a patched SDPA forward that compresses KV during prefill."""

    def mixkv_sdpa_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states   = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if q_len == 1:
                # Decode step: normal append
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                # Prefill: compress, then cache compressed KV
                if hasattr(self, "_mixkv_compressor"):
                    k_comp, v_comp = self._mixkv_compressor.compress(
                        key_states, query_states, value_states
                    )
                    past_key_value.update(k_comp, v_comp, self.layer_idx, cache_kwargs)
                else:
                    past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_for_attn   = _repeat_kv(key_states, self.num_key_value_groups)
        value_for_attn = _repeat_kv(value_states, self.num_key_value_groups)

        # dtype cast (mirrors HF SDPA impl)
        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states   = query_states.to(target_dtype)
            key_for_attn   = key_for_attn.to(target_dtype)
            value_for_attn = value_for_attn.to(target_dtype)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_for_attn.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            query_states.contiguous(), key_for_attn.contiguous(), value_for_attn.contiguous(),
            attn_mask=causal_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=causal_mask is None and q_len > 1,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value

    return mixkv_sdpa_forward


def apply_mixkv_to_model(model, budget, window_size):
    """Monkeypatch all thinker attention layers with MixKV (snapkv method)."""
    thinker = model.thinker if hasattr(model, "thinker") else model
    if hasattr(thinker, "model") and hasattr(thinker.model, "layers"):
        layers = thinker.model.layers
    elif hasattr(thinker, "layers"):
        layers = thinker.layers
    else:
        raise RuntimeError("Cannot find decoder layers in model")

    capacity = max(budget - window_size, 1)
    print(f"Applying MixKV (snapkv) to {len(layers)} layers: budget={budget}, window={window_size}, capacity={capacity}")
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        compressor = MixKVCompressor(
            budget=budget, window_size=window_size, kernel_size=5, layer_idx=i,
            num_kv_heads=attn.num_key_value_heads,
            num_kv_groups=attn.num_key_value_groups,
        )
        attn._mixkv_compressor = compressor
        patched = _make_mixkv_forward(attn.forward)
        attn.forward = types.MethodType(patched, attn)

    print(f"MixKV applied. Capacity per head: {budget} tokens ({capacity} selected + {window_size} window)")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path, dtype_name, budget):
    dt = resolve_model_dtype(dtype_name)
    print(f"Loading model from {model_path} (dtype={dtype_name}) ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dt, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    apply_mixkv_to_model(model, budget, FIXED_WINDOW_SIZE)
    _record_model_loaded_vram()
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

_DROP_KEYS = frozenset({"images", "return_tensors", "text"})


def run_inference(model, processor, video_path, dataset, question, choices,
                  fps, max_pixels, max_new_tokens, use_audio,
                  max_frames=None, temperature=0.1):
    prompt = build_user_prompt_for_dataset(dataset, question, choices)
    system_text = SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX
    video_element = {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels}
    if max_frames is not None:
        video_element["max_frames"] = max_frames
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [video_element, {"type": "text", "text": prompt}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    effective_use_audio = use_audio
    try:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=effective_use_audio)
    except Exception:
        if not effective_use_audio:
            raise
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        effective_use_audio = False

    if not videos or videos[0] is None or getattr(videos[0], "shape", None) is None or videos[0].shape[0] <= 0:
        raise ValueError("Decoded 0 video frames. Try lower --fps/--max_pixels.")
    orig_nframes = int(videos[0].shape[0])

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=effective_use_audio,
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)

    tokenizer = processor.tokenizer
    do_sample = temperature > 0
    gen_kw = {
        "use_audio_in_video": effective_use_audio,
        "return_audio": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if hasattr(model, "thinker"):
        gen_kw["thinker_max_new_tokens"] = max_new_tokens
        gen_kw["thinker_do_sample"] = do_sample
        if do_sample:
            gen_kw["thinker_temperature"] = temperature
    else:
        gen_kw["max_new_tokens"] = max_new_tokens
        gen_kw["do_sample"] = do_sample
        if do_sample:
            gen_kw["temperature"] = temperature

    gen_in = {k: v for k, v in inputs.items() if k not in _DROP_KEYS}

    # Measure prefill time (TTFT) with CUDA events (generate with max_new_tokens=1)
    prefill_kw = dict(gen_kw)
    if "thinker_max_new_tokens" in prefill_kw:
        prefill_kw["thinker_max_new_tokens"] = 1
    else:
        prefill_kw["max_new_tokens"] = 1
    with torch.no_grad():
        prefill_ms, _ = cuda_time_ms(lambda: model.generate(**gen_in, **prefill_kw))

    # Full generation
    with torch.no_grad():
        e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))

    seq_ids = raw_out.sequences if hasattr(raw_out, "sequences") else raw_out
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, orig_nframes, {"prefill_ms": round(prefill_ms, 2), "e2e_ms": round(e2e_ms, 2)}


# ── Per-budget run ─────────────────────────────────────────────────────────────

def run_budget(budget, args, meta, out_dir):
    """Run inference for one budget value and write results to out_dir/budget_{budget}/."""
    capacity = max(budget - FIXED_WINDOW_SIZE, 1)
    val_dir = os.path.join(out_dir, f"budget_{budget}")
    os.makedirs(val_dir, exist_ok=True)

    results_path = os.path.join(val_dir, "results.jsonl")
    vram_path    = os.path.join(val_dir, "vram_log.jsonl")
    console_path = os.path.join(val_dir, "console.log")

    old_stdout = sys.stdout
    tee = Tee(console_path)
    sys.stdout = tee

    print(f"\n{'='*60}")
    print(f"BUDGET ABLATION: budget={budget}")
    print(f"  method={FIXED_METHOD}, window_size={FIXED_WINDOW_SIZE}")
    print(f"  capacity={capacity} selected + {FIXED_WINDOW_SIZE} window = {capacity + FIXED_WINDOW_SIZE} total")
    if budget == 64:
        print(f"  NOTE: very aggressive. 50% of kept tokens are the recency window.")
    print(f"{'='*60}\n")

    try:
        model, processor = load_model(args.model, args.dtype, budget)
    except Exception as e:
        print(f"FATAL: model load failed: {e}")
        sys.stdout = old_stdout
        tee.close()
        return None

    runnable = [e for e in meta if e.get("questions")]
    correct = total = skipped_no_video = 0
    results = []

    with open(results_path, "w") as out_f, open(vram_path, "w") as vram_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video not found")
                skipped_no_video += 1
                continue

            use_audio = (not args.no_audio) and check_video_has_audio(video_path)

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset   = entry.get("dataset", "")
                ds_canon  = _canonicalize(dataset)
                max_frames = (
                    args.max_frames_videomme
                    if ds_canon in {"video-mme", "videomme"}
                    else args.max_frames_other
                )

                before_alloc_gb, before_reserved_gb = _capture_current_vram_gb()
                torch.cuda.reset_peak_memory_stats()
                try:
                    pred, reasoning, orig_nf, timing = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        args.fps, args.max_pixels, args.max_new_tokens, use_audio,
                        max_frames=max_frames, temperature=args.temperature,
                    )
                    status = "ok"
                    error_info = {}
                except Exception as e:
                    import traceback
                    print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                    traceback.print_exc()
                    pred, reasoning, orig_nf, timing = "ERROR", str(e), 0, {}
                    status = "error"
                    error_info = {"error_type": type(e).__name__, "error_message": str(e)}

                after_alloc_gb, after_reserved_gb = _capture_current_vram_gb()
                vram_delta_gb = after_alloc_gb - before_alloc_gb
                vram_entry = {
                    "entry": entry_label, "task_type": task_type,
                    "status": status,
                    "method": f"mixkv-{FIXED_METHOD}",
                    "budget": budget,
                    "capacity": capacity,
                    "window_size": FIXED_WINDOW_SIZE,
                    "duration_s": entry.get("duration_s"),
                    "orig_frames": orig_nf,
                    "model_loaded_alloc_gb": round(MODEL_LOADED_ALLOC_GB or 0.0, 2),
                    "model_loaded_reserved_gb": round(MODEL_LOADED_RESERVED_GB or 0.0, 2),
                    "before_alloc_gb": round(before_alloc_gb, 2),
                    "before_reserved_gb": round(before_reserved_gb, 2),
                    "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                    "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                    "after_alloc_gb": round(after_alloc_gb, 2),
                    "after_reserved_gb": round(after_reserved_gb, 2),
                    "vram_delta_gb": round(vram_delta_gb, 3),
                    **timing, **error_info,
                }
                vram_f.write(json.dumps(vram_entry) + "\n")
                vram_f.flush()
                torch.cuda.empty_cache()

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "ablation": "mixkv_budget",
                    "select_method": FIXED_METHOD,
                    "budget": budget,
                    "capacity": capacity,
                    "window_size": FIXED_WINDOW_SIZE,
                    "dataset": dataset, "task_type": task_type,
                    "duration_s": entry.get("duration_s"),
                    "question": question, "choices": choices,
                    "answer": answer, "prediction": pred,
                    "correct": is_correct, "reasoning": reasoning,
                    "orig_frames": orig_nf,
                    "temperature": args.temperature,
                    **timing,
                }
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                results.append(result)

                sym = "+" if is_correct else "-"
                print(f"  [{sym}] {entry_label} [{task_type}] pred={pred} ans={answer} "
                      f"prefill={timing.get('prefill_ms','?')}ms vram_delta={vram_delta_gb:+.2f}GB")

    acc = correct / total if total else 0.0
    print(f"\nBudget={budget}: {correct}/{total} = {acc:.2%} (skipped={skipped_no_video})")

    # Per-dataset breakdown
    datasets = {}
    for r in results:
        d = r["dataset"]
        datasets.setdefault(d, {"correct": 0, "total": 0})
        datasets[d]["total"] += 1
        if r["correct"]:
            datasets[d]["correct"] += 1

    summary = {
        "ablation": "mixkv_budget",
        "select_method": FIXED_METHOD,
        "budget": budget,
        "capacity": capacity,
        "window_size": FIXED_WINDOW_SIZE,
        "total": total, "correct": correct, "accuracy": acc,
        "skipped_no_video": skipped_no_video,
        "per_dataset": {
            d: {"correct": s["correct"], "total": s["total"],
                "accuracy": s["correct"] / s["total"] if s["total"] else 0.0}
            for d, s in datasets.items()
        },
        "results_path": results_path,
        "vram_path": vram_path,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(val_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    sys.stdout = old_stdout
    tee.close()

    # Free model memory before next budget
    del model
    torch.cuda.empty_cache()

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MixKV ablation: sweep budget at fixed method=snapkv"
    )
    parser.add_argument("--value", type=int, default=None, choices=ALL_BUDGETS,
                        help="Single budget value to run. Omit to run all.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--videos", default="/data/armaan/purs/videos")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: ablation_outputs/mixkv_budget/)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max_frames_videomme", type=int, default=DEFAULT_MAX_FRAMES_VIDEOMME)
    parser.add_argument("--max_frames_other", type=int, default=DEFAULT_MAX_FRAMES_OTHER)
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--category", default=None,
                        help="Filter by dataset or task_type")
    args = parser.parse_args()

    set_run_seed(args.seed)

    if not os.path.exists(args.model):
        if os.path.exists(FALLBACK_MODEL_PATH):
            args.model = FALLBACK_MODEL_PATH
        else:
            print(f"WARNING: model not found at {args.model}")

    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found. Use --no_audio or install ffmpeg.")

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ablation_outputs", "mixkv_budget"
    )
    os.makedirs(out_dir, exist_ok=True)

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} metadata entries")
    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category
                or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    budgets_to_run = [args.value] if args.value else ALL_BUDGETS
    print(f"\nMixKV budget ablation: will run budgets={budgets_to_run}")
    print(f"Fixed: method={FIXED_METHOD}, window_size={FIXED_WINDOW_SIZE}")
    print(f"Output: {out_dir}\n")

    # Print compression level summary
    print("Compression levels:")
    for b in budgets_to_run:
        cap = max(b - FIXED_WINDOW_SIZE, 1)
        pct_window = FIXED_WINDOW_SIZE / b * 100
        print(f"  budget={b:>4}: capacity={cap:>4} selected + {FIXED_WINDOW_SIZE} window  "
              f"(window={pct_window:.0f}% of budget)")
    print()

    sweep_summaries = []
    for budget in budgets_to_run:
        print(f"\n{'#'*60}")
        print(f"# Running budget: {budget}")
        print(f"{'#'*60}")
        summary = run_budget(budget, args, meta, out_dir)
        if summary:
            sweep_summaries.append(summary)

    # Write sweep summary
    sweep_path = os.path.join(out_dir, "sweep_summary.json")
    with open(sweep_path, "w") as f:
        json.dump({
            "ablation": "mixkv_budget",
            "select_method": FIXED_METHOD,
            "window_size": FIXED_WINDOW_SIZE,
            "budgets_run": budgets_to_run,
            "timestamp": datetime.now().isoformat(),
            "results": sweep_summaries,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"{'Budget':<10} {'Capacity':<12} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 52)
    for s in sweep_summaries:
        print(f"  {s['budget']:<8} {s['capacity']:<10} {s['accuracy']:>9.2%} "
              f"{s['correct']:>6}/{s['total']:<6}")
    print(f"\nSweep summary: {sweep_path}")


if __name__ == "__main__":
    main()
