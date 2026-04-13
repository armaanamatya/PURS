"""
eval_qwen_omni_mixkv.py
Runs Qwen2.5-Omni-7B + MixKV KV-cache compression on all videos in metadata.json.

MixKV compresses the KV cache per-head during prefill by mixing three scoring
signals (attention importance, key diversity, value norm) with per-head weights.
This is training-free and only affects what gets cached for decode steps.

Supported methods (--select_method):
  snapkv        – attention-score-only selection (no pre-computed scores needed)
  attn          – same as snapkv
  vnorm         – attention + value-norm scoring
  headwisemixkv – full MixKV: head_score*sim + (1-head_score)*(attn+vnorm)
                  Requires pre-computed head similarity scores in --head_score_path

Usage:
    python eval_qwen_omni_mixkv.py --metadata metadata.json --videos /workspace/videos \\
        --output /workspace/results_mixkv.jsonl --budget 256 --select_method snapkv

    python eval_qwen_omni_mixkv.py --metadata metadata.json --videos /workspace/videos \\
        --output /workspace/results_mixkv.jsonl --budget 128 --select_method headwisemixkv \\
        --head_score_path ./visual_head/head_score/qwen_omni_kv_similarity.json

To generate head similarity scores for Qwen2.5-Omni, adapt distribution_qwen.py
from MixKV-main to run a calibration pass on a small dataset (e.g., 50 TextVQA samples).
"""

import argparse
import json
import math
import os
import glob
import shutil
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(_REPO_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers.cache_utils import DynamicCache
from qwen_omni_utils import process_mm_info
from mcq_answer_parse import parse_answer


def cuda_time_ms(fn):
    """Run fn() with CUDA event timing. Returns (elapsed_ms, result)."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out

# ── Constants ──────────────────────────────────────���─────────────────────────

ENV_MODEL_PATH_KEY = "QWEN_OMNI_MODEL_PATH"
DEFAULT_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"
FALLBACK_MODEL_PATH = "/workspace/model"
MODEL_PATH = os.environ.get(ENV_MODEL_PATH_KEY) or DEFAULT_MODEL_PATH

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


# ── Prompt builders (same as eval_qwen_omni.py) ─────────────────────────────

def _format_choice_lines(choices: list) -> str:
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
    head = ("Listen and watch the video carefully. "
            "Select the best answer to the following multiple-choice question. "
            "Respond with only the letter (A, B, C, or D) of the correct option.")
    return head + "\n" + question + "\n" + _format_choice_lines(choices) + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_default(question, choices):
    return ("Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + question + "\n" + _format_choice_lines(choices) + "\n" + _VIDEO_MME_POST_PROMPT)


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


# ── Utilities ───────────���───────────────────────────────���────────────────────

def resolve_model_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


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


# ── Tee loggers ───────��──────────────────────────────────────────────────────

class Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
        self.log.write(f"\n{'='*60}\nRUN (MixKV): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
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
# MixKV: KV Cache Compression for Qwen2.5-Omni
# ══════════════════════════════════════════════════════════════════════════════

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (B, num_kv_heads, S, D) -> (B, num_heads, S, D)."""
    if n_rep == 1:
        return hidden_states
    B, H, S, D = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def load_head_similarity_scores(path: str) -> Optional[torch.Tensor]:
    """Load pre-computed per-layer, per-head KV similarity scores.

    The JSON format follows MixKV: keys are "layer-head-key", values are lists of
    similarity scores across calibration samples. Returns a (num_layers, num_heads)
    tensor of averaged key similarity.
    """
    if not path or not os.path.exists(path):
        return None

    with open(path) as f:
        raw = json.load(f)

    key_scores = {}
    max_layer = max_head = 0
    for full_key, values in raw.items():
        layer_head, score_type = full_key.rsplit("-", 1)
        layer, head = map(int, layer_head.split("-"))
        if score_type == "key":
            key_scores[(layer, head)] = sum(values) / len(values)
            max_layer = max(max_layer, layer)
            max_head = max(max_head, head)

    tensor = torch.zeros(max_layer + 1, max_head + 1)
    for (l, h), s in key_scores.items():
        tensor[l][h] = s
    return tensor


class MixKVCompressor:
    """SnapKV / MixKV-style KV cache compressor for a single attention layer.

    During prefill (q_len > 1):
      1. Compute attention scores from last `window_size` queries over all keys.
      2. Optionally compute key-similarity and value-norm scores.
      3. Combine scores per-head and select top-k tokens.
      4. Store selected + recent-window tokens as compressed KV.

    During decode (q_len == 1): standard cache append.
    """

    def __init__(
        self,
        budget: int = 256,
        window_size: int = 32,
        kernel_size: int = 5,
        layer_idx: int = 0,
        num_kv_heads: int = 4,
        num_kv_groups: int = 7,
        select_method: str = "snapkv",
        head_scores: Optional[torch.Tensor] = None,  # (num_layers, num_kv_heads)
    ):
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups
        self.select_method = select_method
        self.head_scores = head_scores  # (num_layers, num_kv_heads) tensor or None
        self.capacity = max(budget - window_size, 1)

    def _attention_scores(self, query_states, key_states, head_dim):
        """Compute pooled attention scores from last window_size queries."""
        # query_states: (B, num_heads, q_len, D)  key_states: (B, num_kv_heads, kv_len, D)
        # Expand keys for GQA
        key_expanded = _repeat_kv(key_states, self.num_kv_groups)
        window_q = query_states[:, :, -self.window_size:, :]

        attn_w = torch.matmul(window_q, key_expanded.transpose(2, 3)) / math.sqrt(head_dim)
        # Causal mask for the window portion
        ws = self.window_size
        mask = torch.full((ws, ws), torch.finfo(attn_w.dtype).min, device=attn_w.device)
        mask_cond = torch.arange(ws, device=attn_w.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(ws, 1), 0)
        attn_w[:, :, -ws:, -ws:] += mask[None, None, :, :]

        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Mean over window queries, exclude last window from candidates
        attn_mean = attn_w[:, :, :, :-ws].mean(dim=-2)  # (B, num_heads, kv_len - ws)

        # Collapse GQA groups -> (B, num_kv_heads, kv_len - ws)
        attn_mean = attn_mean.view(attn_mean.shape[0], -1, self.num_kv_groups, attn_mean.shape[-1])
        attn_mean = attn_mean.mean(dim=2)

        # Pooling
        attn_pooled = F.avg_pool1d(attn_mean, kernel_size=self.kernel_size,
                                   padding=self.kernel_size // 2, stride=1)
        return attn_pooled  # (B, num_kv_heads, kv_len - ws)

    def _similarity_scores(self, key_states):
        """Compute negated cosine similarity of each key to the mean key."""
        B, H, N, D = key_states.shape
        valid_len = N - self.window_size
        key_norm = F.normalize(key_states, dim=-1)
        key_valid = key_norm[:, :, :valid_len, :]
        key_mean = key_norm.sum(dim=2, keepdim=True) / N
        sim = torch.matmul(key_valid, key_mean.transpose(-2, -1)).squeeze(-1)  # (B, H, valid_len)
        neg_sim = -sim
        sim_min = neg_sim.amin(dim=-1, keepdim=True)
        sim_max = neg_sim.amax(dim=-1, keepdim=True)
        return (neg_sim - sim_min) / (sim_max - sim_min + 1e-8)

    def _value_norm_scores(self, value_states):
        """Compute L2 norm of each value vector, normalized."""
        valid_len = value_states.shape[2] - self.window_size
        v_valid = value_states[:, :, :valid_len, :]
        vnorm = torch.norm(v_valid, p=2, dim=-1)
        vmin = vnorm.amin(dim=-1, keepdim=True)
        vmax = vnorm.amax(dim=-1, keepdim=True)
        return (vnorm - vmin) / (vmax - vmin + 1e-8)

    def compress_kv(self, key_states, query_states, value_states):
        """Select top-k KV tokens per head and return compressed KV.

        Args:
            key_states:   (B, num_kv_heads, kv_len, D)
            query_states: (B, num_heads, q_len, D)
            value_states: (B, num_kv_heads, kv_len, D)

        Returns:
            compressed_keys, compressed_values: (B, num_kv_heads, capacity + window_size, D)
        """
        B, H_kv, N, D = key_states.shape
        if N <= self.budget:
            return key_states, value_states

        attn_score = self._attention_scores(query_states, key_states, D)
        method = self.select_method

        if method in ("snapkv", "attn"):
            combined = attn_score
        elif method == "vnorm":
            vnorm = self._value_norm_scores(value_states)
            att_mean = attn_score.mean(dim=-1, keepdim=True)
            vnorm_mean = vnorm.mean(dim=-1, keepdim=True)
            vnorm_scaled = vnorm * (att_mean / (vnorm_mean + 1e-8))
            combined = attn_score + vnorm_scaled
        elif method == "headwisemixkv":
            sim = self._similarity_scores(key_states)
            vnorm = self._value_norm_scores(value_states)
            att_mean = attn_score.mean(dim=-1, keepdim=True)
            sim_mean = sim.mean(dim=-1, keepdim=True)
            sim_scaled = sim * (att_mean / (sim_mean + 1e-8))
            vnorm_mean = vnorm.mean(dim=-1, keepdim=True)
            vnorm_scaled = vnorm * (att_mean / (vnorm_mean + 1e-8))
            importance = attn_score + vnorm_scaled

            # Per-head mixing: head_score * sim + (1 - head_score) * importance
            heads_k, heads_v = [], []
            for h in range(H_kv):
                if self.head_scores is not None:
                    hs = self.head_scores[self.layer_idx][h].item()
                else:
                    hs = 0.5  # fallback: equal weight
                score_h = hs * sim_scaled[:, h:h+1, :] + (1 - hs) * importance[:, h:h+1, :]
                _, idx = score_h.sort(dim=-1, descending=True)
                sel = idx[:, :, :self.capacity].unsqueeze(-1).expand(-1, -1, -1, D)
                k_sel = key_states[:, h:h+1, :-self.window_size, :].gather(2, sel)
                v_sel = value_states[:, h:h+1, :-self.window_size, :].gather(2, sel)
                k_win = key_states[:, h:h+1, -self.window_size:, :]
                v_win = value_states[:, h:h+1, -self.window_size:, :]
                heads_k.append(torch.cat([k_sel, k_win], dim=2))
                heads_v.append(torch.cat([v_sel, v_win], dim=2))
            return torch.cat(heads_k, dim=1), torch.cat(heads_v, dim=1)
        else:
            combined = attn_score  # fallback

        # Uniform selection across all heads (non-headwise methods)
        _, indices = combined.sort(dim=-1, descending=True)
        sel_idx = indices[:, :, :self.capacity].unsqueeze(-1).expand(-1, -1, -1, D)
        k_compress = key_states[:, :, :-self.window_size, :].gather(2, sel_idx)
        v_compress = value_states[:, :, :-self.window_size, :].gather(2, sel_idx)
        k_window = key_states[:, :, -self.window_size:, :]
        v_window = value_states[:, :, -self.window_size:, :]
        return torch.cat([k_compress, k_window], dim=2), torch.cat([v_compress, v_window], dim=2)


# ── Monkeypatch: SDPA attention forward with MixKV ──────────────────────────

def _make_mixkv_sdpa_forward(original_forward_fn):
    """Create a patched SDPA forward that compresses KV during prefill."""

    def mixkv_sdpa_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Standard QKV projection
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = position_embeddings
        from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # ── MixKV: compress during prefill, normal append during decode ──
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if q_len == 1:
                # Decode: normal cache update
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                # Prefill: compress KV, then store in cache
                if hasattr(self, "_mixkv_compressor"):
                    k_comp, v_comp = self._mixkv_compressor.compress_kv(
                        key_states, query_states, value_states
                    )
                    past_key_value.update(k_comp, v_comp, self.layer_idx, cache_kwargs)
                else:
                    past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ── Standard SDPA output (uses FULL KV for prefill output) ──
        key_for_attn = _repeat_kv(key_states, self.num_key_value_groups)
        value_for_attn = _repeat_kv(value_states, self.num_key_value_groups)

        # Cast if needed
        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_for_attn = key_for_attn.to(target_dtype)
            value_for_attn = value_for_attn.to(target_dtype)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_for_attn.shape[-2]]

        query_states = query_states.contiguous()
        key_for_attn = key_for_attn.contiguous()
        value_for_attn = value_for_attn.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states, key_for_attn, value_for_attn,
            attn_mask=causal_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=causal_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return mixkv_sdpa_forward


def apply_mixkv_to_model(
    model,
    budget: int,
    window_size: int,
    select_method: str,
    head_scores: Optional[torch.Tensor],
):
    """Monkeypatch all thinker attention layers with MixKV compression."""
    thinker = model.thinker if hasattr(model, "thinker") else model
    # Find decoder layers
    if hasattr(thinker, "model") and hasattr(thinker.model, "layers"):
        layers = thinker.model.layers
    elif hasattr(thinker, "layers"):
        layers = thinker.layers
    else:
        raise RuntimeError("Cannot find decoder layers in model")

    num_layers = len(layers)
    print(f"Applying MixKV to {num_layers} layers: budget={budget}, window={window_size}, method={select_method}")

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        num_kv_heads = attn.num_key_value_heads
        num_kv_groups = attn.num_key_value_groups

        compressor = MixKVCompressor(
            budget=budget,
            window_size=window_size,
            kernel_size=5,
            layer_idx=i,
            num_kv_heads=num_kv_heads,
            num_kv_groups=num_kv_groups,
            select_method=select_method,
            head_scores=head_scores,
        )
        attn._mixkv_compressor = compressor

        # Patch forward method
        import types
        patched_fn = _make_mixkv_sdpa_forward(attn.forward)
        attn.forward = types.MethodType(patched_fn, attn)

    print(f"MixKV applied. Effective KV capacity per head: {budget} tokens ({budget - window_size} selected + {window_size} window)")


# ── Model loading ──────────��─────────────────────────────────────────────────

def load_model(dtype_name, budget, window_size, select_method, head_score_path):
    dt = resolve_model_dtype(dtype_name)
    print(f"Loading model from {MODEL_PATH} (dtype={dtype_name}) ...")

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dt,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    head_scores = load_head_similarity_scores(head_score_path)
    if select_method == "headwisemixkv" and head_scores is None:
        print("WARNING: headwisemixkv selected but no head_score_path provided. Falling back to equal weights (0.5).")

    apply_mixkv_to_model(model, budget, window_size, select_method, head_scores)

    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    return model, processor


# ── Inference ─────────��──────────────────────────────────────────────────────

_OMNI_GENERATE_BATCH_DROP = frozenset({"images", "return_tensors", "text"})


def _prepare_omni_inputs(model, inputs):
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)
    return inputs


def _generation_output_token_ids(gen_out):
    return gen_out.sequences if hasattr(gen_out, "sequences") else gen_out


def run_inference(model, processor, video_path, dataset, question, choices,
                  fps, max_pixels, max_new_tokens, use_audio, measure_prefill=False):
    prompt = build_user_prompt_for_dataset(dataset, question, choices)
    system_text = SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": prompt},
        ]},
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

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=effective_use_audio,
    )
    inputs = _prepare_omni_inputs(model, inputs)

    tokenizer = processor.tokenizer
    gen_kw = {
        "use_audio_in_video": effective_use_audio,
        "return_audio": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if hasattr(model, "thinker"):
        gen_kw["thinker_max_new_tokens"] = max_new_tokens
        gen_kw["thinker_do_sample"] = False
    else:
        gen_kw["max_new_tokens"] = max_new_tokens
        gen_kw["do_sample"] = False

    gen_in = {k: v for k, v in inputs.items() if k not in _OMNI_GENERATE_BATCH_DROP}

    timing = {}
    if measure_prefill:
        prefill_kw = dict(gen_kw)
        if "thinker_max_new_tokens" in prefill_kw:
            prefill_kw["thinker_max_new_tokens"] = 1
        else:
            prefill_kw["max_new_tokens"] = 1
        with torch.no_grad():
            prefill_ms, _ = cuda_time_ms(lambda: model.generate(**gen_in, **prefill_kw))
        timing["prefill_ms"] = round(prefill_ms, 2)

    with torch.no_grad():
        if measure_prefill:
            e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))
            timing["e2e_ms"] = round(e2e_ms, 2)
        else:
            raw_out = model.generate(**gen_in, **gen_kw)

    seq_ids = _generation_output_token_ids(raw_out)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, timing


# ── Main ─────────────────��─────────────────────────���─────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni + MixKV")
    parser.add_argument("--model", default=None)
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--videos", default="/workspace/videos")
    parser.add_argument("--output", default="/workspace/results_mixkv.jsonl")
    parser.add_argument("--log", default="/workspace/eval_mixkv.log")
    parser.add_argument("--errors_log", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_pixels", type=int, default=360*420)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--vram_log", default="/workspace/vram_log_mixkv.jsonl")
    parser.add_argument("--stderr_log", default=None)
    # MixKV-specific
    parser.add_argument("--budget", type=int, default=256,
                        help="KV cache token budget per head (includes window)")
    parser.add_argument("--window_size", type=int, default=32,
                        help="Recent token window kept in full")
    parser.add_argument("--select_method", default="snapkv",
                        choices=["snapkv", "attn", "vnorm", "headwisemixkv"],
                        help="MixKV token selection method")
    parser.add_argument("--head_score_path", default=None,
                        help="Path to pre-computed head similarity JSON (required for headwisemixkv)")
    parser.add_argument("--measure_prefill", action="store_true",
                        help="Measure prefill time (TTFT) via generate(max_new_tokens=1) with CUDA events.")
    args = parser.parse_args()

    global MODEL_PATH
    if args.model:
        MODEL_PATH = args.model
    elif not os.path.exists(MODEL_PATH) and os.path.exists(FALLBACK_MODEL_PATH):
        MODEL_PATH = FALLBACK_MODEL_PATH

    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found. Use --no_audio or install ffmpeg.\n")

    errors_log_path = args.errors_log or os.path.join(os.path.dirname(args.log) or ".", "errors.log")
    for path in [args.log, args.output, args.vram_log, errors_log_path]:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if args.stderr_log:
        os.makedirs(os.path.dirname(args.stderr_log) or ".", exist_ok=True)
        _stderr_f = open(args.stderr_log, "a", encoding="utf-8")
        sys.stderr = StderrTee(_stderr_f, sys.__stderr__)

    tee = Tee(args.log)
    sys.stdout = tee

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} entries")

    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    runnable = [e for e in meta if e.get("questions")]
    skipped_no_qa = len(meta) - len(runnable)
    if skipped_no_qa:
        print(f"Skipping {skipped_no_qa} entries with no Q&A")
    print(f"Running {len(runnable)} entries with MixKV ({args.select_method}, budget={args.budget})\n")

    if not runnable:
        print("Nothing to run.")
        return

    model, processor = load_model(
        args.dtype, args.budget, args.window_size, args.select_method, args.head_score_path,
    )

    correct = total = skipped_no_video = 0
    results = []

    with open(args.output, "w") as out_f, open(args.vram_log, "w") as vram_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video not found")
                skipped_no_video += 1
                continue

            use_audio = (not args.no_audio) and check_video_has_audio(video_path)

            for q in entry["questions"]:
                question = q["question"]
                choices = q["choices"]
                answer = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset = entry.get("dataset", "")

                try:
                    torch.cuda.reset_peak_memory_stats()
                    pred, reasoning, timing = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        args.fps, args.max_pixels, args.max_new_tokens, use_audio,
                        measure_prefill=args.measure_prefill,
                    )
                    vram_entry = {
                        "entry": entry_label, "task_type": task_type,
                        "duration_s": entry.get("duration_s"),
                        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                        "after_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                        "after_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                        **timing,
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                except Exception as e:
                    import traceback
                    print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                    with open(errors_log_path, "a") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{traceback.format_exc()}\n")
                    pred, reasoning = "ERROR", str(e)
                    timing = {}

                torch.cuda.empty_cache()
                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "model_variant": "qwen2.5-omni",
                    "dataset": entry.get("dataset"),
                    "task_type": task_type,
                    "duration_s": entry.get("duration_s"),
                    "question": question, "choices": choices,
                    "answer": answer, "prediction": pred,
                    "correct": is_correct, "reasoning": reasoning,
                    "method": f"mixkv-{args.select_method}",
                    "budget": args.budget,
                    **timing,
                }
                out_f.write(json.dumps(result) + "\n")
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer}")

    acc = correct / total if total else 0
    print(f"\n{'='*50}")
    print(f"Model: qwen2.5-omni + MixKV ({args.select_method})")
    print(f"Budget: {args.budget} | Window: {args.window_size}")
    print(f"Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Results: {args.output}")

    datasets = {}
    for r in results:
        d = r["dataset"]
        datasets.setdefault(d, {"correct": 0, "total": 0})
        datasets[d]["total"] += 1
        if r["correct"]:
            datasets[d]["correct"] += 1
    print("\nPer-dataset:")
    for ds, s in sorted(datasets.items()):
        print(f"  {ds:<20} {s['correct']}/{s['total']} = {s['correct']/s['total']:.2%}")


if __name__ == "__main__":
    main()
