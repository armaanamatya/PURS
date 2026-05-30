"""
mixkv_method.py
Ablation: sweep MixKV select_method in {"snapkv", "vnorm", "headwisemixkv"}
at a fixed budget=256, window_size=32.

Scoring methods:
  snapkv        – attention-only (SnapKV baseline): score = attn_pooled
  vnorm         – attn + value-norm: score = attn_pooled + vnorm * scale
  headwisemixkv – per-head mix of diversity and (attn+vnorm):
                  score_h = hs_h * sim_scaled_h + (1-hs_h) * (attn_h + vnorm_h)
                  Without calibrated head scores (head_scores=None), hs_h = 0.5 (equal mix).
                  Full headwisemixkv requires running MixKV-main/MixKV-main/distribution_qwen.py
                  to calibrate per-head similarity scores first.

Usage (single value):
    python mixkv_method.py --value snapkv --metadata metadata.json \\
        --videos /data/armaan/purs/videos --model /data/armaan/models/Qwen2.5-Omni-7B

Usage (all values):
    python mixkv_method.py --metadata metadata.json \\
        --videos /data/armaan/purs/videos --model /data/armaan/models/Qwen2.5-Omni-7B

Output:
    ablation_outputs/mixkv_method/snapkv/
    ablation_outputs/mixkv_method/vnorm/
    ablation_outputs/mixkv_method/headwisemixkv_uncalibrated/
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

FIXED_BUDGET = 256
FIXED_WINDOW_SIZE = 32

ALL_METHODS = ["snapkv", "vnorm", "headwisemixkv"]
# headwisemixkv without calibrated scores -> label as "uncalibrated"
METHOD_DIR_NAMES = {
    "snapkv": "snapkv",
    "vnorm": "vnorm",
    "headwisemixkv": "headwisemixkv_uncalibrated",
}

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
        self.log.write(f"\n{'='*60}\nRUN (mixkv_method): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
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
    KV cache compressor implementing SnapKV / MixKV-style selection.

    During prefill (q_len > 1):
      1. Compute attention scores from last `window_size` queries over all keys.
      2. Optionally compute key-similarity and value-norm scores.
      3. Combine scores per-head and select top-k tokens.
      4. Store [selected_tokens | window_tokens] as compressed KV.

    During decode (q_len == 1): standard cache append (no compression).

    Methods:
      snapkv        – score = attn_pooled
      vnorm         – score = attn_pooled + vnorm * scale
      headwisemixkv – per-head: score_h = hs_h * sim_scaled + (1-hs_h) * (attn + vnorm)
                      hs_h from calibration JSON; falls back to 0.5 if not provided.
    """

    def __init__(self, budget=256, window_size=32, kernel_size=5, layer_idx=0,
                 num_kv_heads=4, num_kv_groups=7, select_method="snapkv",
                 head_scores=None):
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.layer_idx = layer_idx
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups
        self.select_method = select_method
        self.head_scores = head_scores  # (num_layers, num_kv_heads) tensor or None
        self.capacity = max(budget - window_size, 1)

    def _attn_scores(self, Q, K, head_dim):
        """Pooled attention scores from last window_size queries. (B, num_kv_heads, kv_len-W)"""
        K_exp = _repeat_kv(K, self.num_kv_groups)
        Q_w = Q[:, :, -self.window_size:, :]
        ws = self.window_size
        attn_w = torch.matmul(Q_w, K_exp.transpose(2, 3)) / math.sqrt(head_dim)
        # Causal mask for window portion
        mask = torch.full((ws, ws), torch.finfo(attn_w.dtype).min, device=attn_w.device)
        mc = torch.arange(ws, device=attn_w.device)
        mask.masked_fill_(mc < (mc + 1).view(ws, 1), 0)
        attn_w[:, :, -ws:, -ws:] += mask[None, None, :, :]
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(Q.dtype)
        # Mean over window queries; exclude window tokens from candidates
        attn_mean = attn_w[:, :, :, :-ws].mean(dim=-2)  # (B, num_heads, kv_len-W)
        # Collapse GQA groups -> (B, num_kv_heads, kv_len-W)
        attn_mean = attn_mean.view(
            attn_mean.shape[0], -1, self.num_kv_groups, attn_mean.shape[-1]
        ).mean(dim=2)
        return F.avg_pool1d(
            attn_mean, kernel_size=self.kernel_size,
            padding=self.kernel_size // 2, stride=1
        )

    def _sim_scores(self, K):
        """Negated cosine similarity of each key to the mean key, normalized to [0,1]."""
        B, H, N, D = K.shape
        vl = N - self.window_size
        Kn = F.normalize(K, dim=-1)
        Kv = Kn[:, :, :vl, :]
        Km = Kn.sum(dim=2, keepdim=True) / N
        sim = torch.matmul(Kv, Km.transpose(-2, -1)).squeeze(-1)
        ns = -sim
        return (ns - ns.amin(dim=-1, keepdim=True)) / (
            ns.amax(dim=-1, keepdim=True) - ns.amin(dim=-1, keepdim=True) + 1e-8
        )

    def _vnorm_scores(self, V):
        """L2 norm of each value vector, normalized to [0,1]."""
        vl = V.shape[2] - self.window_size
        Vv = V[:, :, :vl, :]
        vn = torch.norm(Vv, p=2, dim=-1)
        return (vn - vn.amin(dim=-1, keepdim=True)) / (
            vn.amax(dim=-1, keepdim=True) - vn.amin(dim=-1, keepdim=True) + 1e-8
        )

    def compress(self, K, Q, V):
        """
        Select top-capacity KV tokens per head and return compressed KV.
        Args:
            K: (B, num_kv_heads, kv_len, D)
            Q: (B, num_heads, q_len, D)
            V: (B, num_kv_heads, kv_len, D)
        Returns:
            (K_compressed, V_compressed): (B, num_kv_heads, capacity+window_size, D)
        """
        B, H_kv, N, D = K.shape
        if N <= self.budget:
            return K, V

        attn = self._attn_scores(Q, K, D)
        m = self.select_method

        if m in ("snapkv", "attn"):
            # Attention-only (SnapKV): score = attn_pooled
            combined = attn

        elif m == "vnorm":
            # Attention + value-norm: adds value magnitude signal
            # High-magnitude values may be important even when attention is low
            vn = self._vnorm_scores(V)
            am = attn.mean(dim=-1, keepdim=True)
            vm = vn.mean(dim=-1, keepdim=True)
            vnorm_scaled = vn * (am / (vm + 1e-8))
            combined = attn + vnorm_scaled

        elif m == "headwisemixkv":
            # Per-head mix: diversity (sim) vs. importance (attn + vnorm)
            # score_h = hs_h * sim_scaled + (1 - hs_h) * (attn + vnorm)
            # hs_h from calibration; defaults to 0.5 (equal mix) if not provided.
            # NOTE: Full headwisemixkv requires pre-computed head scores from
            # MixKV-main/MixKV-main/distribution_qwen.py calibration pass.
            sim = self._sim_scores(K)
            vn = self._vnorm_scores(V)
            am = attn.mean(dim=-1, keepdim=True)
            sm = sim.mean(dim=-1, keepdim=True)
            vm = vn.mean(dim=-1, keepdim=True)
            sim_scaled = sim * (am / (sm + 1e-8))
            vn_scaled = vn * (am / (vm + 1e-8))
            importance = attn + vn_scaled

            heads_k, heads_v = [], []
            for h in range(H_kv):
                hs = self.head_scores[self.layer_idx][h].item() if self.head_scores is not None else 0.5
                score_h = hs * sim_scaled[:, h:h+1, :] + (1 - hs) * importance[:, h:h+1, :]
                _, idx = score_h.sort(dim=-1, descending=True)
                sel = idx[:, :, :self.capacity].unsqueeze(-1).expand(-1, -1, -1, D)
                hk = torch.cat([
                    K[:, h:h+1, :-self.window_size, :].gather(2, sel),
                    K[:, h:h+1, -self.window_size:, :]
                ], dim=2)
                hv = torch.cat([
                    V[:, h:h+1, :-self.window_size, :].gather(2, sel),
                    V[:, h:h+1, -self.window_size:, :]
                ], dim=2)
                heads_k.append(hk)
                heads_v.append(hv)
            return torch.cat(heads_k, dim=1), torch.cat(heads_v, dim=1)

        else:
            combined = attn  # fallback

        # Uniform selection (non-headwise)
        _, indices = combined.sort(dim=-1, descending=True)
        sel_idx = indices[:, :, :self.capacity].unsqueeze(-1).expand(-1, -1, -1, D)
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
                # Prefill: compress, then cache
                if hasattr(self, "_mixkv_compressor"):
                    k_comp, v_comp = self._mixkv_compressor.compress(
                        key_states, query_states, value_states
                    )
                    past_key_value.update(k_comp, v_comp, self.layer_idx, cache_kwargs)
                else:
                    past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_for_attn   = _repeat_kv(key_states, self.num_key_value_groups)
        value_for_attn = _repeat_kv(value_states, self.num_key_value_groups)

        # dtype cast (same as HF SDPA implementation)
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


def load_head_similarity_scores(path):
    """Load pre-computed per-layer, per-head KV similarity scores from JSON.

    JSON format (MixKV): keys like "layer-head-key" or "layer-head-value",
    values are lists of similarity scores across calibration samples.
    Returns (num_layers, num_heads) tensor of averaged key similarity.
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


def apply_mixkv_to_model(model, budget, window_size, select_method, head_scores):
    """Monkeypatch all thinker attention layers with MixKV compression."""
    thinker = model.thinker if hasattr(model, "thinker") else model
    if hasattr(thinker, "model") and hasattr(thinker.model, "layers"):
        layers = thinker.model.layers
    elif hasattr(thinker, "layers"):
        layers = thinker.layers
    else:
        raise RuntimeError("Cannot find decoder layers in model")

    print(f"Applying MixKV to {len(layers)} layers: budget={budget}, window={window_size}, method={select_method}")
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        compressor = MixKVCompressor(
            budget=budget, window_size=window_size, kernel_size=5, layer_idx=i,
            num_kv_heads=attn.num_key_value_heads,
            num_kv_groups=attn.num_key_value_groups,
            select_method=select_method, head_scores=head_scores,
        )
        attn._mixkv_compressor = compressor
        patched = _make_mixkv_forward(attn.forward)
        attn.forward = types.MethodType(patched, attn)

    cap = budget - window_size
    print(f"MixKV applied. Capacity per head: {budget} tokens ({cap} selected + {window_size} window)")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_path, dtype_name, select_method, head_score_path=None):
    dt = resolve_model_dtype(dtype_name)
    print(f"Loading model from {model_path} (dtype={dtype_name}) ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dt, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    head_scores = load_head_similarity_scores(head_score_path)
    if select_method == "headwisemixkv" and head_scores is None:
        print(
            "NOTE: headwisemixkv running WITHOUT calibrated head scores (hs=0.5 per head).\n"
            "      For full headwisemixkv, generate scores with:\n"
            "      MixKV-main/MixKV-main/distribution_qwen.py\n"
            "      then pass --head_score_path <path>"
        )
    apply_mixkv_to_model(model, FIXED_BUDGET, FIXED_WINDOW_SIZE, select_method, head_scores)
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

    # Measure prefill time (TTFT) with CUDA events
    prefill_kw = dict(gen_kw)
    if "thinker_max_new_tokens" in prefill_kw:
        prefill_kw["thinker_max_new_tokens"] = 1
    else:
        prefill_kw["max_new_tokens"] = 1
    with torch.no_grad():
        prefill_ms, _ = cuda_time_ms(lambda: model.generate(**gen_in, **prefill_kw))

    with torch.no_grad():
        e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))

    seq_ids = raw_out.sequences if hasattr(raw_out, "sequences") else raw_out
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, orig_nframes, {"prefill_ms": round(prefill_ms, 2), "e2e_ms": round(e2e_ms, 2)}


# ── Per-method run ─────────────────────────────────────────────────────────────

def run_method(method, args, meta, out_dir):
    """Run inference for one select_method and write results to out_dir."""
    dir_name = METHOD_DIR_NAMES[method]
    val_dir = os.path.join(out_dir, dir_name)
    os.makedirs(val_dir, exist_ok=True)

    results_path = os.path.join(val_dir, "results.jsonl")
    vram_path    = os.path.join(val_dir, "vram_log.jsonl")
    console_path = os.path.join(val_dir, "console.log")

    old_stdout = sys.stdout
    tee = Tee(console_path)
    sys.stdout = tee

    print(f"\n{'='*60}")
    print(f"METHOD ABLATION: select_method={method}")
    print(f"  budget={FIXED_BUDGET}, window_size={FIXED_WINDOW_SIZE}")
    print(f"  capacity={FIXED_BUDGET - FIXED_WINDOW_SIZE} selected + {FIXED_WINDOW_SIZE} window")
    if method == "headwisemixkv":
        print(f"  NOTE: running uncalibrated (hs=0.5). For calibrated scores,")
        print(f"        run MixKV-main/MixKV-main/distribution_qwen.py first.")
    print(f"{'='*60}\n")

    try:
        model, processor = load_model(
            args.model, args.dtype, method,
            head_score_path=args.head_score_path if method == "headwisemixkv" else None,
        )
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
                    "method": f"mixkv-{method}",
                    "budget": FIXED_BUDGET, "window_size": FIXED_WINDOW_SIZE,
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
                    "ablation": "mixkv_method",
                    "select_method": method,
                    "budget": FIXED_BUDGET,
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
    print(f"\nMethod={method}: {correct}/{total} = {acc:.2%} (skipped={skipped_no_video})")

    # Per-dataset breakdown
    datasets = {}
    for r in results:
        d = r["dataset"]
        datasets.setdefault(d, {"correct": 0, "total": 0})
        datasets[d]["total"] += 1
        if r["correct"]:
            datasets[d]["correct"] += 1

    summary = {
        "ablation": "mixkv_method",
        "select_method": method,
        "dir_name": dir_name,
        "budget": FIXED_BUDGET,
        "window_size": FIXED_WINDOW_SIZE,
        "capacity": FIXED_BUDGET - FIXED_WINDOW_SIZE,
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

    # Free model memory before next method
    del model
    torch.cuda.empty_cache()

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MixKV ablation: sweep select_method at fixed budget=256"
    )
    parser.add_argument("--value", default=None, choices=ALL_METHODS,
                        help="Single method to run. Omit to run all.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default="metadata.json")
    parser.add_argument("--videos", default="/data/armaan/purs/videos")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory (default: ablation_outputs/mixkv_method/)")
    parser.add_argument("--head_score_path", default=None,
                        help="Path to calibrated head scores JSON for headwisemixkv")
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
        "ablation_outputs", "mixkv_method"
    )
    os.makedirs(out_dir, exist_ok=True)

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} metadata entries")
    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category
                or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    methods_to_run = [args.value] if args.value else ALL_METHODS
    print(f"\nMixKV method ablation: will run {methods_to_run}")
    print(f"Fixed: budget={FIXED_BUDGET}, window_size={FIXED_WINDOW_SIZE}")
    print(f"Output: {out_dir}\n")

    sweep_summaries = []
    for method in methods_to_run:
        print(f"\n{'#'*60}")
        print(f"# Running method: {method}")
        print(f"{'#'*60}")
        summary = run_method(method, args, meta, out_dir)
        if summary:
            sweep_summaries.append(summary)

    # Write sweep summary
    sweep_path = os.path.join(out_dir, "sweep_summary.json")
    with open(sweep_path, "w") as f:
        json.dump({
            "ablation": "mixkv_method",
            "budget": FIXED_BUDGET,
            "window_size": FIXED_WINDOW_SIZE,
            "methods_run": methods_to_run,
            "timestamp": datetime.now().isoformat(),
            "results": sweep_summaries,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Accuracy':>10} {'Correct/Total':>15}")
    print("-" * 55)
    for s in sweep_summaries:
        label = s.get("dir_name", s["select_method"])
        print(f"  {label:<23} {s['accuracy']:>9.2%} {s['correct']:>6}/{s['total']:<6}")
    print(f"\nSweep summary: {sweep_path}")


if __name__ == "__main__":
    main()
