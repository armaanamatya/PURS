"""
stack_omnizip_mixkv.py
Ablation: OmniZip (input-token compression) + MixKV (KV-cache compression) stacked.

OmniZip compresses the INPUT sequence before layer 0 (shorter prefill).
MixKV compresses the KV cache PER HEAD during the forward pass (smaller decode memory).
The two are orthogonal and can be applied simultaneously.

Conditions:
  omnizip_only        – OmniZip rho_v=0.6, rho_a=0.3 (no MixKV)
  mixkv_only          – MixKV budget=256, snapkv (no OmniZip, vanilla model)
  omnizip_mixkv       – OmniZip rho_v=0.6, rho_a=0.3 + MixKV budget=256, snapkv
  omnizip_mixkv_aggressive – OmniZip rho_v=0.3, rho_a=0.3 + MixKV budget=128, snapkv

Usage:
  python stack_omnizip_mixkv.py --metadata /data/armaan/purs/metadata.json \
      --videos /data/armaan/purs/videos --output_base ablation_outputs/stack_omnizip_mixkv
  # Run single condition:
  python stack_omnizip_mixkv.py ... --condition omnizip_only
"""

import argparse
import json
import math
import os
import glob
import sys
import time
import traceback
import types
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

# ── sys.path setup ────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_PURS_ROOT = os.path.join(_REPO_ROOT, "..")
OMNIZIP_DIR = os.path.join(_PURS_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
sys.path.insert(0, OMNIZIP_DIR)
sys.path.insert(0, QWEN_OMNI_UTILS_SRC)
sys.path.insert(0, _PURS_ROOT)

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration as OmniZipModel
from transformers import Qwen2_5OmniForConditionalGeneration as VanillaModel
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ── Constants ─────────────────────────────────────────────────────────────────
BASELINE_MODEL_PATH = "/data/armaan/models/Qwen2.5-Omni-7B"

DEFAULT_FPS = 2.0
DEFAULT_MAX_PIXELS = 100352
DEFAULT_MAX_FRAMES_VIDEOMME = 768
DEFAULT_MAX_FRAMES_OTHER = 128
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1

CONDITIONS = {
    "omnizip_only": {
        "use_omnizip": True, "use_mixkv": False,
        "rho_video": 0.6, "rho_audio": 0.3, "g": 3, "contextual_ratio": 0.05,
        "budget": None, "window_size": None, "select_method": None,
    },
    "mixkv_only": {
        "use_omnizip": False, "use_mixkv": True,
        "rho_video": None, "rho_audio": None, "g": None, "contextual_ratio": None,
        "budget": 256, "window_size": 32, "select_method": "snapkv",
    },
    "omnizip_mixkv": {
        "use_omnizip": True, "use_mixkv": True,
        "rho_video": 0.6, "rho_audio": 0.3, "g": 3, "contextual_ratio": 0.05,
        "budget": 256, "window_size": 32, "select_method": "snapkv",
    },
    "omnizip_mixkv_aggressive": {
        "use_omnizip": True, "use_mixkv": True,
        "rho_video": 0.3, "rho_audio": 0.3, "g": 3, "contextual_ratio": 0.05,
        "budget": 128, "window_size": 32, "select_method": "snapkv",
    },
}

# ── Answer parsing (local fallback) ───────────────────────────────────────────
try:
    from mcq_answer_parse import parse_answer
except ImportError:
    def parse_answer(text: str, choices: list) -> str:
        import re
        t = text.strip()
        m = re.search(r'\b([A-D])\b', t)
        if m:
            return m.group(1).upper()
        for i, c in enumerate(choices):
            if t.lower().startswith(c.lower()[:20]):
                return chr(65 + i)
        return t[:1].upper() if t else "A"

# ── CUDA timing ────────────────────────────────────────────────────────────────
def cuda_time_ms(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out

# ── Tee logger ────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path, label=""):
        self.terminal = sys.stdout
        self.log = open(path, "a")
        self.log.write(f"\n{'='*60}\nRUN {label}: {datetime.now()}\n{'='*60}\n")
        self.log.flush()

    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
    def isatty(self): return self.terminal.isatty()
    def close(self): self.log.close()


class StderrTee:
    def __init__(self, log_file, terminal):
        self.log = log_file; self.terminal = terminal

    def write(self, m): self.terminal.write(m); self.log.write(m); self.log.flush()
    def flush(self): self.terminal.flush(); self.log.flush()
    def isatty(self): return self.terminal.isatty()

# ── MixKV implementation ───────────────────────────────────────────────────────
def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    B, H, S, D = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


class MixKVCompressor:
    def __init__(self, budget=256, window_size=32, kernel_size=5, layer_idx=0,
                 num_kv_heads=4, num_kv_groups=7, select_method="snapkv", head_scores=None):
        self.budget = budget; self.window_size = window_size; self.kernel_size = kernel_size
        self.layer_idx = layer_idx; self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups; self.select_method = select_method
        self.head_scores = head_scores
        self.capacity = max(budget - window_size, 1)

    def _attn_scores(self, Q, K, head_dim):
        K_exp = _repeat_kv(K, self.num_kv_groups)
        Q_w = Q[:, :, -self.window_size:, :]
        ws = self.window_size
        aw = torch.matmul(Q_w, K_exp.transpose(2, 3)) / math.sqrt(head_dim)
        mask = torch.full((ws, ws), torch.finfo(aw.dtype).min, device=aw.device)
        mc = torch.arange(ws, device=aw.device)
        mask.masked_fill_(mc < (mc + 1).view(ws, 1), 0)
        aw[:, :, -ws:, -ws:] += mask[None, None]
        aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(Q.dtype)
        am = aw[:, :, :, :-ws].mean(dim=-2)
        am = am.view(am.shape[0], -1, self.num_kv_groups, am.shape[-1]).mean(dim=2)
        return F.avg_pool1d(am, kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2, stride=1)

    def _vnorm_scores(self, V):
        vl = V.shape[2] - self.window_size
        vn = torch.norm(V[:, :, :vl, :], p=2, dim=-1)
        return (vn - vn.amin(-1, keepdim=True)) / (
            vn.amax(-1, keepdim=True) - vn.amin(-1, keepdim=True) + 1e-8)

    def compress(self, K, Q, V):
        B, H_kv, N, D = K.shape
        if N <= self.budget:
            return K, V
        attn = self._attn_scores(Q, K, D)
        if self.select_method in ("snapkv", "attn"):
            combined = attn
        elif self.select_method == "vnorm":
            vn = self._vnorm_scores(V)
            am = attn.mean(-1, keepdim=True); vm = vn.mean(-1, keepdim=True)
            combined = attn + vn * (am / (vm + 1e-8))
        else:
            combined = attn
        _, idx = combined.sort(dim=-1, descending=True)
        sel = idx[:, :, :self.capacity].unsqueeze(-1).expand(-1, -1, -1, D)
        Kc = torch.cat([K[:, :, :-self.window_size, :].gather(2, sel),
                        K[:, :, -self.window_size:, :]], dim=2)
        Vc = torch.cat([V[:, :, :-self.window_size, :].gather(2, sel),
                        V[:, :, -self.window_size:, :]], dim=2)
        return Kc, Vc


def apply_mixkv(model, budget=256, window_size=32, select_method="snapkv", head_scores=None):
    thinker = model.thinker if hasattr(model, "thinker") else model
    layers = thinker.model.layers if hasattr(thinker, "model") else thinker.layers
    print(f"Applying MixKV: {len(layers)} layers, budget={budget}, method={select_method}")
    for i, layer in enumerate(layers):
        attn = layer.self_attn
        attn._mixkv_compressor = MixKVCompressor(
            budget=budget, window_size=window_size, kernel_size=5, layer_idx=i,
            num_kv_heads=attn.num_key_value_heads,
            num_kv_groups=attn.num_key_value_groups,
            select_method=select_method, head_scores=head_scores)

        def _make_fwd(attn_module):
            def _fwd(self, hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None, **kw):
                bsz, q_len, _ = hidden_states.size()
                Q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                K = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                V = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                cos, sin = position_embeddings
                from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb
                Q, K = apply_multimodal_rotary_pos_emb(Q, K, cos, sin, self.rope_scaling["mrope_section"])
                if past_key_value is not None:
                    ck = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    if q_len == 1:
                        K, V = past_key_value.update(K, V, self.layer_idx, ck)
                    else:
                        if hasattr(self, "_mixkv_compressor"):
                            Kc, Vc = self._mixkv_compressor.compress(K, Q, V)
                            past_key_value.update(Kc, Vc, self.layer_idx, ck)
                        else:
                            past_key_value.update(K, V, self.layer_idx, ck)
                Ke = _repeat_kv(K, self.num_key_value_groups)
                Ve = _repeat_kv(V, self.num_key_value_groups)
                cm = attention_mask[:, :, :, :Ke.shape[-2]] if attention_mask is not None else None
                out = F.scaled_dot_product_attention(
                    Q.contiguous(), Ke.contiguous(), Ve.contiguous(),
                    attn_mask=cm, dropout_p=0.0, is_causal=cm is None and q_len > 1)
                return self.o_proj(out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)), None, past_key_value
            return _fwd
        attn.forward = types.MethodType(_make_fwd(attn), attn)

# ── Prompt builders ───────────────────────────────────────────────────────────
SYSTEM_PROMPT_DEFAULT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
SYSTEM_MCQ_SUFFIX = (
    "For multiple-choice questions, reply with only one letter: A, B, C, or D. "
    "Do not explain, do not ask follow-up questions, and do not add text after the letter."
)

def _format_choice_lines(choices):
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))

def _canon(dataset):
    return (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")

def build_user_prompt_for_dataset(dataset, question, choices):
    d = _canon(dataset)
    opts = _format_choice_lines(choices)
    post = "The best answer is:"
    if d in {"video-mme", "videomme"}:
        head = ("Select the best answer to the following multiple-choice question based on the video and the subtitles. "
                "Respond with only the letter (A, B, C, or D) of the correct option.")
        return f"{head}\n{question}\n{opts}\n{post}"
    if d == "worldsense":
        sys_ws = ("Carefully watch this video and pay attention to every detail. "
                  "Based on your observations, select the best option that accurately addresses the question.")
        frames = ("These are the frames of a video and the corresponding audio. "
                  "Select the best answer to the following multiple-choice question based on the video. "
                  "Respond with only the letter (A, B, C, or D) of the correct option.")
        return f"{sys_ws}\n{frames}\n{question}\n{opts}"
    if d in {"daily-omni", "dailyomni"}:
        head = ("Listen and watch the video carefully. "
                "Select the best answer to the following multiple-choice question. "
                "Respond with only the letter (A, B, C, or D) of the correct option.")
        return f"{head}\n{question}\n{opts}\n{post}"
    return ("Select the best answer to the following multiple-choice question based on the video. "
            f"Respond with only the letter (A, B, C, or D) of the correct option.\n{question}\n{opts}\n{post}")

# ── Audio check ───────────────────────────────────────────────────────────────
def check_video_has_audio(video_path):
    try:
        import av
        c = av.open(video_path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False

# ── Video path resolution ─────────────────────────────────────────────────────
def resolve_video_path(file_field, videos_dir):
    if os.path.exists(file_field):
        return file_field
    normalized = file_field.replace("\\", "/")
    if os.path.exists(normalized):
        return normalized
    rel = normalized
    for prefix in ("videos/", "videos\\"):
        if normalized.startswith(prefix):
            rel = normalized[len(prefix):]
            break
    candidate = os.path.normpath(os.path.join(videos_dir, rel))
    if os.path.exists(candidate):
        return candidate
    rel_norm = rel.replace("\\", "/")
    filename = rel_norm.split("/")[-1]
    suffix_matches = [m for m in glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
                      if m.replace("\\", "/").endswith(rel_norm)]
    if suffix_matches:
        return suffix_matches[0]
    basename_matches = glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None

# ── VRAM helpers ──────────────────────────────────────────────────────────────
def capture_vram():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3

# ── Model loaders ─────────────────────────────────────────────────────────────
def load_omnizip_model(rho_video, rho_audio, g, contextual_ratio, dtype=torch.bfloat16):
    print(f"Loading OmniZip model from {BASELINE_MODEL_PATH} ...")
    print(f"  rho_video={rho_video}  rho_audio={rho_audio}  g={g}  contextual_ratio={contextual_ratio}")
    model = OmniZipModel.from_pretrained(
        BASELINE_MODEL_PATH, torch_dtype=dtype, device_map="auto",
        attn_implementation="flash_attention_2")
    model.thinker.omnizip_config = {
        "rho_audio": rho_audio, "rho_video": rho_video,
        "g": g, "contextual_ratio": contextual_ratio,
    }
    processor = Qwen2_5OmniProcessor.from_pretrained(BASELINE_MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model, processor


def load_vanilla_model(dtype=torch.bfloat16):
    print(f"Loading vanilla model from {BASELINE_MODEL_PATH} ...")
    model = VanillaModel.from_pretrained(
        BASELINE_MODEL_PATH, torch_dtype=dtype, device_map="auto",
        attn_implementation="flash_attention_2")
    processor = Qwen2_5OmniProcessor.from_pretrained(BASELINE_MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model, processor

# ── Inference ─────────────────────────────────────────────────────────────────
_DROP_KEYS = frozenset({"images", "return_tensors", "text"})

def _prepare_inputs(model, inputs):
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)
    return inputs

def run_inference(model, processor, video_path, dataset, question, choices,
                  use_audio, run_cfg):
    prompt = build_user_prompt_for_dataset(dataset, question, choices)
    ds = _canon(dataset)
    max_frames = (run_cfg["max_frames_videomme"] if ds in {"video-mme", "videomme"}
                  else run_cfg["max_frames_other"])

    sys_text = SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_text}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path,
             "fps": run_cfg["fps"], "max_pixels": run_cfg["max_pixels"],
             "max_frames": max_frames},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio)

    if not videos or videos[0] is None or videos[0].shape[0] <= 0:
        raise ValueError("Decoded 0 video frames")

    num_frames = int(videos[0].shape[0])
    if hasattr(model, "thinker"):
        model.thinker.nframes = num_frames

    inputs = processor(text=text, audio=audios, images=images, videos=videos,
                       return_tensors="pt", padding=True, use_audio_in_video=use_audio)
    inputs = _prepare_inputs(model, inputs)

    gen_in = {k: v for k, v in inputs.items() if k not in _DROP_KEYS}
    tok = processor.tokenizer
    do_sample = run_cfg["temperature"] > 0
    gen_kw = dict(use_audio_in_video=use_audio, return_audio=False,
                  thinker_max_new_tokens=run_cfg["max_new_tokens"],
                  thinker_do_sample=do_sample,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if do_sample:
        gen_kw["thinker_temperature"] = run_cfg["temperature"]

    torch.cuda.reset_peak_memory_stats()
    e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))

    seq_ids = raw_out.sequences if hasattr(raw_out, "sequences") else raw_out
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, num_frames, round(e2e_ms, 2)

# ── Run one condition ─────────────────────────────────────────────────────────
def run_condition(cond_name, cond_cfg, meta, videos_dir, output_base, run_cfg, no_audio):
    out_dir = os.path.join(output_base, cond_name)
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, "results.jsonl")
    vram_path    = os.path.join(out_dir, "vram_log.jsonl")
    console_path = os.path.join(out_dir, "console.log")
    stderr_path  = os.path.join(out_dir, "stderr.log")
    summary_path = os.path.join(out_dir, "run_summary.json")

    _old_stdout = sys.stdout
    _old_stderr = sys.stderr
    tee = Tee(console_path, label=cond_name)
    sys.stdout = tee
    _sf = open(stderr_path, "a")
    sys.stderr = StderrTee(_sf, sys.__stderr__)

    print(f"\nCondition: {cond_name}")
    print(f"Config: {cond_cfg}")

    # Load model
    if cond_cfg["use_omnizip"]:
        model, processor = load_omnizip_model(
            cond_cfg["rho_video"], cond_cfg["rho_audio"],
            cond_cfg["g"], cond_cfg["contextual_ratio"])
    else:
        model, processor = load_vanilla_model()

    model_loaded_alloc, model_loaded_reserved = capture_vram()
    print(f"Model loaded. VRAM: {model_loaded_alloc:.2f} GB alloc, {model_loaded_reserved:.2f} GB reserved")

    if cond_cfg["use_mixkv"]:
        apply_mixkv(model, budget=cond_cfg["budget"],
                    window_size=cond_cfg["window_size"],
                    select_method=cond_cfg["select_method"])

    runnable = [e for e in meta if e.get("questions")]
    correct = total = skipped = 0
    t_start = time.time()

    with open(results_path, "w") as rf, open(vram_path, "w") as vf:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], videos_dir)
            label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {label}: video not found")
                skipped += 1
                continue

            use_audio = (not no_audio) and check_video_has_audio(video_path)

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset   = entry.get("dataset", "")

                before_alloc, before_reserved = capture_vram()
                try:
                    pred, decoded, n_frames, e2e_ms = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        use_audio, run_cfg)
                    status = "ok"
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"  ERROR {label}: {e!r}")
                    pred, decoded, n_frames, e2e_ms = "ERROR", str(e), 0, 0.0
                    status = "error"
                    tb_str = tb

                peak_alloc  = torch.cuda.max_memory_allocated() / 1024**3
                peak_res    = torch.cuda.max_memory_reserved() / 1024**3
                after_alloc, after_res = capture_vram()

                vram_entry = {
                    "condition": cond_name, "entry": label, "task_type": task_type,
                    "status": status, "n_frames": n_frames, "e2e_ms": e2e_ms,
                    "model_loaded_alloc_gb": round(model_loaded_alloc, 2),
                    "model_loaded_reserved_gb": round(model_loaded_reserved, 2),
                    "before_alloc_gb": round(before_alloc, 2),
                    "before_reserved_gb": round(before_reserved, 2),
                    "peak_alloc_gb": round(peak_alloc, 2),
                    "peak_reserved_gb": round(peak_res, 2),
                    "after_alloc_gb": round(after_alloc, 2),
                    "after_reserved_gb": round(after_res, 2),
                }
                vf.write(json.dumps(vram_entry) + "\n"); vf.flush()

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "condition": cond_name, "dataset": dataset, "task_type": task_type,
                    "question": question, "choices": choices, "answer": answer,
                    "prediction": pred, "correct": is_correct, "reasoning": decoded,
                    "n_frames": n_frames, "e2e_ms": e2e_ms,
                    "use_omnizip": cond_cfg["use_omnizip"],
                    "use_mixkv": cond_cfg["use_mixkv"],
                }
                rf.write(json.dumps(result) + "\n"); rf.flush()

                sym = "+" if is_correct else "-"
                print(f"  [{sym}] {label} pred={pred} ans={answer} e2e={e2e_ms:.0f}ms")

    elapsed = time.time() - t_start
    acc = correct / total if total else 0.0
    summary = {
        "condition": cond_name, "correct": correct, "total": total,
        "accuracy": round(acc, 4), "skipped": skipped,
        "elapsed_s": round(elapsed, 1),
        "config": cond_cfg,
    }
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print(f"\n[{cond_name}] Accuracy: {correct}/{total} = {acc:.2%}  ({elapsed:.0f}s)")

    sys.stdout = _old_stdout
    sys.stderr = _old_stderr
    tee.close(); _sf.close()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Ablation: OmniZip + MixKV stacking")
    parser.add_argument("--metadata",    default="/data/armaan/purs/metadata.json")
    parser.add_argument("--videos",      default="/data/armaan/purs/videos")
    parser.add_argument("--output_base", default="/data/armaan/purs/ablation_outputs/stack_omnizip_mixkv")
    parser.add_argument("--condition",   default=None, choices=list(CONDITIONS.keys()),
                        help="Run a single condition (default: all)")
    parser.add_argument("--fps",              type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels",       type=int,   default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max_frames_videomme", type=int, default=DEFAULT_MAX_FRAMES_VIDEOMME)
    parser.add_argument("--max_frames_other",    type=int, default=DEFAULT_MAX_FRAMES_OTHER)
    parser.add_argument("--max_new_tokens",   type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature",      type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no_audio",         action="store_true")
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_cfg = {
        "fps": args.fps, "max_pixels": args.max_pixels,
        "max_frames_videomme": args.max_frames_videomme,
        "max_frames_other": args.max_frames_other,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    os.makedirs(args.output_base, exist_ok=True)
    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} metadata entries")

    to_run = {args.condition: CONDITIONS[args.condition]} if args.condition else CONDITIONS
    all_summaries = {}

    for cond_name, cond_cfg in to_run.items():
        summary = run_condition(cond_name, cond_cfg, meta, args.videos,
                                args.output_base, run_cfg, args.no_audio)
        all_summaries[cond_name] = summary

    cmp_path = os.path.join(args.output_base, "comparison_summary.json")
    with open(cmp_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nComparison summary written to {cmp_path}")
    for name, s in all_summaries.items():
        print(f"  {name:<35} acc={s['accuracy']:.4f}  ({s['correct']}/{s['total']})")

if __name__ == "__main__":
    main()
