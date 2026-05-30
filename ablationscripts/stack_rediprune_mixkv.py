"""
stack_rediprune_mixkv.py
Ablation: ReDiPrune (frame-level selection) + MixKV (KV-cache compression) stacked.

ReDiPrune selects K representative frames from the decoded video before passing to
the model (reduces sequence length). MixKV then compresses the KV cache of the
shorter sequence during the forward pass.

Combined effect: ~keep_ratio fewer frames × budget KV tokens per head.

Conditions:
  rediprune_only  – keep_ratio=0.5, no MixKV
  mixkv_only      – budget=256 snapkv, no ReDiPrune (all frames)
  stacked         – keep_ratio=0.5 + budget=256 snapkv

Usage:
  python stack_rediprune_mixkv.py --metadata /data/armaan/purs/metadata.json \
      --videos /data/armaan/purs/videos --output_base ablation_outputs/stack_rediprune_mixkv
  python stack_rediprune_mixkv.py ... --condition rediprune_only
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

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
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
    "rediprune_only": {
        "use_rediprune": True, "keep_ratio": 0.5, "alpha": 0.5, "tau": 0.1,
        "use_mixkv": False, "budget": None, "window_size": None, "select_method": None,
    },
    "mixkv_only": {
        "use_rediprune": False, "keep_ratio": None, "alpha": None, "tau": None,
        "use_mixkv": True, "budget": 256, "window_size": 32, "select_method": "snapkv",
    },
    "stacked": {
        "use_rediprune": True, "keep_ratio": 0.5, "alpha": 0.5, "tau": 0.1,
        "use_mixkv": True, "budget": 256, "window_size": 32, "select_method": "snapkv",
    },
}

# ── Answer parsing ─────────────────────────────────────────────────────────────
try:
    from mcq_answer_parse import parse_answer
except ImportError:
    def parse_answer(text, choices):
        import re
        t = text.strip()
        m = re.search(r'\b([A-D])\b', t)
        if m:
            return m.group(1).upper()
        return t[:1].upper() if t else "A"

# ── CUDA timing ───────────────────────────────────────────────────────────────
def cuda_time_ms(fn):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(); out = fn(); e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out

# ── Tee loggers ───────────────────────────────────────────────────────────────
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

# ── ReDiPrune implementation ──────────────────────────────────────────────────
@torch.no_grad()
def rediprune_select(visual_tokens, text_query, keep_ratio, alpha=0.5, tau=0.1):
    P, Cv = visual_tokens.shape
    K = max(1, int(round(P * keep_ratio)))
    if K >= P:
        return torch.arange(P, device=visual_tokens.device)
    dev = visual_tokens.device
    Vn = F.normalize(visual_tokens.float(), p=2, dim=1)
    tq = text_query.float().to(dev)
    tq = tq.unsqueeze(0) if tq.dim() == 1 else tq
    Dq = tq.shape[1]
    if Dq != Cv:
        if Dq > Cv:
            tq = (tq.view(1, Cv, -1).mean(2) if Dq % Cv == 0
                  else F.adaptive_avg_pool1d(tq.unsqueeze(1), Cv).squeeze(1))
        else:
            tq = tq.repeat(1, (Cv + Dq - 1) // Dq)[:, :Cv]
    Tn = F.normalize(tq, p=2, dim=1)
    rel = (Vn @ Tn.t()).squeeze(1)
    cand = (torch.arange(P, device=dev) if tau == 0
            else (torch.where(rel >= tau)[0] if (rel >= tau).sum() >= K
                  else torch.topk(rel, K)[1]))
    div = 1.0 - (Vn[cand] @ Vn[cand].t())
    rc = rel[cand]
    Nc = cand.shape[0]
    sel = torch.empty(K, dtype=torch.long, device=dev)
    sel[0] = rc.argmax()
    for i in range(1, K):
        md = div[sel[:i]].min(0).values
        sc = md + alpha * rc
        m = torch.ones(Nc, dtype=torch.bool, device=dev)
        m[sel[:i]] = False
        sc[~m] = -float('inf')
        sel[i] = sc.argmax()
    return torch.sort(cand[sel])[0]


def get_text_query(model, processor, question):
    tids = processor.tokenizer(question, return_tensors="pt",
                               truncation=True, max_length=128)["input_ids"]
    dev = next(model.parameters()).device
    emb_layer = (model.thinker.model.embed_tokens if hasattr(model, "thinker")
                 else model.model.embed_tokens)
    with torch.no_grad():
        emb = emb_layer(tids.to(dev))
    return emb.mean(dim=1).squeeze(0)

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

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_video_has_audio(video_path):
    try:
        import av
        c = av.open(video_path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False


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


def capture_vram():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3

# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(dtype=torch.bfloat16):
    print(f"Loading model from {BASELINE_MODEL_PATH} ...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        BASELINE_MODEL_PATH, torch_dtype=dtype, device_map="auto",
        attn_implementation="flash_attention_2")
    processor = Qwen2_5OmniProcessor.from_pretrained(BASELINE_MODEL_PATH)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    return model, processor

# ── Input prep ────────────────────────────────────────────────────────────────
_DROP_KEYS = frozenset({"images", "return_tensors", "text"})

def _prepare_inputs(model, inputs):
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)
    return inputs

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, processor, video_path, dataset, question, choices,
                  use_audio, run_cfg, cond_cfg):
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

    orig_frames = int(videos[0].shape[0])
    used_frames = orig_frames

    # ReDiPrune: frame selection on raw video tensor before processor
    if cond_cfg["use_rediprune"]:
        vid_tensor = videos[0]  # shape: (T, C, H, W) typically
        # global avg pool per frame for features
        if vid_tensor.dim() == 4:
            features = vid_tensor.float().mean(dim=(-2, -1))  # (T, C)
        else:
            features = vid_tensor.float().reshape(vid_tensor.shape[0], -1)
        dev = next(model.parameters()).device
        features = features.to(dev)
        text_query = get_text_query(model, processor, question)
        idx = rediprune_select(
            features, text_query,
            keep_ratio=cond_cfg["keep_ratio"],
            alpha=cond_cfg["alpha"],
            tau=cond_cfg["tau"],
        )
        idx_cpu = idx.cpu()
        videos[0] = vid_tensor[idx_cpu]
        used_frames = int(idx_cpu.shape[0])

    if hasattr(model, "thinker"):
        model.thinker.nframes = used_frames

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
    return letter, decoded, orig_frames, used_frames, round(e2e_ms, 2)

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

    model, processor = load_model()
    model_loaded_alloc, model_loaded_reserved = capture_vram()
    print(f"Model loaded. VRAM: {model_loaded_alloc:.2f} GB alloc, "
          f"{model_loaded_reserved:.2f} GB reserved")

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
                    pred, decoded, orig_nf, used_nf, e2e_ms = run_inference(
                        model, processor, video_path, dataset, question, choices,
                        use_audio, run_cfg, cond_cfg)
                    status = "ok"
                except Exception as exc:
                    print(f"  ERROR {label}: {exc!r}")
                    traceback.print_exc()
                    pred, decoded, orig_nf, used_nf, e2e_ms = "ERROR", str(exc), 0, 0, 0.0
                    status = "error"

                peak_alloc  = torch.cuda.max_memory_allocated() / 1024**3
                peak_res    = torch.cuda.max_memory_reserved() / 1024**3
                after_alloc, after_res = capture_vram()

                vf.write(json.dumps({
                    "condition": cond_name, "entry": label, "task_type": task_type,
                    "status": status, "orig_frames": orig_nf, "used_frames": used_nf,
                    "e2e_ms": e2e_ms,
                    "model_loaded_alloc_gb": round(model_loaded_alloc, 2),
                    "model_loaded_reserved_gb": round(model_loaded_reserved, 2),
                    "before_alloc_gb": round(before_alloc, 2),
                    "before_reserved_gb": round(before_reserved, 2),
                    "peak_alloc_gb": round(peak_alloc, 2),
                    "peak_reserved_gb": round(peak_res, 2),
                    "after_alloc_gb": round(after_alloc, 2),
                    "after_reserved_gb": round(after_res, 2),
                }) + "\n"); vf.flush()

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                rf.write(json.dumps({
                    "condition": cond_name, "dataset": dataset, "task_type": task_type,
                    "question": question, "choices": choices, "answer": answer,
                    "prediction": pred, "correct": is_correct, "reasoning": decoded,
                    "orig_frames": orig_nf, "used_frames": used_nf, "e2e_ms": e2e_ms,
                    "use_rediprune": cond_cfg["use_rediprune"],
                    "use_mixkv": cond_cfg["use_mixkv"],
                }) + "\n"); rf.flush()

                sym = "+" if is_correct else "-"
                print(f"  [{sym}] {label} pred={pred} ans={answer} "
                      f"frames={used_nf}/{orig_nf} e2e={e2e_ms:.0f}ms")

    elapsed = time.time() - t_start
    acc = correct / total if total else 0.0
    summary = {
        "condition": cond_name, "correct": correct, "total": total,
        "accuracy": round(acc, 4), "skipped": skipped,
        "elapsed_s": round(elapsed, 1), "config": cond_cfg,
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
    parser = argparse.ArgumentParser(description="Ablation: ReDiPrune + MixKV stacking")
    parser.add_argument("--metadata",    default="/data/armaan/purs/metadata.json")
    parser.add_argument("--videos",      default="/data/armaan/purs/videos")
    parser.add_argument("--output_base", default="/data/armaan/purs/ablation_outputs/stack_rediprune_mixkv")
    parser.add_argument("--condition",   default=None, choices=list(CONDITIONS.keys()),
                        help="Run a single condition (default: all)")
    parser.add_argument("--fps",                 type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels",          type=int,   default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max_frames_videomme", type=int,   default=DEFAULT_MAX_FRAMES_VIDEOMME)
    parser.add_argument("--max_frames_other",    type=int,   default=DEFAULT_MAX_FRAMES_OTHER)
    parser.add_argument("--max_new_tokens",      type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature",         type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--no_audio",            action="store_true")
    parser.add_argument("--seed",                type=int,   default=42)
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
    print(f"\nComparison summary: {cmp_path}")
    for name, s in all_summaries.items():
        print(f"  {name:<25} acc={s['accuracy']:.4f}  ({s['correct']}/{s['total']})")

if __name__ == "__main__":
    main()
