"""
omnizip_audio_off.py
====================
ABLATION: Disable audio-guided video selection; use uniform video compression.

Standard OmniZip computes per-group video merging ratios based on audio retention:
    video_merging_ratio[i] = max_ratio + (min_ratio - max_ratio) * audio_group_retention[i]
    where min_ratio=0.35, max_ratio=0.75

This ablation bypasses that cross-modal signal and instead uses:
    video_merging_ratio[i] = rho_video  (uniform, same for all groups)

This isolates whether the audio→video guidance is actually helpful, or whether
a uniform video compression at the same global rate is sufficient.

TWO CONDITIONS:
  1. "uniform"  — uniform video merging ratio; audio still used as INPUT to model
                  (audio tokens still processed, but their retention pattern does
                  NOT modulate which video tokens are kept)
  2. "noaudio"  — same uniform compression + --no_audio (audio not input either)
                  (completely removes audio modality)

Implementation: monkey-patch omnizip.omnizip_units.omnizip after model load,
replacing the audio_group_retention → mapped_vs → adjusted_vs pipeline with a
flat [rho_video] * n_groups vector. Audio compression (omnizip_audio_attn) still
runs normally to keep audio tokens themselves compressed, but the output mask is
NOT used to gate video selection.

Fixed params: rho_video=0.6, rho_audio=0.3, g=3, contextual_ratio=0.05

Usage:
    # Run uniform condition (audio as input, uniform video compression)
    python omnizip_audio_off.py --condition uniform \\
        --metadata /data/armaan/purs/metadata.json \\
        --videos   /data/armaan/purs/videos

    # Run no-audio condition
    python omnizip_audio_off.py --condition noaudio ...

    # Run both sequentially
    python omnizip_audio_off.py --all_conditions ...
"""

import argparse
import gc
import glob
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.dirname(_SCRIPT_DIR)
OMNIZIP_DIR         = os.path.join(_REPO_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
for _p in (OMNIZIP_DIR, QWEN_OMNI_UTILS_SRC, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from mcq_answer_parse import parse_answer

# ── Constants ──────────────────────────────────────────────────────────────────
ABLATION_NAME = "omnizip_audio_off"

CONDITIONS = ["uniform", "noaudio"]

DEFAULT_MODEL            = "/data/armaan/models/Qwen2.5-Omni-7B"
DEFAULT_FPS              = 2.0
DEFAULT_MAX_PIXELS       = 100352
DEFAULT_MAX_FRAMES_VMME  = 768
DEFAULT_MAX_FRAMES_OTHER = 128
DEFAULT_MAX_NEW_TOKENS   = 256
DEFAULT_TEMPERATURE      = 0.1

FIXED_RHO_VIDEO        = 0.6
FIXED_RHO_AUDIO        = 0.3
FIXED_G                = 3
FIXED_CONTEXTUAL_RATIO = 0.05

# ── Prompt builders ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
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


def _fmt_choices(choices):
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))


def build_prompt(dataset, question, choices):
    ds = (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")
    opts = _fmt_choices(choices)
    if ds in {"video-mme", "videomme"}:
        return _VIDEO_MME_OPTION_PROMPT + "\n" + question + "\n" + opts + "\n" + _VIDEO_MME_POST_PROMPT
    if ds == "worldsense":
        return _WORLD_SENSE_SYS + _WORLD_SENSE_FRAMES_AUDIO + question + "\n" + "\n".join(choices) + "\n"
    if ds in {"daily-omni", "dailyomni"}:
        return (
            "Listen and watch the video carefully. Select the best answer to the following multiple-choice question. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + question + "\n" + opts + "\n" + _VIDEO_MME_POST_PROMPT
        )
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        + question + "\n" + opts + "\n" + _VIDEO_MME_POST_PROMPT
    )


# ── Video utilities ────────────────────────────────────────────────────────────
def resolve_video_path(file_field, videos_dir):
    if os.path.exists(file_field):
        return file_field
    normalized = file_field.replace("\\", "/")
    filename = normalized.split("/")[-1]
    stem = filename.rsplit(".", 1)[0]
    cand = os.path.join(videos_dir, filename)
    if os.path.exists(cand):
        return cand
    for ext in ("mp4", "mkv", "webm", "avi"):
        m = glob.glob(os.path.join(videos_dir, "**", f"{stem}.{ext}"), recursive=True)
        if m:
            return m[0]
    return None


def check_video_has_audio(path):
    try:
        import av
        c = av.open(path)
        has = len(c.streams.audio) > 0
        c.close()
        return has
    except Exception:
        return False


# ── CUDA timing ───────────────────────────────────────────────────────────────
def cuda_time_ms(fn):
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), out


# ── Tee logger ────────────────────────────────────────────────────────────────
class Tee:
    def __init__(self, path, label=""):
        self.terminal = sys.stdout
        self.log = open(path, "a", encoding="utf-8")
        self.log.write(f"\n{'='*60}\n{label} {datetime.now()}\n{'='*60}\n")

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
    def __init__(self, path):
        self.terminal = sys.__stderr__
        self.log = open(path, "a", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


# ── Monkey-patch: uniform video compression (no audio guidance) ────────────────
def _install_uniform_video_patch(rho_video: float):
    """
    Replace omnizip_units.omnizip with a version that uses uniform per-group
    video merging ratios ([rho_video] * n_groups) instead of audio-guided ones.

    Audio tokens are STILL compressed via omnizip_audio_attn (audio IS processed),
    but the resulting audio_mask is NOT used to modulate video token selection.
    Instead, all groups get the same merging ratio = rho_video.

    This is a direct reimplementation of the omnizip() body from omnizip_units.py
    with the single change: base_vs / adjusted_vs are replaced by a flat list.
    """
    import omnizip.omnizip_units as _ou
    from omnizip.omnizip_units import omnizip_audio_attn, omnizip_istm

    def _omnizip_uniform_video(
        input_embeds: torch.Tensor,
        attn_logits: torch.Tensor,
        input_ids: torch.Tensor,
        audio_token_id: int,
        video_token_id: int,
        num_input_frames: int,
        merging_ratio_audio: float = 0.5,
        merging_ratio_v: float = 0.5,
        contextual_ratio: float = 0.05,
        g: int = 3,
    ):
        """
        Uniform-video variant of omnizip().

        All per-group video merging ratios are set to `merging_ratio_v` (= rho_video)
        unconditionally. Audio tokens still go through omnizip_audio_attn for their
        own compression (dominant + contextual selection + merge), but the resulting
        audio_mask is NOT used to derive video_merging_ratios.
        """
        device = input_embeds.device
        is_batched = input_embeds.dim() == 3
        if is_batched:
            B, L, D = input_embeds.shape
            flat_embeds = input_embeds.reshape(-1, D)
            flat_ids = input_ids.reshape(-1)
        else:
            L, D = input_embeds.shape
            flat_embeds = input_embeds
            flat_ids = input_ids

        video_token_mask = (flat_ids == video_token_id).to(device)
        audio_token_mask = (flat_ids == audio_token_id).to(device)

        video_indices = torch.nonzero(video_token_mask, as_tuple=True)[0]
        audio_indices = torch.nonzero(audio_token_mask, as_tuple=True)[0]

        video_feature = flat_embeds[video_indices]
        audio_feature = flat_embeds[audio_indices]

        video_token_per_frame = video_feature.shape[0] // max(1, num_input_frames)

        # ── Audio compression (still runs, produces audio_mask) ──────────────
        audio_mask, merge_plan = omnizip_audio_attn(
            audio_feature=audio_feature,
            video_feature=video_feature,
            attn_logits=attn_logits,
            merging_ratio=merging_ratio_audio,
            contextual_ratio=contextual_ratio,
            g=g,
        )

        # Apply audio merge plan to flat_embeds (same as original)
        if g > 0 and len(merge_plan) > 0:
            a_norm = audio_feature / (audio_feature.norm(dim=-1, keepdim=True) + 1e-6)
            v_norm = (
                video_feature / (video_feature.norm(dim=-1, keepdim=True) + 1e-6)
                if video_feature.numel() > 0 else None
            )
            for anchor_rel_idx, merge_rel_list in merge_plan.items():
                if not merge_rel_list:
                    continue
                merge_rel = torch.tensor(merge_rel_list, device=device, dtype=torch.long)
                if v_norm is not None:
                    scores = (a_norm[merge_rel] @ v_norm.T).max(dim=1).values
                else:
                    scores = (a_norm[merge_rel] @ a_norm[anchor_rel_idx].unsqueeze(-1)).squeeze(-1)
                w = torch.softmax(scores, dim=0)
                anchor_vec = audio_feature[anchor_rel_idx]
                merged_vec = (audio_feature[merge_rel] * w.unsqueeze(-1)).sum(dim=0)
                new_anchor = (anchor_vec + merged_vec) / (1.0 + w.sum())
                anchor_global_idx = audio_indices[anchor_rel_idx]
                flat_embeds[anchor_global_idx] = new_anchor

        # ── UNIFORM video merging (KEY CHANGE: ignore audio_mask for video) ──
        # Determine grouping mode (same branching logic as original omnizip())
        if num_input_frames % 4 == 0:
            group_count = num_input_frames // 4
            num_video_tokens_per_group = max(1, video_feature.shape[0] // group_count)

            # ABLATION: uniform ratios — ignore audio_group_retention entirely
            video_merging_ratios = [merging_ratio_v] * group_count

            video_group_masks = []
            for i in range(0, group_count, 2):
                if i + 2 <= group_count:
                    video_merging_ratio = video_merging_ratios[i:i+2]
                    v_start = i * num_video_tokens_per_group
                    v_end = (i + 2) * num_video_tokens_per_group if i < group_count - 1 else video_feature.shape[0]
                    group_feat = video_feature[v_start:v_end]
                    group_len = group_feat.size(0)
                    if group_len % 4 == 0:
                        group_mask = omnizip_istm(
                            group_feat,
                            num_tokens_per_frame=video_token_per_frame * 2,
                            merging_ratio=video_merging_ratio,
                        )
                    else:
                        group_mask = torch.ones(group_len, dtype=torch.bool, device=group_feat.device)
                    video_group_masks.append(group_mask)
                else:
                    video_merging_ratio = video_merging_ratios[i:i+2]
                    v_start = i * num_video_tokens_per_group
                    v_end = video_feature.shape[0]
                    group_feat = video_feature[v_start:v_end]
                    group_len = group_feat.size(0)
                    group_mask = torch.ones(group_len, dtype=torch.bool, device=group_feat.device)
                    video_group_masks.append(group_mask)

        else:
            # Non-mod-4 path: same group structure as original
            VIDEO_GROUP_SIZE = video_token_per_frame * 4
            AUDIO_GROUP_SIZE = 50

            num_video_tokens = video_feature.shape[0]
            num_audio_tokens = audio_feature.shape[0]

            video_groups = []
            audio_groups = []
            v_ptr = a_ptr = 0
            while v_ptr + VIDEO_GROUP_SIZE <= num_video_tokens and a_ptr + AUDIO_GROUP_SIZE <= num_audio_tokens:
                video_groups.append((v_ptr, v_ptr + VIDEO_GROUP_SIZE))
                audio_groups.append((a_ptr, a_ptr + AUDIO_GROUP_SIZE))
                v_ptr += VIDEO_GROUP_SIZE
                a_ptr += AUDIO_GROUP_SIZE
            if v_ptr < num_video_tokens:
                if a_ptr < num_audio_tokens:
                    video_groups.append((v_ptr, num_video_tokens))
                    audio_groups.append((a_ptr, num_audio_tokens))
                else:
                    video_groups.append((v_ptr, num_video_tokens))
                    audio_groups.append((a_ptr, a_ptr))
            elif a_ptr < num_audio_tokens:
                video_groups.append((v_ptr, v_ptr))
                audio_groups.append((a_ptr, num_audio_tokens))

            assert len(video_groups) == len(audio_groups)
            group_count = len(video_groups)

            # ABLATION: uniform ratios
            video_merging_ratios = [merging_ratio_v] * group_count

            video_group_masks = []
            idx = 0
            while idx < len(video_groups):
                if idx + 1 < len(video_groups):
                    v_start_0, v_end_0 = video_groups[idx]
                    v_start_1, v_end_1 = video_groups[idx + 1]
                    group_feat = video_feature[v_start_0:v_end_1]
                    video_merging_ratio = [video_merging_ratios[idx], video_merging_ratios[idx + 1]]
                else:
                    v_start_0, v_end_0 = video_groups[idx]
                    group_feat = video_feature[v_start_0:v_end_0]
                    video_merging_ratio = [video_merging_ratios[idx]]

                group_len = group_feat.size(0)
                is_tail = (group_len != 2 * VIDEO_GROUP_SIZE) if (idx + 1 < len(video_groups)) else (group_len != VIDEO_GROUP_SIZE)
                if group_len == 0:
                    group_mask = torch.zeros(0, dtype=torch.bool, device=video_feature.device)
                elif is_tail:
                    group_mask = torch.ones(group_len, dtype=torch.bool, device=video_feature.device)
                else:
                    group_mask = omnizip_istm(
                        group_feat,
                        num_tokens_per_frame=video_token_per_frame * 2,
                        merging_ratio=video_merging_ratio,
                    )
                video_group_masks.append(group_mask)
                idx += 2

        video_mask = torch.cat(video_group_masks, dim=0)
        global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)

        assert video_mask.shape[0] == video_indices.shape[0], (
            f"video_mask {video_mask.shape[0]} != video_indices {video_indices.shape[0]}"
        )
        assert audio_mask.shape[0] == audio_indices.shape[0], (
            f"audio_mask {audio_mask.shape[0]} != audio_indices {audio_indices.shape[0]}"
        )

        global_mask[video_indices] = video_mask
        global_mask[audio_indices] = audio_mask

        if is_batched:
            input_embeds_out = flat_embeds.reshape(B, L, D)
        else:
            input_embeds_out = flat_embeds

        return input_embeds_out, global_mask

    # Install the patch
    _ou.omnizip = _omnizip_uniform_video
    print(f"[{ABLATION_NAME}] Monkey-patched omnizip_units.omnizip → uniform video compression "
          f"(rho_video={rho_video} for all groups)")


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(model_path):
    """Load model with standard OmniZip config; patch is applied after."""
    print(f"[{ABLATION_NAME}] Loading model: {model_path}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.thinker.omnizip_config = {
        "rho_audio":        FIXED_RHO_AUDIO,
        "rho_video":        FIXED_RHO_VIDEO,
        "g":                FIXED_G,
        "contextual_ratio": FIXED_CONTEXTUAL_RATIO,
    }
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    alloc_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"  Model loaded. VRAM: {alloc_gb:.2f} GB")
    return model, processor, alloc_gb


def unload_model(model, processor):
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Inference ──────────────────────────────────────────────────────────────────
def run_one(model, processor, video_path, dataset, question, choices,
            use_audio, fps, max_pixels, max_frames_vmme, max_frames_other,
            max_new_tokens, temperature):
    ds = (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")
    max_frames = max_frames_vmme if ds in {"video-mme", "videomme"} else max_frames_other

    system_text = SYSTEM_PROMPT + " " + SYSTEM_MCQ_SUFFIX
    prompt = build_prompt(dataset, question, choices)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "fps": fps,
             "max_pixels": max_pixels, "max_frames": max_frames},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio)

    if not videos or videos[0] is None or videos[0].shape[0] == 0:
        raise ValueError("0 video frames decoded")

    num_input_frames = int(videos[0].shape[0])
    model.thinker.nframes = num_input_frames

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=use_audio,
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)

    gen_in = {k: v for k, v in inputs.items() if k not in {"images", "return_tensors", "text"}}
    do_sample = temperature > 0
    gen_kw = dict(
        use_audio_in_video=use_audio,
        return_audio=False,
        thinker_max_new_tokens=max_new_tokens,
        thinker_do_sample=do_sample,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kw["thinker_temperature"] = temperature

    with torch.no_grad():
        prefill_ms, _ = cuda_time_ms(
            lambda: model.generate(**gen_in, **dict(gen_kw, thinker_max_new_tokens=1))
        )
    with torch.no_grad():
        e2e_ms, raw_out = cuda_time_ms(lambda: model.generate(**gen_in, **gen_kw))

    seq = raw_out.sequences if hasattr(raw_out, "sequences") else raw_out
    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, seq)]
    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, num_input_frames, round(prefill_ms, 2), round(e2e_ms, 2)


# ── Per-condition runner ───────────────────────────────────────────────────────
def run_condition(condition, args, out_dir):
    """
    condition: "uniform" or "noaudio"
      uniform  — audio as input, uniform video compression (patch installed)
      noaudio  — no audio input, uniform video compression (patch installed)
    """
    assert condition in CONDITIONS, f"Unknown condition: {condition}"
    force_no_audio = (condition == "noaudio")

    os.makedirs(out_dir, exist_ok=True)
    console_path = os.path.join(out_dir, "console.log")
    stderr_path  = os.path.join(out_dir, "stderr.log")
    results_path = os.path.join(out_dir, "results.jsonl")
    vram_path    = os.path.join(out_dir, "vram_log.jsonl")
    summary_path = os.path.join(out_dir, "run_summary.json")

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    tee  = Tee(console_path, label=f"condition={condition}")
    stee = StderrTee(stderr_path)
    sys.stdout, sys.stderr = tee, stee

    try:
        model, processor, model_alloc_gb = load_model(args.model)
        # Install uniform-video patch AFTER model load
        _install_uniform_video_patch(FIXED_RHO_VIDEO)

        meta = json.loads(Path(args.metadata).read_text())
        runnable = [e for e in meta if e.get("questions")]
        print(f"  {len(runnable)} entries to run  [condition={condition}]")

        correct = total = skipped = 0
        q_idx = 0

        with open(results_path, "w") as rf, open(vram_path, "w") as vf:
            for entry in runnable:
                vp = resolve_video_path(entry["file"], args.videos)
                if vp is None:
                    print(f"  SKIP: video not found — {entry['file']}")
                    skipped += 1
                    continue

                if force_no_audio:
                    use_audio = False
                else:
                    use_audio = check_video_has_audio(vp) and not args.no_audio

                for q in entry["questions"]:
                    question  = q["question"]
                    choices   = q["choices"]
                    answer    = q["answer"].strip().upper()
                    dataset   = entry.get("dataset", "")
                    task_type = q.get("task_type", entry.get("task_type", ""))

                    before_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                    try:
                        pred, decoded, nframes, prefill_ms, e2e_ms = run_one(
                            model, processor, vp, dataset, question, choices,
                            use_audio, args.fps, args.max_pixels,
                            args.max_frames_vmme, args.max_frames_other,
                            args.max_new_tokens, args.temperature,
                        )
                        after_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                        delta = round(after_alloc - before_alloc, 4)
                        status = "ok"
                        err_msg = None
                    except Exception as exc:
                        tb = traceback.format_exc()
                        print(f"  ERROR [{dataset}/{task_type}]: {exc}")
                        pred, decoded = "ERROR", str(exc)
                        nframes = prefill_ms = e2e_ms = 0
                        after_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                        delta = round(after_alloc - before_alloc, 4)
                        status = "error"
                        err_msg = tb

                    is_correct = (pred.strip().upper() == answer)
                    if pred != "ERROR":
                        correct += int(is_correct)
                        total += 1
                    mark = "✓" if is_correct else "✗"
                    print(f"  [{mark}] {dataset}/{task_type} pred={pred} ans={answer} "
                          f"prefill={prefill_ms:.0f}ms e2e={e2e_ms:.0f}ms")

                    result = {
                        "dataset": dataset, "task_type": task_type,
                        "video": entry["file"], "question": question,
                        "answer": answer, "prediction": pred, "correct": is_correct,
                        "orig_nframes": nframes, "used_nframes": nframes,
                        "prefill_ms": prefill_ms, "e2e_ms": e2e_ms,
                        "vram_alloc_delta_gb": delta,
                        "method": ABLATION_NAME,
                        "config": {
                            "rho_video": FIXED_RHO_VIDEO, "rho_audio": FIXED_RHO_AUDIO,
                            "g": FIXED_G, "contextual_ratio": FIXED_CONTEXTUAL_RATIO,
                            "uniform_video": True, "audio_input": not force_no_audio,
                        },
                        "ablation_value": condition,
                        "condition": condition,
                        "status": status,
                    }
                    if err_msg:
                        result["error"] = err_msg[:500]
                    rf.write(json.dumps(result) + "\n"); rf.flush()

                    vram_entry = {
                        "question_idx": q_idx, "dataset": dataset,
                        "before_alloc_gb": round(before_alloc, 4),
                        "after_alloc_gb": round(after_alloc, 4),
                        "delta_gb": delta,
                    }
                    vf.write(json.dumps(vram_entry) + "\n"); vf.flush()
                    q_idx += 1

        acc = correct / total if total else 0.0
        summary = {
            "ablation": ABLATION_NAME, "condition": condition,
            "config": {
                "rho_video": FIXED_RHO_VIDEO, "rho_audio": FIXED_RHO_AUDIO,
                "g": FIXED_G, "contextual_ratio": FIXED_CONTEXTUAL_RATIO,
                "uniform_video": True, "audio_input": not force_no_audio,
            },
            "correct": correct, "total": total, "accuracy": round(acc, 4),
            "skipped": skipped,
        }
        Path(summary_path).write_text(json.dumps(summary, indent=2))
        print(f"\n  condition={condition}: {correct}/{total} = {acc:.2%}")
        return summary

    finally:
        unload_model(model, processor)
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        tee.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=f"Ablation: {ABLATION_NAME}")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--metadata", default="/data/armaan/purs/metadata.json")
    parser.add_argument("--videos",   default="/data/armaan/purs/videos")
    parser.add_argument("--output_root", default="/data/armaan/purs/ablation_outputs")
    parser.add_argument("--condition", choices=CONDITIONS, default=None,
                        help="Which condition to run: 'uniform' or 'noaudio'.")
    parser.add_argument("--all_conditions", action="store_true",
                        help="Run both conditions sequentially.")
    parser.add_argument("--no_audio", action="store_true",
                        help="Globally suppress audio input (overrides per-video detection).")
    parser.add_argument("--fps",              type=float, default=DEFAULT_FPS)
    parser.add_argument("--max_pixels",       type=int,   default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--max_frames_vmme",  type=int,   default=DEFAULT_MAX_FRAMES_VMME)
    parser.add_argument("--max_frames_other", type=int,   default=DEFAULT_MAX_FRAMES_OTHER)
    parser.add_argument("--max_new_tokens",   type=int,   default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature",      type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    if args.condition is None and not args.all_conditions:
        parser.error("Specify --condition {uniform,noaudio} or --all_conditions")

    conds = CONDITIONS if args.all_conditions else [args.condition]

    sweep_dir = os.path.join(args.output_root, ABLATION_NAME)
    os.makedirs(sweep_dir, exist_ok=True)

    all_summaries = []
    for cond in conds:
        out_dir = os.path.join(sweep_dir, cond)
        print(f"\n{'='*60}")
        print(f"[{ABLATION_NAME}] condition={cond}  → {out_dir}")
        print(f"{'='*60}")
        summary = run_condition(cond, args, out_dir)
        all_summaries.append(summary)

    sweep_summary_path = os.path.join(sweep_dir, "sweep_summary.json")
    Path(sweep_summary_path).write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[{ABLATION_NAME}] Done. Summary → {sweep_summary_path}")
    for s in all_summaries:
        print(f"  condition={s['condition']}: {s['correct']}/{s['total']} = {s['accuracy']:.2%}")


if __name__ == "__main__":
    main()
