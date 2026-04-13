"""
eval_qwen_omni.py
Runs Qwen2.5-Omni-7B on all videos in metadata.json and answers MCQ questions.
Reads Q&A from metadata.json — no HuggingFace needed on the server.

Generation is always text-only (disable_talker + return_audio=False). Default input is
video+audio (from file)+text when the video has an audio stream; --no_audio only disables input audio.

Usage:
    python eval_qwen_omni.py --metadata metadata.json --videos /workspace/videos --output results.jsonl
    python eval_qwen_omni.py --metadata metadata.json --videos /workspace/videos --output results.jsonl --category lecture
"""

import argparse
import glob
import importlib.util
import json
import os
import shutil
import sys
import time
import torch
from datetime import datetime
from pathlib import Path
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OMNIZIP_DIR = os.path.join(_REPO_ROOT, "OmniZip-main")
QWEN_OMNI_UTILS_SRC = os.path.join(OMNIZIP_DIR, "qwen-omni-utils", "src")
if QWEN_OMNI_UTILS_SRC not in sys.path:
    sys.path.insert(0, QWEN_OMNI_UTILS_SRC)

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


class Tee:
    """Writes to both stdout and a log file simultaneously."""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
        self.log.write(f"\n{'='*60}\n")
        self.log.write(f"RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*60}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()

    def close(self):
        self.log.close()


class StderrTee:
    """Duplicate stderr to a file (use instead of shell `| tee` when the directory may not exist yet)."""

    def __init__(self, log_file, terminal):
        self.log = log_file
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


ENV_MODEL_PATH_KEY = "QWEN_OMNI_MODEL_PATH"
DEFAULT_MODEL_PATHS = {
    "none": "/data/armaan/models/Qwen2.5-Omni-7B",
    "gptq": "/data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4",
    "awq": "/data/armaan/models/Qwen2.5-Omni-7B-AWQ",
}
DEFAULT_MODEL_PATH = DEFAULT_MODEL_PATHS["none"]
FALLBACK_MODEL_PATH = "/workspace/model"
LOW_VRAM_MODE_DIR = os.path.join(_REPO_ROOT, "Qwen2.5-Omni", "low-VRAM-mode")

MODEL_PATH = os.environ.get(ENV_MODEL_PATH_KEY) or DEFAULT_MODEL_PATH
MODEL_LOADED_ALLOC_GB: float | None = None
MODEL_LOADED_RESERVED_GB: float | None = None

# ── Prompts (same sources as Qwen2.5-Omni web_demo + OmniZip lmms-eval task utils) ─────────

SYSTEM_PROMPT_DEFAULT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

# Reduces chatty continuations ("What do you think?") on multiple-choice.
SYSTEM_MCQ_SUFFIX = (
    "For multiple-choice questions, reply with only one letter: A, B, C, or D. "
    "Do not explain, do not ask follow-up questions, and do not add text after the letter."
)

_WORLD_SENSE_SYS = (
    "Carefully watch this video and pay attention to every detail. "
    "Based on your observations, select the best option that accurately addresses the question."
)

_WORLD_SENSE_FRAMES_AUDIO = """
These are the frames of a video and the corresponding audio. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

_VIDEO_MME_OPTION_PROMPT = (
    "Select the best answer to the following multiple-choice question based on the video and the subtitles. "
    "Respond with only the letter (A, B, C, or D) of the correct option."
)

_VIDEO_MME_POST_PROMPT = "The best answer is:"


def _format_choice_lines(choices: list) -> str:
    if not choices:
        return ""
    if choices[0].startswith("A"):
        return "\n".join(choices)
    return "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))


def _build_user_prompt_video_mme(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    question_block = question + "\n" + option_lines
    return _VIDEO_MME_OPTION_PROMPT + "\n" + question_block + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_worldsense(question: str, choices: list) -> str:
    parts: list[str] = [_WORLD_SENSE_SYS, _WORLD_SENSE_FRAMES_AUDIO, question + "\n"]
    for op in choices:
        parts.append(op + "\n")
    return "".join(parts)


def _build_user_prompt_daily_omni(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    head = (
        "Listen and watch the video carefully. "
        "Select the best answer to the following multiple-choice question. "
        "Respond with only the letter (A, B, C, or D) of the correct option."
    )
    return head + "\n" + question + "\n" + option_lines + "\n" + _VIDEO_MME_POST_PROMPT


def _build_user_prompt_default(question: str, choices: list) -> str:
    option_lines = _format_choice_lines(choices)
    return (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, or D) of the correct option.\n"
        + question
        + "\n"
        + option_lines
        + "\n"
        + _VIDEO_MME_POST_PROMPT
    )


def _canonicalize_dataset_name(dataset: str | None) -> str:
    return (dataset or "").strip().lower().replace("_", "-").replace(" ", "-")


def build_user_prompt_for_dataset(dataset: str, question: str, choices: list) -> str:
    dataset_name = _canonicalize_dataset_name(dataset)
    if dataset_name in {"video-mme", "videomme"}:
        return _build_user_prompt_video_mme(question, choices)
    if dataset_name == "worldsense":
        return _build_user_prompt_worldsense(question, choices)
    if dataset_name in {"daily-omni", "dailyomni"}:
        return _build_user_prompt_daily_omni(question, choices)
    return _build_user_prompt_default(question, choices)


def resolve_model_dtype(dtype_name: str):
    table = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in table:
        keys = ", ".join(sorted(table.keys()))
        raise ValueError(f"Unknown dtype_name {dtype_name!r}; expected one of: {keys}")
    return table[dtype_name]


def _quantized_dtype(dtype_name: str, quantization: str) -> torch.dtype:
    if quantization == "none":
        return resolve_model_dtype(dtype_name)
    if dtype_name != "float16":
        print(
            f"Quantized {quantization.upper()} loading is forcing torch_dtype=float16 "
            f"(ignoring --dtype {dtype_name})."
        )
    return torch.float16


def _resolve_model_path(explicit_model_path: str | None, quantization: str) -> str:
    if explicit_model_path:
        return explicit_model_path
    env_model_path = os.environ.get(ENV_MODEL_PATH_KEY)
    if env_model_path:
        return env_model_path
    candidate = DEFAULT_MODEL_PATHS[quantization]
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(FALLBACK_MODEL_PATH):
        return FALLBACK_MODEL_PATH
    return candidate


def _ensure_low_vram_mode_dir() -> None:
    if LOW_VRAM_MODE_DIR not in sys.path:
        sys.path.insert(0, LOW_VRAM_MODE_DIR)


def _replace_transformers_qwen_model_module(module_path: str) -> None:
    original_mod_name = "transformers.models.qwen2_5_omni.modeling_qwen2_5_omni"
    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]
    spec = importlib.util.spec_from_file_location(original_mod_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import patched Qwen2.5-Omni module from {module_path}")
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)
    sys.modules[original_mod_name] = new_mod


def _unwrap_qwen_model(model: object) -> torch.nn.Module:
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "thinker"):
        return inner
    return model


def _capture_current_vram_gb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
    )


def _record_model_loaded_vram() -> None:
    global MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB
    MODEL_LOADED_ALLOC_GB, MODEL_LOADED_RESERVED_GB = _capture_current_vram_gb()
    print(
        f"Model loaded. VRAM: {MODEL_LOADED_ALLOC_GB:.1f} GB allocated, "
        f"{MODEL_LOADED_RESERVED_GB:.1f} GB reserved"
    )


def _model_dtype(model: object) -> torch.dtype:
    qwen_model = _unwrap_qwen_model(model)
    dt = getattr(qwen_model, "dtype", None)
    if dt is not None:
        return dt
    dt = getattr(model, "dtype", None)
    if dt is not None:
        return dt
    return next(qwen_model.parameters()).dtype


def _model_device(model: object) -> torch.device:
    qwen_model = _unwrap_qwen_model(model)
    try:
        return next(qwen_model.parameters()).device
    except StopIteration:
        device = getattr(qwen_model, "device", None)
        if device is not None:
            return torch.device(device)
        device = getattr(model, "device", None)
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_input_device(model: object) -> torch.device:
    qwen_model = _unwrap_qwen_model(model)
    try:
        return qwen_model.thinker.model.embed_tokens.weight.device
    except Exception:
        return _model_device(model)


def _move_quantized_modal_modules_to_cuda(model: object) -> None:
    qwen_model = _unwrap_qwen_model(model)
    qwen_model.thinker.model.embed_tokens = qwen_model.thinker.model.embed_tokens.to("cuda")
    qwen_model.thinker.visual = qwen_model.thinker.visual.to("cuda")
    qwen_model.thinker.audio_tower = qwen_model.thinker.audio_tower.to("cuda")
    qwen_model.thinker.visual.rotary_pos_emb = qwen_model.thinker.visual.rotary_pos_emb.to("cuda")
    qwen_model.thinker.model.rotary_emb = qwen_model.thinker.model.rotary_emb.to("cuda")
    for layer in qwen_model.thinker.model.layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to("cuda")


def _disable_talker(model: object) -> None:
    qwen_model = _unwrap_qwen_model(model)
    if hasattr(qwen_model, "disable_talker"):
        qwen_model.disable_talker()


def _load_qwen2_5_omni_gptq(dtype_name: str):
    _ensure_low_vram_mode_dir()
    from gptqmodel import GPTQModel
    from gptqmodel.models.auto import MODEL_MAP
    from gptqmodel.models.base import BaseGPTQModel
    from gptqmodel.models._const import SUPPORTED_MODELS
    from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniForConditionalGeneration as LowVramQwenModel
    from transformers.utils.hub import cached_file

    class Qwen25OmniThinkerGPTQ(BaseGPTQModel):
        loader = LowVramQwenModel
        base_modules = [
            "thinker.model.embed_tokens",
            "thinker.model.norm",
            "token2wav",
            "thinker.audio_tower",
            "thinker.model.rotary_emb",
            "thinker.visual",
            "talker",
        ]
        pre_lm_head_norm_module = "thinker.model.norm"
        require_monkeypatch = False
        layers_node = "thinker.model.layers"
        layer_type = "Qwen2_5OmniDecoderLayer"
        layer_modules = [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

    MODEL_MAP["qwen2_5_omni"] = Qwen25OmniThinkerGPTQ
    if "qwen2_5_omni" not in SUPPORTED_MODELS:
        SUPPORTED_MODELS.append("qwen2_5_omni")

    @classmethod
    def patched_from_config(cls, config, *args, **kwargs):
        kwargs.pop("trust_remote_code", None)
        model = cls._from_config(config, **kwargs)
        spk_path = cached_file(
            MODEL_PATH,
            "spk_dict.pt",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if spk_path is None:
            raise ValueError(f"Speaker dictionary not found at {spk_path}")
        model.load_speakers(spk_path)
        return model

    LowVramQwenModel.from_config = patched_from_config

    dt = _quantized_dtype(dtype_name, "gptq")
    print(
        f"Loading qwen2.5-omni GPTQ model from {MODEL_PATH} "
        f"(torch_dtype={dt}, attn_implementation=flash_attention_2) ..."
    )
    model = GPTQModel.load(
        MODEL_PATH,
        device_map={
            "thinker.model": "cuda",
            "thinker.lm_head": "cuda",
            "thinker.visual": "cuda",
            "thinker.audio_tower": "cuda",
            "talker": "cpu",
            "token2wav": "cpu",
        },
        torch_dtype=dt,
        attn_implementation="flash_attention_2",
    )
    _move_quantized_modal_modules_to_cuda(model)
    _disable_talker(model)
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    _record_model_loaded_vram()
    return model, processor


def _load_qwen2_5_omni_awq(dtype_name: str):
    _ensure_low_vram_mode_dir()
    module_path = os.path.join(LOW_VRAM_MODE_DIR, "modeling_qwen2_5_omni_low_VRAM_mode.py")
    _replace_transformers_qwen_model_module(module_path)

    from awq.models.base import BaseAWQForCausalLM
    from modeling_qwen2_5_omni_low_VRAM_mode import Qwen2_5OmniDecoderLayer

    class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
        layer_type = "Qwen2_5OmniDecoderLayer"
        max_seq_len_key = "max_position_embeddings"
        modules_to_not_convert = ["visual"]

        @staticmethod
        def get_model_layers(model: torch.nn.Module):
            return model.thinker.model.layers

        @staticmethod
        def get_act_for_scaling(module: Qwen2_5OmniDecoderLayer):
            return dict(is_scalable=False)

        @staticmethod
        def move_embed(model: torch.nn.Module, device: str):
            model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
            model.thinker.visual = model.thinker.visual.to(device)
            model.thinker.audio_tower = model.thinker.audio_tower.to(device)
            model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(device)
            model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)
            for layer in model.thinker.model.layers:
                layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

        @staticmethod
        def get_layers_for_scaling(module: Qwen2_5OmniDecoderLayer, input_feat, module_kwargs):
            layers = []
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj],
                    inp=input_feat["self_attn.q_proj"],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )
            return layers

    dt = _quantized_dtype(dtype_name, "awq")
    print(
        f"Loading qwen2.5-omni AWQ model from {MODEL_PATH} "
        f"(torch_dtype={dt}, attn_implementation=flash_attention_2) ..."
    )
    model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
        MODEL_PATH,
        model_type="qwen2_5_omni",
        torch_dtype=dt,
        attn_implementation="flash_attention_2",
    )
    _move_quantized_modal_modules_to_cuda(model)
    _disable_talker(model)
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    _record_model_loaded_vram()
    return model, processor


# ── Audio: lmms-eval Qwen2_5_Omni sets use_audio_in_video per video via _check_if_video_has_audio ──

def check_video_has_audio(video_path: str) -> bool:
    """Return True if the video file contains an audio stream."""
    try:
        import av
        container = av.open(video_path)
        has = len(container.streams.audio) > 0
        container.close()
        return has
    except Exception:
        return False


# ── Answer parsing: see mcq_answer_parse.py ───────────────────────────────────

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_variant: str, dtype_name: str, quantization: str):
    if quantization != "none":
        if model_variant != "qwen2.5-omni":
            raise ValueError("Quantized loading is currently supported only for --model_variant qwen2.5-omni.")
        if quantization == "gptq":
            return _load_qwen2_5_omni_gptq(dtype_name)
        if quantization == "awq":
            return _load_qwen2_5_omni_awq(dtype_name)
        raise ValueError(f"Unsupported quantization mode: {quantization}")

    dt = resolve_model_dtype(dtype_name)
    print(f"Loading {model_variant} model from {MODEL_PATH} (torch_dtype={dtype_name}) ...")
    if model_variant == "qwen2.5-omni":
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=dt,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    elif model_variant == "qwen3-omni":
        try:
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as e:
            raise ImportError(
                "Qwen3 Omni classes not found in your installed transformers. "
                "Install a transformers version that includes Qwen3-Omni support, "
                "or run with --model_variant qwen2.5-omni."
            ) from e

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=dt,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
    else:
        raise ValueError(f"Unsupported model variant: {model_variant}")
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    _record_model_loaded_vram()
    return model, processor


def _prepare_omni_inputs(model: torch.nn.Module, inputs: object) -> object:
    """Move processor outputs to the model device; cast only floating tensors to model dtype (never input_ids)."""
    device = _model_input_device(model)
    dtype = _model_dtype(model)
    inputs = inputs.to(device)
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            if value.device != device:
                value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
            inputs[key] = value
    return inputs


def _generation_output_token_ids(gen_out: object) -> torch.Tensor:
    """HF generate() may return a LongTensor or GenerateDecoderOnlyOutput; iterating the latter iterates dict keys."""
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    return gen_out


# Keys Qwen2.5-OmniProcessor may put on the batch but model.generate() does not accept (HF warnings).
_OMNI_GENERATE_BATCH_DROP = frozenset({"images", "return_tensors", "text"})


def _batch_for_omni_generate(batch: object, model: object | None = None) -> dict:
    out = {k: v for k, v in batch.items() if k not in _OMNI_GENERATE_BATCH_DROP}
    if model is None:
        return out

    device = _model_input_device(model)
    dtype = _model_dtype(model)
    for key, value in list(out.items()):
        if isinstance(value, torch.Tensor):
            if value.device != device:
                value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
            out[key] = value
    return out


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    processor,
    video_path: str,
    dataset: str,
    question: str,
    choices: list,
    fps: float,
    max_pixels: int,
    max_new_tokens: int,
    use_audio: bool,
    max_frames: int | None = None,
    measure_prefill: bool = False,
) -> tuple[str, str, dict]:
    prompt = build_user_prompt_for_dataset(dataset, question, choices)

    system_text = SYSTEM_PROMPT_DEFAULT + " " + SYSTEM_MCQ_SUFFIX
    video_element = {"type": "video", "video": video_path, "fps": fps, "max_pixels": max_pixels}
    if max_frames is not None:
        video_element["max_frames"] = max_frames
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [
            video_element,
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    effective_use_audio = use_audio
    try:
        audios, images, videos = process_mm_info(messages, use_audio_in_video=effective_use_audio)
    except Exception as err:
        if not effective_use_audio:
            raise
        raise RuntimeError(
            f"Audio input was requested for {video_path} but audio decoding failed. "
            "This evaluator keeps audio mandatory unless you explicitly pass --no_audio."
        ) from err

    inputs = processor(
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=effective_use_audio,
    )
    inputs = _prepare_omni_inputs(model, inputs)

    tokenizer = processor.tokenizer
    # Text-only: use return_audio=False on the top-level Omni wrapper. Do not pass generation_mode here —
    # many transformers builds forward unknown kwargs to thinker.generate(), which then errors on
    # unused model_kwargs (generation_mode is only valid on some wrapper versions).
    gen_kw: dict = {
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
        gen_kw["temperature"] = 0.0
        gen_kw["do_sample"] = False

    gen_in = _batch_for_omni_generate(inputs, model=model)

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
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, seq_ids)]
    decoded = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    letter = parse_answer(decoded, choices)
    return letter, decoded, timing

# ── Video lookup ──────────────────────────────────────────────────────────────

def resolve_video_path(file_field: str, videos_dir: str) -> str | None:
    """Resolve metadata paths robustly across Windows/Linux separators and repeated basenames."""
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
    suffix_matches = []
    for match in glob.glob(os.path.join(videos_dir, "**", filename), recursive=True):
        if match.replace("\\", "/").endswith(rel_norm):
            suffix_matches.append(match)
    if suffix_matches:
        return suffix_matches[0]

    basename_matches = glob.glob(os.path.join(videos_dir, "**", filename), recursive=True)
    if len(basename_matches) == 1:
        return basename_matches[0]

    return None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=None, help=f"Model path (or set {ENV_MODEL_PATH_KEY}).")
    parser.add_argument("--metadata", default="metadata.json", help="Path to enriched metadata.json")
    parser.add_argument("--videos",   default="/workspace/videos", help="Directory containing video files")
    parser.add_argument("--output",   default="/workspace/results.jsonl", help="Output JSONL file")
    parser.add_argument("--log",      default="/workspace/eval.log", help="Log file (appended each run)")
    parser.add_argument("--errors_log", default=None, help="Where to append tracebacks (default: alongside --log).")
    parser.add_argument("--category", default=None, help="Only run this category (e.g. lecture)")
    parser.add_argument("--fps",      type=float, default=2.0, help="Video sampling fps")
    parser.add_argument("--max_pixels", type=int, default=360*420, help="Max pixels per frame")
    parser.add_argument("--max_frames_videomme", type=int, default=768,
                        help="Max frames for VideoMME videos (paper default: 768)")
    parser.add_argument("--max_frames_other", type=int, default=128,
                        help="Max frames for non-VideoMME datasets (paper default: 128)")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Generation cap (lmms-eval qwen2_5_omni default)")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model weights dtype for non-quantized runs. Quantized GPTQ/AWQ loads force float16.",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "gptq", "awq"],
        help="Load a Qwen2.5-Omni quantized checkpoint via Qwen's reference GPTQ/AWQ path.",
    )
    parser.add_argument(
        "--no_audio",
        action="store_true",
        help="Input: do not load audio from the video. Generation stays text-only either way.",
    )
    parser.add_argument("--vram_log", default="/workspace/vram_log.jsonl", help="Per-question VRAM JSONL (allocated/reserved)")
    parser.add_argument(
        "--stderr_log",
        default=None,
        help="Append stderr to this file (parent dirs created automatically). Prefer this over `| tee` when run2/... may not exist yet.",
    )
    parser.add_argument(
        "--model_variant",
        default="qwen2.5-omni",
        choices=["qwen2.5-omni", "qwen3-omni"],
        help="Which Omni model variant to run",
    )
    parser.add_argument(
        "--measure_prefill",
        action="store_true",
        help="Measure prefill time (TTFT) via generate(max_new_tokens=1) with CUDA events. "
             "Adds a second generate() call per question — use for benchmarking only.",
    )
    args = parser.parse_args()

    global MODEL_PATH
    MODEL_PATH = _resolve_model_path(args.model, args.quantization)

    if (not args.no_audio) and shutil.which("ffmpeg") is None:
        print(
            "WARNING: ffmpeg not found in PATH. Decoding MP4 audio needs ffmpeg (or use --no_audio, or "
            "QWEN_OMNI_AUDIO_WAV_ROOT + pre-extracted .wav). Without sudo: "
            "`conda install -c conda-forge ffmpeg`, or put a static ffmpeg in ~/bin and export PATH, "
            "or ask an admin for system ffmpeg.\n"
        )

    errors_log_path = args.errors_log or os.path.join(os.path.dirname(args.log) or ".", "errors.log")
    paths_for_dirs = [args.log, args.output, args.vram_log, errors_log_path]
    if args.stderr_log is not None:
        paths_for_dirs.append(args.stderr_log)
    for path in paths_for_dirs:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    if args.stderr_log is not None:
        _stderr_f = open(args.stderr_log, "a", encoding="utf-8")
        sys.stderr = StderrTee(_stderr_f, sys.__stderr__)

    tee = Tee(args.log)
    sys.stdout = tee

    meta = json.loads(Path(args.metadata).read_text())
    print(f"Loaded {len(meta)} entries")

    # Filter by dataset
    if args.category:
        meta = [e for e in meta if e.get("dataset") == args.category or e.get("task_type") == args.category]
        print(f"Filtered to {len(meta)} entries for '{args.category}'")

    # Only entries that have questions
    runnable = [e for e in meta if e.get("questions")]
    skipped_no_qa = len(meta) - len(runnable)
    if skipped_no_qa:
        print(f"Skipping {skipped_no_qa} entries with no Q&A")
    print(f"Running {len(runnable)} entries")
    print(
        "Each line in --output is written after one question finishes; the first can take many minutes "
        "(video decode + optional MP4 audio via librosa/audioread + generation). Use --no_audio to skip separate audio loading.\n"
    )
    audio_mode = "video+audio+text" if not args.no_audio else "video+text (--no_audio)"
    print(
        f"Input mode: {audio_mode}; sample_fps={args.fps}; max_pixels={args.max_pixels}; "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print("Note: qwen-vl-utils/decord logs the source video's native fps, not the requested sample_fps.\n")

    if not runnable:
        print("Nothing to run. Exiting.")
        return

    model, processor = load_model(args.model_variant, args.dtype, args.quantization)

    correct = total = skipped_no_video = 0
    results = []

    with open(args.output, "w") as out_f, open(args.vram_log, "w") as vram_f:
        for entry in runnable:
            video_path = resolve_video_path(entry["file"], args.videos)
            entry_label = f"{entry.get('dataset','?')}/{entry.get('task_type','?')}"
            if video_path is None:
                print(f"  SKIP {entry_label}: video file not found")
                skipped_no_video += 1
                continue

            use_audio = (not args.no_audio) and check_video_has_audio(video_path)
            if (not args.no_audio) and (not use_audio):
                print(f"  INFO {entry_label}: no audio stream detected; using video+text input only.")

            for q in entry["questions"]:
                question  = q["question"]
                choices   = q["choices"]
                answer    = q["answer"].strip().upper()
                task_type = q.get("task_type", entry.get("task_type", ""))
                dataset   = entry.get("dataset", "")

                try:
                    before_alloc_gb, before_resv_gb = _capture_current_vram_gb()
                    torch.cuda.reset_peak_memory_stats()
                    ds_name = _canonicalize_dataset_name(dataset)
                    max_frames = args.max_frames_videomme if ds_name in {"video-mme", "videomme"} else args.max_frames_other
                    pred, reasoning, timing = run_inference(
                        model,
                        processor,
                        video_path,
                        dataset,
                        question,
                        choices,
                        args.fps,
                        args.max_pixels,
                        args.max_new_tokens,
                        use_audio,
                        max_frames=max_frames,
                        measure_prefill=args.measure_prefill,
                    )
                    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                    peak_resv_gb = torch.cuda.max_memory_reserved() / 1024**3
                    curr_alloc_gb = torch.cuda.memory_allocated() / 1024**3
                    curr_resv_gb = torch.cuda.memory_reserved() / 1024**3
                    vram_entry = {
                        "entry": entry_label,
                        "task_type": task_type,
                        "status": "ok",
                        "quantization": args.quantization,
                        "duration_s": entry.get("duration_s"),
                        "model_loaded_alloc_gb": round(MODEL_LOADED_ALLOC_GB or 0.0, 2),
                        "model_loaded_reserved_gb": round(MODEL_LOADED_RESERVED_GB or 0.0, 2),
                        "before_alloc_gb": round(before_alloc_gb, 2),
                        "before_reserved_gb": round(before_resv_gb, 2),
                        "peak_alloc_gb": round(peak_alloc_gb, 2),
                        "peak_reserved_gb": round(peak_resv_gb, 2),
                        "after_alloc_gb": round(curr_alloc_gb, 2),
                        "after_reserved_gb": round(curr_resv_gb, 2),
                        **timing,
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    print(f"  ERROR {entry_label}: {type(e).__name__}: {e!r}")
                    peak_alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                    peak_resv_gb = torch.cuda.max_memory_reserved() / 1024**3
                    curr_alloc_gb, curr_resv_gb = _capture_current_vram_gb()
                    vram_entry = {
                        "entry": entry_label,
                        "task_type": task_type,
                        "status": "error",
                        "quantization": args.quantization,
                        "duration_s": entry.get("duration_s"),
                        "model_loaded_alloc_gb": round(MODEL_LOADED_ALLOC_GB or 0.0, 2),
                        "model_loaded_reserved_gb": round(MODEL_LOADED_RESERVED_GB or 0.0, 2),
                        "before_alloc_gb": round(before_alloc_gb, 2),
                        "before_reserved_gb": round(before_resv_gb, 2),
                        "peak_alloc_gb": round(peak_alloc_gb, 2),
                        "peak_reserved_gb": round(peak_resv_gb, 2),
                        "after_alloc_gb": round(curr_alloc_gb, 2),
                        "after_reserved_gb": round(curr_resv_gb, 2),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                    vram_f.write(json.dumps(vram_entry) + "\n")
                    vram_f.flush()
                    with open(errors_log_path, "a") as ef:
                        ef.write(f"\n--- {entry_label} ---\n{tb}\n")
                    pred, reasoning = "ERROR", str(e)
                    timing = {}

                is_correct = pred.strip().upper() == answer
                if is_correct:
                    correct += 1
                total += 1

                result = {
                    "model_variant": args.model_variant,
                    "quantization": args.quantization,
                    "dataset":    entry.get("dataset"),
                    "task_type":  task_type,
                    "duration_s": entry.get("duration_s"),
                    "question":   question,
                    "choices":    choices,
                    "answer":     answer,
                    "prediction": pred,
                    "correct":    is_correct,
                    "reasoning":  reasoning,
                    **timing,
                }
                out_f.write(json.dumps(result) + "\n")
                results.append(result)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {entry_label} [{task_type}] pred={pred} ans={answer}")

    # Final summary
    acc = correct / total if total else 0
    print(f"\n{'='*50}")
    print(f"Model variant: {args.model_variant}")
    print(f"Quantization:  {args.quantization}")
    print(f"Accuracy:      {correct}/{total} = {acc:.2%}")
    print(f"Skipped (no video): {skipped_no_video}")
    print(f"Results saved: {args.output}")

    # Per-dataset breakdown
    datasets: dict = {}
    for r in results:
        d = r["dataset"]
        datasets.setdefault(d, {"correct": 0, "total": 0})
        datasets[d]["total"] += 1
        if r["correct"]:
            datasets[d]["correct"] += 1
    print(f"\nPer-dataset:")
    for ds, s in sorted(datasets.items()):
        print(f"  {ds:<20} {s['correct']}/{s['total']} = {s['correct']/s['total']:.2%}")

if __name__ == "__main__":
    main()
