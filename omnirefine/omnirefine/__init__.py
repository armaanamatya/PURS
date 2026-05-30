"""OmniRefine -- training-free alignment-aware cooperative compression.

Reference implementation of arXiv:2605.12056v1 (Deng et al., 2026).
Citation-anchored; ambiguities documented in REPRODUCTION_NOTES.md.
"""
from .config import OmniRefineConfig
from .cpcr import (
    RefinedChunk,
    correspondence_preserving_refinement,
    refine_chunks,
    similarity_field,
)
from .video_compress import VideoNode, compress_video_chunk, video_retention
from .audio_compress import AudioResult, compress_audio_chunk, semantic_intervals
from .budget import audio_merge_ratio, audio_retention
from .pipeline import KeepMask, ProbeInputs, compress
from .adapter import OmniRefineAdapter
from .torch_runtime import (
    TorchOmniRefineDiagnostics,
    apply_keep_to_prefill,
    build_probe_inputs_from_qwen_prefill,
    omnirefine_qwen_prefill,
)

__all__ = [
    "OmniRefineConfig",
    "RefinedChunk",
    "correspondence_preserving_refinement",
    "refine_chunks",
    "similarity_field",
    "VideoNode",
    "compress_video_chunk",
    "video_retention",
    "AudioResult",
    "compress_audio_chunk",
    "semantic_intervals",
    "audio_merge_ratio",
    "audio_retention",
    "KeepMask",
    "ProbeInputs",
    "compress",
    "OmniRefineAdapter",
    "TorchOmniRefineDiagnostics",
    "apply_keep_to_prefill",
    "build_probe_inputs_from_qwen_prefill",
    "omnirefine_qwen_prefill",
]

__version__ = "0.1.0"
