"""Loader for the existing L6 saliency cache.

The cache file is produced by `precompute_l6_saliency.py` (already in the repo root)
and lives at `vizzing/layer_depth_all_full.jsonl`. Each record contains BOTH audio and
video saliency scores at the configured layer (default L6):

    {
      "video_id": "...",
      "dataset":  "...",
      "file":     "videos/...",
      "raw_audio_scores": {"6": [[<N_audio floats>]]},
      "raw_video_scores": {"6": [[<N_video floats>]]},
      ...
    }

Scores are text->modality Q*K cross-attention values, mean-reduced over heads and text
queries. They are 1D vectors of length N_modality, NOT hidden states.

Use `load_video_scores()` for VideoZip; the audio side is already handled by
`eval_qwen_omni_zip_cached.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

DEFAULT_CACHE_PATHS = [
    "vizzing/layer_depth_all_full.jsonl",
    "/workspace/vizzing/layer_depth_all_full.jsonl",
    "/data/armaan/purs/vizzing/layer_depth_all_full.jsonl",
]


def _normalize_path(p: str) -> str:
    return p.replace("\\", "/").strip().lower()


def _resolve_cache_path(explicit: Optional[str]) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    for candidate in DEFAULT_CACHE_PATHS:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "L6 cache JSONL not found. Tried: "
        + ", ".join([explicit] if explicit else [] + DEFAULT_CACHE_PATHS)
    )


def iter_records(cache_path: Optional[str] = None) -> Iterable[dict]:
    """Yield raw records from the L6 cache JSONL."""
    path = _resolve_cache_path(cache_path)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_video_scores(
    cache_path: Optional[str] = None,
    layer: int = 6,
    key: str = "file",
) -> Dict[str, np.ndarray]:
    """Build {video_key: video_score_vector} for the requested layer.

    Args:
        cache_path: explicit JSONL path; falls back to DEFAULT_CACHE_PATHS.
        layer: which layer's scores to extract (default 6).
        key: which record field to use as the lookup key. "file" is the relative
             video path; "video_id" is the safe basename. Use whichever is more stable
             across the audio-side eval pipeline.

    Returns:
        Mapping from video_key -> 1D float32 numpy array of length N_video. Records
        without a video score for the requested layer are skipped silently.
    """
    out: Dict[str, np.ndarray] = {}
    layer_str = str(layer)
    for rec in iter_records(cache_path):
        scores_dict = rec.get("raw_video_scores", {})
        per_question = scores_dict.get(layer_str)
        if not per_question:
            continue
        first = per_question[0]
        if not first:
            continue
        arr = np.asarray(first, dtype=np.float32)
        if arr.size == 0:
            continue
        k = rec.get(key)
        if k is None:
            continue
        out[_normalize_path(k) if key == "file" else str(k)] = arr
    return out


def load_audio_scores(
    cache_path: Optional[str] = None,
    layer: int = 6,
    key: str = "file",
) -> Dict[str, np.ndarray]:
    """Audio-side equivalent of `load_video_scores`. Useful for cross-checking against
    the existing OmniZip+L6 eval, and for the §7 hybrid-score ablation in
    docs/futureplanning.md."""
    out: Dict[str, np.ndarray] = {}
    layer_str = str(layer)
    for rec in iter_records(cache_path):
        scores_dict = rec.get("raw_audio_scores", {})
        per_question = scores_dict.get(layer_str)
        if not per_question:
            continue
        first = per_question[0]
        if not first:
            continue
        arr = np.asarray(first, dtype=np.float32)
        if arr.size == 0:
            continue
        k = rec.get(key)
        if k is None:
            continue
        out[_normalize_path(k) if key == "file" else str(k)] = arr
    return out


def cache_summary(cache_path: Optional[str] = None, layer: int = 6) -> dict:
    """Per-dataset coverage stats. Cheap, runs on CPU only."""
    path = _resolve_cache_path(cache_path)
    n_total = 0
    by_dataset: Dict[str, dict] = {}
    layer_str = str(layer)
    for rec in iter_records(path):
        n_total += 1
        ds = rec.get("dataset", "unknown")
        slot = by_dataset.setdefault(ds, {
            "n_records": 0,
            "n_with_audio": 0,
            "n_with_video": 0,
            "audio_token_lens": [],
            "video_token_lens": [],
        })
        slot["n_records"] += 1
        a_list = rec.get("raw_audio_scores", {}).get(layer_str)
        v_list = rec.get("raw_video_scores", {}).get(layer_str)
        if a_list and a_list[0]:
            slot["n_with_audio"] += 1
            slot["audio_token_lens"].append(len(a_list[0]))
        if v_list and v_list[0]:
            slot["n_with_video"] += 1
            slot["video_token_lens"].append(len(v_list[0]))
    return {
        "cache_path": path,
        "layer": layer,
        "n_total": n_total,
        "by_dataset": by_dataset,
    }
