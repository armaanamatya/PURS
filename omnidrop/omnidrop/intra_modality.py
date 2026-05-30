"""Intra-modality token pruning prior to the LLM (OmniDrop Section 3.4).

Before any tokens enter the decoder, each modality is independently compressed:

  * Audio: keep the highest-attention tokens from the last audio-encoder layer
    (following OmniZip [31]). Paper audio prune ratio = 0.3 -> 70% retained.
  * Video: Temporal Token Merging (TTM) from Dycoke [24] over every 4 frames,
    prune-rate 0.8 -> 40% retained.

Combined target: ~45% of tokens retained before the LLM (App. A: per chunk
50 audio + 288 video tokens -> 0.7*50 + 0.4*288 = 150.2 / 338 ~= 44%).

The video routine here is a faithful re-implementation of the *described*
Dycoke TTM mechanism (anchor frame kept, 80% of redundant tokens dropped from
the other 3 frames by similarity), not a port of the Dycoke codebase.
"""
from __future__ import annotations

import numpy as np


def prune_audio_by_attention(
    audio_attn_saliency: np.ndarray,
    prune_ratio: float = 0.3,
) -> np.ndarray:
    """Keep the most salient audio tokens (OmniZip-style, Sec. 3.4).

    Args:
        audio_attn_saliency: 1-D saliency per audio token (e.g. mean attention
            received in the last audio-encoder layer). Higher = more salient.
        prune_ratio: fraction to drop (paper: 0.3 -> 70% retained).

    Returns:
        Sorted 1-D int array of audio-token indices to KEEP.
    """
    s = np.asarray(audio_attn_saliency, dtype=np.float64)
    n = s.shape[0]
    n_keep = int(round(n * (1.0 - prune_ratio)))
    n_keep = max(0, min(n, n_keep))
    if n_keep == n:
        return np.arange(n, dtype=np.int64)
    keep = np.argpartition(-s, n_keep - 1)[:n_keep] if n_keep > 0 else np.empty(0, np.int64)
    return np.sort(keep.astype(np.int64))


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (T, d) arrays."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.sum(an * bn, axis=1)


def prune_video_ttm(
    video_tokens: np.ndarray,
    group_size: int = 4,
    prune_rate: float = 0.8,
) -> np.ndarray:
    """Temporal Token Merging over frame groups (Dycoke-style, Sec. 3.4).

    Within each group of ``group_size`` frames, the first frame is kept as an
    anchor; for every other frame, the ``prune_rate`` fraction of tokens most
    similar to the anchor (i.e. most temporally redundant) is dropped.

    For group_size=4, prune_rate=0.8 the retained fraction is
        (1 + 3*(1-0.8)) / 4 = 1.6 / 4 = 0.40,
    matching the paper's 40% video retention.

    Args:
        video_tokens: (n_frames, tokens_per_frame, d) float array.
        group_size: frames per TTM group (paper: 4).
        prune_rate: fraction of non-anchor tokens to drop (paper: 0.8).

    Returns:
        Sorted 1-D int array of *flattened* token indices to KEEP, where the
        flat index of frame f, token t is f * tokens_per_frame + t.
    """
    v = np.asarray(video_tokens, dtype=np.float64)
    if v.ndim != 3:
        raise ValueError("video_tokens must be (n_frames, tokens_per_frame, d)")
    n_frames, tpf, _ = v.shape
    keep_flat: list[int] = []

    for g0 in range(0, n_frames, group_size):
        group = range(g0, min(g0 + group_size, n_frames))
        anchor_f = g0
        # anchor frame: keep all tokens.
        keep_flat.extend(anchor_f * tpf + t for t in range(tpf))
        for f in group:
            if f == anchor_f:
                continue
            sim = _cosine(v[f], v[anchor_f])     # (tpf,) similarity to anchor
            n_keep = int(round(tpf * (1.0 - prune_rate)))
            n_keep = max(0, min(tpf, n_keep))
            if n_keep == 0:
                continue
            # keep the LEAST similar (least redundant) tokens.
            local_keep = np.argpartition(sim, n_keep - 1)[:n_keep]
            keep_flat.extend(int(f * tpf + t) for t in local_keep)

    return np.sort(np.asarray(keep_flat, dtype=np.int64))
