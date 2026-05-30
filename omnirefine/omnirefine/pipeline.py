"""OmniRefine top-level pipeline.

Paper: Sec 3.2 "Overview" of arXiv:2605.12056v1.  A training-free two-stage
compressor applied to interleaved audio/video tokens *before* the LLM
prefill stage:

    CPCR  (Stage 1, cpcr.py)  -- refine native chunk boundaries into
            cross-modally aligned compression units.
    MACC  (Stage 2)           -- per refined chunk, cooperatively compress
            video (video_compress.py) and audio (audio_compress.py) with a
            cross-modal budget (budget.py).

The core is a *pure function*:

    compress(ProbeInputs, OmniRefineConfig) -> KeepMask

All model-specific work (running prefill to the probe layer L, reading
hidden states / attention, building the native chunk indexing, then pruning
the KV cache and continuing prefill) lives behind ``adapter.py``.  This file
has no torch / model dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .audio_compress import AudioResult, compress_audio_chunk
from .budget import audio_retention, audio_merge_ratio
from .config import OmniRefineConfig
from .cpcr import RefinedChunk, correspondence_preserving_refinement
from .utils import as_numpy
from .video_compress import (
    VideoNode,
    compress_video_chunk,
    video_retention,
)


@dataclass
class ProbeInputs:
    """Everything the algorithm needs, extracted at probe layer L by an adapter.

    All hidden states are at the same probe layer (cfg.layer_probe).
    """

    # Video, laid out per frame as (H, W, D) hidden grids + matching global ids.
    video_grid_hidden: List[np.ndarray]          # len F, each (H, W, D)
    video_grid_ids: List[np.ndarray]             # len F, each (H, W) int
    frame_native_chunk: np.ndarray               # (F,) native chunk index per frame

    # Audio as a flat sequence.
    audio_hidden: np.ndarray                     # (N, D)
    audio_global_ids: List[int]                  # (N,)
    audio_native_chunk: np.ndarray               # (N,) native chunk index per token
    audio_saliency: np.ndarray                   # (N,) fused attention importance

    # Optional video saliency (token_id -> score) for bound clamping.
    video_saliency: Optional[Dict[int, float]] = None


@dataclass
class KeepMask:
    """Result of compression: which tokens survive, and their new reps."""

    video_keep_ids: List[int] = field(default_factory=list)
    audio_keep_ids: List[int] = field(default_factory=list)
    video_reps: Dict[int, np.ndarray] = field(default_factory=dict)
    audio_reps: Dict[int, np.ndarray] = field(default_factory=dict)
    audio_assignment: Dict[int, int] = field(default_factory=dict)
    chunks: List[RefinedChunk] = field(default_factory=list)
    # Per-chunk diagnostics.
    chunk_video_retention: List[float] = field(default_factory=list)
    chunk_audio_retention: List[float] = field(default_factory=list)

    @property
    def n_video_kept(self) -> int:
        return len(self.video_keep_ids)

    @property
    def n_audio_kept(self) -> int:
        return len(self.audio_keep_ids)


def _enforce_video_bounds(
    survivors: List[VideoNode],
    chunk_token_ids: List[int],
    chunk_token_hidden: np.ndarray,
    cfg: OmniRefineConfig,
    video_saliency: Optional[Dict[int, float]],
) -> List[VideoNode]:
    """Clamp observed video retention into [v_min, v_max] (Appendix A).

    Threshold-driven quadtree/temporal merging yields an observed R_v.  The
    paper's hard bounds prevent catastrophic semantic loss (too few tokens)
    and insufficient compression (too many).  If R_v exceeds v_max we drop
    the lowest-saliency nodes; if it is below v_min we add back the
    highest-saliency *individual* tokens that the merge had discarded, as
    singleton nodes.  Re-ranking is by ``video_saliency`` when available, else
    by region size / token order (PARTIAL -- the paper fixes the bounds but
    not the exact re-ranking; see REPRODUCTION_NOTES).
    """
    n_total = len(chunk_token_ids)
    if n_total == 0:
        return survivors
    lo = int(np.ceil(cfg.v_min * n_total))
    hi = max(int(np.floor(cfg.v_max * n_total)), lo)

    def node_score(node: VideoNode) -> float:
        if video_saliency is None:
            return node.weight
        return max((video_saliency.get(tid, 0.0) for tid in node.token_ids), default=0.0)

    if len(survivors) > hi:
        survivors = sorted(survivors, key=node_score, reverse=True)[:hi]
    elif len(survivors) < lo:
        # Only the per-node anchor token is actually retained; every other
        # covered token was pruned (folded into the merged rep) and is a
        # valid add-back candidate.
        kept = {int(n.anchor_token) for n in survivors}
        pool = [(tid, h) for tid, h in zip(chunk_token_ids, chunk_token_hidden)
                if tid not in kept]
        if video_saliency is not None:
            pool.sort(key=lambda x: video_saliency.get(x[0], 0.0), reverse=True)
        for tid, h in pool[: lo - len(survivors)]:
            survivors.append(VideoNode(frame=-1, token_ids=[int(tid)],
                                       rep=np.asarray(h, dtype=np.float64), weight=1.0))
    return survivors


def compress(inputs: ProbeInputs, cfg: OmniRefineConfig) -> KeepMask:
    """Run the full two-stage OmniRefine compression. Pure, model-free."""
    # ---------- Stage 1: CPCR ----------
    video_hidden_by_frame = [
        as_numpy(g).reshape(-1, as_numpy(g).shape[-1]) for g in inputs.video_grid_hidden
    ]
    chunks = correspondence_preserving_refinement(
        video_hidden_by_frame=video_hidden_by_frame,
        audio_hidden=inputs.audio_hidden,
        frame_native_chunk=inputs.frame_native_chunk,
        audio_native_chunk=inputs.audio_native_chunk,
        cfg=cfg,
    )

    A = as_numpy(inputs.audio_hidden)
    sal = as_numpy(inputs.audio_saliency).ravel()

    out = KeepMask(chunks=chunks)

    # ---------- Stage 2: MACC, per refined chunk ----------
    for ch in chunks:
        # ----- Video branch (Eq 6, 7, 11) -----
        grid_h = inputs.video_grid_hidden[ch.frame_start:ch.frame_end]
        grid_i = inputs.video_grid_ids[ch.frame_start:ch.frame_end]
        flat_ids = [int(t) for g in grid_i for t in np.asarray(g).reshape(-1)]
        flat_hidden = np.concatenate(
            [as_numpy(g).reshape(-1, as_numpy(g).shape[-1]) for g in grid_h], axis=0
        )
        n_vid_tokens = len(flat_ids)
        survivors = compress_video_chunk(grid_h, grid_i, cfg)
        survivors = _enforce_video_bounds(
            survivors, flat_ids, flat_hidden, cfg, inputs.video_saliency
        )
        r_v = video_retention(survivors, n_vid_tokens)
        for node in survivors:
            out.video_keep_ids.append(int(node.anchor_token))
            out.video_reps[int(node.anchor_token)] = node.rep
        video_reps_arr = (
            np.stack([n.rep for n in survivors], axis=0) if survivors else None
        )

        # ----- Cross-modal budget (Eq 13, 14) -----
        r_a = audio_retention(r_v, cfg)

        # ----- Audio branch (Eq 9, 11) -----
        a_hidden = A[ch.audio_start:ch.audio_end]
        a_ids = inputs.audio_global_ids[ch.audio_start:ch.audio_end]
        a_sal = sal[ch.audio_start:ch.audio_end]
        ares: AudioResult = compress_audio_chunk(
            audio_hidden=a_hidden,
            audio_global_ids=a_ids,
            audio_saliency=a_sal,
            retention=r_a,
            cfg=cfg,
            video_reps=video_reps_arr,
        )
        out.audio_keep_ids.extend(ares.keep_ids)
        out.audio_reps.update(ares.merged_reps)
        out.audio_assignment.update(ares.assignment)

        out.chunk_video_retention.append(r_v)
        out.chunk_audio_retention.append(r_a)

    return out
