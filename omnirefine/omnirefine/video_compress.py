"""Stage 2a -- Tree-Structured Spatio-Temporal Video Compression (Compress_v).

Paper: Sec 3.4 "Tree-Structured Spatio-Temporal Compression" of
arXiv:2605.12056v1.

Spatial (within each frame), Eq (6):
  Reorganize a chunk's video tokens into frame-wise 2D grids using the
  original spatial position encoding.  Build a multi-scale spatial
  hierarchy where each parent node owns 2x2 child regions.  Top-down: if
  *all* children are sufficiently similar to the parent,
        cos( z(R), z(R_c) ) >= tau_s  for all R_c in C(R),
  the parent is kept as a single representative token (children pruned);
  otherwise recurse into the children.

Temporal (across consecutive frames), Eq (7):
  Merge surviving nodes n_i^(t-1), n_j^(t) when
        cos( z(n_i^(t-1)), z(n_j^(t)) ) >= tau_t.
  The merged representative is a weighted average; only surviving nodes are
  mapped back to the visual token mask.

z(.) is mean-pooled hidden states of the region (Eq 12).

Retained set:  K_v^(g) = Compress_v( V^(g); tau_s, tau_t )  (Eq 11 video).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import OmniRefineConfig
from .utils import as_numpy, cosine, mean_pool


@dataclass
class VideoNode:
    """A surviving spatio-temporal region after compression."""

    frame: int                    # frame index within the chunk
    token_ids: List[int]          # original (global) video-token indices it covers
    rep: np.ndarray               # z(.) representative (mean-pooled hidden)
    weight: float = 1.0           # token-count weight for temporal merging
    anchor_token: int = field(default=-1)  # the single token kept for this node

    def __post_init__(self) -> None:
        if self.anchor_token < 0 and self.token_ids:
            self.anchor_token = self.token_ids[0]


def _spatial_quadtree(
    grid_hidden: np.ndarray,
    grid_ids: np.ndarray,
    tau_s: float,
) -> List[VideoNode]:
    """Coarse-to-fine 2x2 quadtree on one frame's (H, W, D) grid (Eq 6).

    ``grid_ids`` is the (H, W) array of global token indices.  Returns the
    surviving spatial nodes for the frame.
    """
    H, W, D = grid_hidden.shape
    nodes: List[VideoNode] = []

    def rec(r0: int, r1: int, c0: int, c1: int) -> None:
        region = grid_hidden[r0:r1, c0:c1].reshape(-1, D)
        ids = grid_ids[r0:r1, c0:c1].reshape(-1).tolist()
        z_R = region.mean(axis=0)
        # Leaf: a single token.
        if (r1 - r0) <= 1 and (c1 - c0) <= 1:
            nodes.append(VideoNode(frame=-1, token_ids=ids, rep=z_R, weight=float(len(ids))))
            return
        rm = (r0 + r1 + 1) // 2 if (r1 - r0) > 1 else r1
        cm = (c0 + c1 + 1) // 2 if (c1 - c0) > 1 else c1
        quadrants = [
            (r0, rm, c0, cm),
            (r0, rm, cm, c1),
            (rm, r1, c0, cm),
            (rm, r1, cm, c1),
        ]
        children = [(a, b, c, d) for (a, b, c, d) in quadrants if a < b and c < d]
        # children that actually differ from this region (dedupe degenerate split)
        children = [q for q in children if not (q == (r0, r1, c0, c1))]
        if not children:
            nodes.append(VideoNode(frame=-1, token_ids=ids, rep=z_R, weight=float(len(ids))))
            return
        all_similar = True
        for (a, b, c, d) in children:
            z_c = grid_hidden[a:b, c:d].reshape(-1, D).mean(axis=0)
            if cosine(z_R, z_c) < tau_s:
                all_similar = False
                break
        if all_similar:
            # Keep parent as a single representative token.
            nodes.append(VideoNode(frame=-1, token_ids=ids, rep=z_R, weight=float(len(ids))))
        else:
            for (a, b, c, d) in children:
                rec(a, b, c, d)

    rec(0, H, 0, W)
    return nodes


def _temporal_merge(
    frame_nodes: List[List[VideoNode]],
    tau_t: float,
) -> List[VideoNode]:
    """Greedy temporal merge across consecutive frames (Eq 7).

    Each node in frame t is matched to its most-similar surviving node in
    frame t-1; if cosine >= tau_t it is merged into that earlier node
    (weighted-average representative) and does not survive on its own.
    """
    survivors: List[VideoNode] = []
    prev_layer: List[VideoNode] = []
    for t, layer in enumerate(frame_nodes):
        for n in layer:
            n.frame = t
        if not prev_layer:
            survivors.extend(layer)
            prev_layer = list(layer)
            continue
        new_prev: List[VideoNode] = []
        prev_reps = np.stack([p.rep for p in prev_layer], axis=0)
        for n in layer:
            sims = prev_reps @ n.rep / (
                np.linalg.norm(prev_reps, axis=1) * np.linalg.norm(n.rep) + 1e-8
            )
            best = int(np.argmax(sims))
            if sims[best] >= tau_t:
                # merge n into prev_layer[best] (weighted average, Eq 7/11-style)
                p = prev_layer[best]
                w = p.weight + n.weight
                p.rep = (p.weight * p.rep + n.weight * n.rep) / w
                p.weight = w
                p.token_ids = p.token_ids + n.token_ids
            else:
                survivors.append(n)
                new_prev.append(n)
        # carry forward this frame's *surviving* nodes plus untouched prev anchors
        prev_layer = new_prev if new_prev else prev_layer
    return survivors


def compress_video_chunk(
    grid_hidden_by_frame: List[np.ndarray],
    grid_ids_by_frame: List[np.ndarray],
    cfg: OmniRefineConfig,
    video_saliency: Optional[dict] = None,
) -> List[VideoNode]:
    """Compress_v for one refined chunk (Eq 6, 7, 11).

    ``grid_hidden_by_frame[k]`` is the (H, W, D) hidden-state grid of the
    chunk's k-th frame; ``grid_ids_by_frame[k]`` the matching (H, W) global
    token indices.  Returns surviving :class:`VideoNode` s.

    ``video_saliency`` (optional {token_id: score}) is used only by
    :func:`enforce_video_bounds` to break ties when clamping retention into
    [v_min, v_max].  PARTIAL in the paper -- see REPRODUCTION_NOTES.
    """
    frame_nodes = [
        _spatial_quadtree(as_numpy(h), np.asarray(ids), cfg.tau_s)
        for h, ids in zip(grid_hidden_by_frame, grid_ids_by_frame)
    ]
    survivors = _temporal_merge(frame_nodes, cfg.tau_t)
    return survivors


def video_retention(survivors: List[VideoNode], n_total: int) -> float:
    """Observed video retention ratio R_v = |kept anchors| / |tokens|."""
    return len(survivors) / max(1, n_total)
