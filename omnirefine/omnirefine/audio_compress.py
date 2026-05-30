"""Stage 2b -- Semantic-Anchor Audio Compression (SAAC / Compress_a).

Paper: Sec 3.4 "Semantic-Anchor Audio Compression" of arXiv:2605.12056v1.

Pipeline for a refined chunk g with audio tokens A^(g) = {a_t}:
  1. Semantic anchors: scan adjacent tokens; whenever
     cos(a_t, a_{t-1}) < threshold, mark a_t as an anchor boundary,
     partitioning the chunk into local semantic intervals (Sec 3.4).
  2. Retain a set of *dominant* audio tokens by fused attention-based
     importance, plus a small number of *contextual* anchors from the rest
     (contextual_ratio).  H^(g) = retained anchors.
  3. (9) Assign each residual non-anchor token to its most similar retained
     anchor *within its semantic interval*:
         pi(t) = argmax_{h in H^(g)} cos(a_t, a_h).
  4. Within each anchor group select merge candidates by cross-modal
     matching with the retained video reps, then (11) fuse:
         a~_h = ( a_h + sum_{t in M(h)} w_t a_t ) / ( 1 + sum_{t in M(h)} w_t )
     where w_t normalizes the relevance scores inside M(h).

Budget: the number retained follows R_a from the cross-modal budget
(budget.py, Eq 13/14).

PARTIAL items (see REPRODUCTION_NOTES): the exact "fused attention-based
importance" definition, the dominant-vs-contextual split, and the relevance
score behind w_t.  We expose ``audio_saliency`` as an explicit input and use
cross-modal cosine for relevance, both documented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import OmniRefineConfig
from .utils import as_numpy, cosine, l2_normalize


@dataclass
class AudioResult:
    keep_ids: List[int]                      # global indices of retained anchors
    merged_reps: Dict[int, np.ndarray]       # anchor id -> fused representation a~_h
    assignment: Dict[int, int]               # residual token id -> anchor id (pi)
    intervals: List[List[int]] = field(default_factory=list)


def semantic_intervals(
    audio_hidden: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    """Partition consecutive tokens into semantic intervals (Sec 3.4).

    A new interval starts at token t whenever cos(a_t, a_{t-1}) < threshold.
    Returns a list of intervals, each a list of *local* token positions.
    """
    A = as_numpy(audio_hidden)
    n = A.shape[0]
    if n == 0:
        return []
    intervals: List[List[int]] = [[0]]
    for t in range(1, n):
        if cosine(A[t], A[t - 1]) < threshold:
            intervals.append([t])
        else:
            intervals[-1].append(t)
    return intervals


def compress_audio_chunk(
    audio_hidden: np.ndarray,
    audio_global_ids: List[int],
    audio_saliency: np.ndarray,
    retention: float,
    cfg: OmniRefineConfig,
    video_reps: Optional[np.ndarray] = None,
) -> AudioResult:
    """Compress_a for one refined chunk (Eq 9, 11).

    Parameters
    ----------
    audio_hidden : (N_g, D) hidden states of the chunk's audio tokens.
    audio_global_ids : matching global token indices.
    audio_saliency : (N_g,) fused attention-based importance per token.
    retention : target audio retention R_a (from budget.audio_retention).
    video_reps : (K_v, D) retained video representatives for cross-modal
        guidance (relevance / merge weighting).  May be None.
    """
    A = as_numpy(audio_hidden)
    n = A.shape[0]
    ids = list(audio_global_ids)
    if n == 0:
        return AudioResult(keep_ids=[], merged_reps={}, assignment={}, intervals=[])

    intervals = semantic_intervals(A, cfg.semantic_anchor_threshold)

    # ---- Budget: how many anchors to retain (Eq 13/14 give `retention`) ----
    k_total = int(round(retention * n))
    k_total = max(1, min(n, k_total))
    n_ctx = int(round(cfg.contextual_ratio * n))
    n_dom = max(1, k_total - n_ctx)

    order = np.argsort(-audio_saliency)  # descending saliency (dominant first)
    dominant = list(order[:n_dom])
    remaining = list(order[n_dom:])
    # Contextual anchors: spread across remaining (here: next-highest saliency,
    # but at least one per under-represented interval would also be valid --
    # PARTIAL, documented).
    contextual = list(remaining[:n_ctx])
    anchor_local = sorted(set(dominant) | set(contextual))
    anchor_set = set(anchor_local)

    # ---- Eq 9: assign each residual token to nearest anchor in its interval ----
    # Pre-index which interval each local position belongs to.
    interval_of = {}
    for gi, iv in enumerate(intervals):
        for p in iv:
            interval_of[p] = gi

    assignment: Dict[int, int] = {}
    merge_sets: Dict[int, List[int]] = {a: [] for a in anchor_local}
    A_norm = l2_normalize(A)
    for t in range(n):
        if t in anchor_set:
            continue
        iv = intervals[interval_of[t]]
        local_anchors = [a for a in iv if a in anchor_set]
        if not local_anchors:
            # interval has no anchor: fall back to nearest global anchor
            local_anchors = anchor_local
        sims = A_norm[local_anchors] @ A_norm[t]
        h = local_anchors[int(np.argmax(sims))]   # pi(t)  (Eq 9)
        assignment[ids[t]] = ids[h]
        merge_sets[h].append(t)

    # ---- Eq 11: similarity-weighted fusion of merged tokens into anchors ----
    merged_reps: Dict[int, np.ndarray] = {}
    for h in anchor_local:
        members = merge_sets[h]
        if not members:
            merged_reps[ids[h]] = A[h]
            continue
        # relevance score behind w_t: cross-modal cosine to best video rep if
        # available, else audio cosine to the anchor (documented fallback).
        if video_reps is not None and len(video_reps) > 0:
            Vn = l2_normalize(as_numpy(video_reps))
            rel = np.max(A_norm[members] @ Vn.T, axis=1)
        else:
            rel = A_norm[members] @ A_norm[h]
        rel = np.clip(rel, 1e-6, None)
        w = rel / rel.sum()                       # normalized relevance -> w_t
        fused = A[h] + (w[:, None] * A[members]).sum(axis=0)
        merged_reps[ids[h]] = fused / (1.0 + w.sum())

    keep_ids = [ids[a] for a in anchor_local]
    return AudioResult(
        keep_ids=keep_ids,
        merged_reps=merged_reps,
        assignment=assignment,
        intervals=intervals,
    )
