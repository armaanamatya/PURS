"""Temporal Diversity Score (TDS) -- OmniDrop Section 3.3, Algorithm 1.

Query-guided importance alone can concentrate survival on a few high-attention
chunks. TDS boosts tokens that are temporally far from the key chunk so global
context survives. Algorithm 1, verbatim:

    Require: importance scores S, chunk indices C, budget k_l, weight lambda_div
    1: c_max  <- chunk_idx(argmax S)                # key chunk
    2: I_cand <- indices of the 2*k_l lowest scores in S
    3: D_i    <- |c_max - c_i| / max(C)             # normalized temporal distance
    4: S_TDS,i <- S_i + lambda_div * D_i            # score augmentation
    5: P      <- indices of the k_l lowest S_TDS among candidates
    6: return P                                     # tokens to prune
"""
from __future__ import annotations

import numpy as np


def tds_select_to_prune(
    S: np.ndarray,
    chunk_ids: np.ndarray,
    k: int,
    lambda_div: float = 0.2,
    max_chunk: int | None = None,
) -> np.ndarray:
    """Select which token positions to prune using TDS (Algorithm 1).

    Args:
        S: 1-D importance scores for the current AV tokens (Eq. 5 output).
        chunk_ids: 1-D chunk index of each AV token (same order as S).
        k: number of tokens to prune at this layer (k_l, Eq. 6).
        lambda_div: diversity weight (paper: 0.2).
        max_chunk: denominator max(C). Defaults to the max chunk id present;
                   clamped to >= 1 to avoid division by zero.

    Returns:
        1-D int array of local indices into S to PRUNE (length min(k, len(S))).
    """
    S = np.asarray(S, dtype=np.float64)
    chunk_ids = np.asarray(chunk_ids, dtype=np.int64)
    n = S.shape[0]
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)

    # line 1: key chunk = chunk of the highest-importance token.
    c_max = int(chunk_ids[int(np.argmax(S))])

    # line 2: candidate buffer = the 2*k lowest-importance tokens.
    n_cand = min(2 * k, n)
    # argpartition for the n_cand smallest, then stable order for determinism.
    cand = np.argpartition(S, n_cand - 1)[:n_cand]
    cand = cand[np.argsort(S[cand], kind="stable")]

    # line 3: normalized temporal distance for candidates.
    denom = max_chunk if max_chunk is not None else int(chunk_ids.max())
    denom = max(denom, 1)
    D = np.abs(c_max - chunk_ids[cand]) / denom

    # line 4: augmented score.
    S_tds = S[cand] + lambda_div * D

    # line 5: prune the k lowest augmented scores among candidates.
    order = np.argsort(S_tds, kind="stable")[:k]
    return cand[order]


def topk_lowest_to_prune(S: np.ndarray, k: int) -> np.ndarray:
    """Plain query-guided pruning (no diversity): drop the k lowest-S tokens.

    Used for layers before the TDS start layer (layer 14 for 7B / 19 for 3B,
    Sec. 4) where Algorithm 1 is not yet applied.
    """
    S = np.asarray(S, dtype=np.float64)
    n = S.shape[0]
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)
    cand = np.argpartition(S, k - 1)[:k]
    return cand[np.argsort(S[cand], kind="stable")]
