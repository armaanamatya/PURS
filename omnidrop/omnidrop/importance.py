"""Query-guided token importance (OmniDrop Section 3.2, Eq. 5).

The importance of an audiovisual token is how strongly the *text-query* tokens
attend to it, averaged over the query:

    Eq. 5:  S_j = (1 / n_T) * sum_{q in text-query} Attn_{q -> j}

where Attn is the post-softmax self-attention of the *current* decoder layer,
n_T is the number of text-query tokens, and j ranges over audiovisual tokens.

Design choices (flagged in REPRODUCTION_NOTES, paper does not pin them):
  * Heads are averaged (mean over heads) before scoring.
  * A single global ranking is produced over the union of audio + video tokens;
    this is what yields the paper's task-adaptive modality balance (Sec. 4.3) --
    no per-modality budget is imposed.
  * Text / system tokens are never scored as prune candidates.
"""
from __future__ import annotations

import numpy as np


def query_guided_importance(
    attn: np.ndarray,
    text_query_idx: np.ndarray,
    av_idx: np.ndarray,
) -> np.ndarray:
    """Compute the query-guided importance score S for each audiovisual token.

    Args:
        attn: post-softmax attention of the current layer. Accepts either
              (n_heads, seq, seq) or already head-averaged (seq, seq).
              ``attn[..., q, j]`` is the weight from query position q to key j.
        text_query_idx: 1-D int array of text-query token row positions.
        av_idx: 1-D int array of audiovisual token column positions to score.

    Returns:
        S: 1-D float array of length len(av_idx); S[i] is the importance of the
           token at column av_idx[i].  [Eq. 5]
    """
    attn = np.asarray(attn, dtype=np.float64)
    if attn.ndim == 3:           # (n_heads, seq, seq) -> mean over heads
        attn = attn.mean(axis=0)
    elif attn.ndim != 2:
        raise ValueError(f"attn must be 2-D or 3-D, got shape {attn.shape}")

    text_query_idx = np.asarray(text_query_idx, dtype=np.int64)
    av_idx = np.asarray(av_idx, dtype=np.int64)
    if text_query_idx.size == 0:
        raise ValueError("no text-query tokens: query guidance is undefined")

    # rows = text-query tokens, cols = AV tokens; mean over the n_T query rows.
    sub = attn[np.ix_(text_query_idx, av_idx)]   # (n_T, n_av)
    S = sub.mean(axis=0)                          # (n_av,)  -- the 1/n_T average
    return S
