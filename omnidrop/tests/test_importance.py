"""Tests for query-guided importance (Eq. 5)."""
import numpy as np

from omnidrop.importance import query_guided_importance


def test_importance_is_mean_text_attention_to_av():
    # seq = [t0, t1 (text-query), a0, a1, v0 (av)]
    seq = 5
    attn = np.zeros((seq, seq))
    text_idx = np.array([0, 1])
    av_idx = np.array([2, 3, 4])
    # text rows attend: to col2 -> [0.1, 0.3]; col3 -> [0.5, 0.5]; col4 -> [0.9, 0.1]
    attn[0, 2], attn[1, 2] = 0.1, 0.3
    attn[0, 3], attn[1, 3] = 0.5, 0.5
    attn[0, 4], attn[1, 4] = 0.9, 0.1
    S = query_guided_importance(attn, text_idx, av_idx)
    np.testing.assert_allclose(S, [0.2, 0.5, 0.5])


def test_head_averaging():
    # two heads -> mean over heads before scoring.
    attn = np.zeros((2, 4, 4))
    text_idx = np.array([0])
    av_idx = np.array([1, 2, 3])
    attn[0, 0, 1:] = [0.2, 0.4, 0.6]
    attn[1, 0, 1:] = [0.4, 0.4, 0.4]
    S = query_guided_importance(attn, text_idx, av_idx)
    np.testing.assert_allclose(S, [0.3, 0.4, 0.5])


def test_empty_text_query_raises():
    attn = np.zeros((3, 3))
    try:
        query_guided_importance(attn, np.array([], dtype=int), np.array([1, 2]))
        assert False
    except ValueError:
        pass
