"""Tests for the Temporal Diversity Score (Algorithm 1)."""
import numpy as np

from omnidrop.tds import tds_select_to_prune, topk_lowest_to_prune


def test_tds_spares_temporally_distant_low_score_token():
    # 6 tokens across chunks 0..5. The key chunk is where the max-S token lives.
    S = np.array([0.9, 0.10, 0.11, 0.12, 0.13, 0.05])
    chunk_ids = np.array([0, 1, 2, 3, 4, 5])
    # argmax S -> index 0 -> c_max = 0. The lowest raw score is idx5 (chunk 5),
    # but it is temporally farthest, so a strong diversity weight should spare it
    # relative to a near-chunk low-scorer.
    k = 1
    # large lambda so distance dominates among the 2k=2 candidates {idx5(0.05),
    # idx1(0.10)}: idx5 gets +1.0*lambda, idx1 gets +0.2*lambda.
    pruned_div = tds_select_to_prune(S, chunk_ids, k, lambda_div=1.0)
    assert pruned_div.tolist() == [1]   # idx1 (near key chunk) pruned, distant idx5 spared
    # with lambda=0 the raw lowest (idx5) is pruned.
    pruned_nodiv = tds_select_to_prune(S, chunk_ids, k, lambda_div=0.0)
    assert pruned_nodiv.tolist() == [5]


def test_candidate_buffer_is_2k():
    # With k=2 the candidate buffer is the 4 lowest; diversity only re-ranks
    # within that buffer, never promoting a high-S token to be pruned.
    S = np.array([0.01, 0.02, 0.03, 0.04, 0.90, 0.91])
    chunk_ids = np.array([0, 1, 2, 3, 4, 5])
    pruned = tds_select_to_prune(S, chunk_ids, k=2, lambda_div=10.0)
    assert set(pruned.tolist()).issubset({0, 1, 2, 3})
    assert len(pruned) == 2


def test_k_zero_and_k_ge_n():
    S = np.array([0.1, 0.2, 0.3])
    c = np.array([0, 1, 2])
    assert tds_select_to_prune(S, c, 0).size == 0
    assert sorted(tds_select_to_prune(S, c, 5).tolist()) == [0, 1, 2]


def test_topk_lowest_plain():
    S = np.array([0.5, 0.1, 0.9, 0.2])
    assert sorted(topk_lowest_to_prune(S, 2).tolist()) == [1, 3]


def test_single_chunk_no_div_by_zero():
    # all tokens in chunk 0 -> denom clamps to >=1, no crash.
    S = np.array([0.3, 0.1, 0.2])
    c = np.array([0, 0, 0])
    out = tds_select_to_prune(S, c, 1, lambda_div=0.2)
    assert out.tolist() == [1]
