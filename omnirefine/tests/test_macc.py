"""MACC invariant tests: video quadtree, audio anchors, budget, retention bounds."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnirefine import (
    OmniRefineConfig,
    audio_merge_ratio,
    audio_retention,
    compress_audio_chunk,
    compress_video_chunk,
    semantic_intervals,
    video_retention,
)


def test_video_quadtree_merges_uniform_frame():
    """A perfectly uniform frame must collapse to a single representative."""
    cfg = OmniRefineConfig()
    D, H, W = 16, 4, 4
    base = np.random.default_rng(0).normal(size=D)
    grid = np.broadcast_to(base, (H, W, D)).copy()
    ids = np.arange(H * W).reshape(H, W)
    survivors = compress_video_chunk([grid], [ids], cfg)
    assert len(survivors) == 1, f"uniform frame -> {len(survivors)} nodes"
    print("video quadtree collapse ok")


def test_video_quadtree_splits_diverse_frame():
    """A frame with 4 distinct quadrants must NOT collapse to one node."""
    cfg = OmniRefineConfig()
    D, H, W = 16, 4, 4
    rng = np.random.default_rng(1)
    grid = np.empty((H, W, D))
    quads = rng.normal(size=(4, D)) * 5  # very different quadrants
    grid[:2, :2] = quads[0]; grid[:2, 2:] = quads[1]
    grid[2:, :2] = quads[2]; grid[2:, 2:] = quads[3]
    ids = np.arange(H * W).reshape(H, W)
    survivors = compress_video_chunk([grid], [ids], cfg)
    assert len(survivors) >= 4, f"diverse frame -> {len(survivors)} nodes"
    r = video_retention(survivors, H * W)
    print(f"video quadtree split ok: {len(survivors)} nodes, R_v={r:.2f}")


def test_budget_monotonic_and_bounded():
    """Eq 13/14: more video kept -> less audio merged (higher audio retention),
    and m_a stays within [a_min, a_max]."""
    cfg = OmniRefineConfig()
    r_low = audio_retention(0.2, cfg)
    r_high = audio_retention(0.5, cfg)
    assert r_high >= r_low  # higher R_v -> lower m_a -> higher R_a
    for rv in np.linspace(0, 1, 21):
        ma = audio_merge_ratio(rv, cfg)
        assert cfg.a_min - 1e-9 <= ma <= cfg.a_max + 1e-9
    print(f"budget ok: R_a(0.2)={r_low:.3f}, R_a(0.5)={r_high:.3f}")


def test_audio_semantic_intervals_and_retention():
    cfg = OmniRefineConfig()
    rng = np.random.default_rng(2)
    D = 16
    # two semantic groups with explicitly orthogonal centers (cos = 0 < 0.4)
    g = np.zeros((2, D))
    g[0, 0] = 1.0
    g[1, 1] = 1.0
    A = np.stack([g[0] + 0.02 * rng.normal(size=D) for _ in range(60)]
                 + [g[1] + 0.02 * rng.normal(size=D) for _ in range(60)])
    ivs = semantic_intervals(A, cfg.semantic_anchor_threshold)
    assert len(ivs) >= 2, "should detect >=2 semantic intervals"

    ids = list(range(120))
    sal = rng.random(120)
    target = 0.5
    res = compress_audio_chunk(A, ids, sal, retention=target, cfg=cfg)
    # retained count approximates the target ratio
    assert abs(len(res.keep_ids) / 120 - target) <= 0.1
    # every residual token is assigned to a kept anchor
    keep = set(res.keep_ids)
    for tok, anc in res.assignment.items():
        assert anc in keep
        assert tok not in keep
    print(f"audio ok: {len(ivs)} intervals, kept {len(res.keep_ids)}/120")


if __name__ == "__main__":
    test_video_quadtree_merges_uniform_frame()
    test_video_quadtree_splits_diverse_frame()
    test_budget_monotonic_and_bounded()
    test_audio_semantic_intervals_and_retention()
    print("ALL MACC TESTS PASSED")
