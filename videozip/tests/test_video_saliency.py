"""Smoke tests for the three video-saliency variants.

Run with: pytest videozip/tests/ -v
Or: python -m videozip.tests.test_video_saliency
"""

from __future__ import annotations

import torch

import videozip  # bootstraps sys.path
from videozip.src._utils import _build_audio_groups, _map_retention_to_ratios
from videozip.src.video_saliency import (
    dispatch_video_saliency,
    omnizip_video_saliency,
    omnizip_video_saliency_l6,
    omnizip_video_saliency_simonly,
)


def _make_fake_inputs(num_frames=8, tokens_per_frame=16, audio_tokens=64, hidden_dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    N_v = num_frames * tokens_per_frame
    N_a = audio_tokens
    T = N_v + N_a
    flat_embeds = torch.randn(T, hidden_dim, generator=g)
    video_indices = torch.arange(0, N_v)
    audio_indices = torch.arange(N_v, T)
    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]
    attn_logits = torch.randn(4, T, T, generator=g).softmax(dim=-1)
    # Cache stores 1D Q*K saliency vectors of length N_v, not hidden states.
    l6_video_scores = torch.rand(N_v, generator=g)
    return {
        "video_feature": video_feature,
        "audio_feature": audio_feature,
        "video_indices": video_indices,
        "attn_logits": attn_logits,
        "l6_video_scores": l6_video_scores,
        "num_frames": num_frames,
        "tokens_per_frame": tokens_per_frame,
    }


def test_attn_saliency_shape_and_range():
    fx = _make_fake_inputs()
    out = omnizip_video_saliency(
        fx["video_feature"], fx["video_indices"], fx["attn_logits"],
        num_input_frames=fx["num_frames"],
        video_token_per_frame=fx["tokens_per_frame"],
        num_groups=2,
    )
    assert len(out) == 2
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in out)
    assert max(out) > 0.99  # normalization makes the max group hit 1


def test_l6_saliency_shape_and_range():
    fx = _make_fake_inputs()
    out = omnizip_video_saliency_l6(
        fx["l6_video_scores"],
        num_input_frames=fx["num_frames"],
        video_token_per_frame=fx["tokens_per_frame"],
        num_groups=4,
    )
    assert len(out) == 4
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in out)


def test_l6_saliency_rejects_non_1d():
    fx = _make_fake_inputs()
    bad = torch.zeros(8, 16)  # 2D — should raise
    try:
        omnizip_video_saliency_l6(
            bad,
            num_input_frames=fx["num_frames"],
            video_token_per_frame=fx["tokens_per_frame"],
            num_groups=2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("2D scores should raise ValueError")


def test_simonly_saliency_shape_and_range():
    fx = _make_fake_inputs()
    out = omnizip_video_saliency_simonly(
        fx["video_feature"], fx["audio_feature"],
        num_input_frames=fx["num_frames"],
        video_token_per_frame=fx["tokens_per_frame"],
        num_groups=2,
    )
    assert len(out) == 2
    assert all(0.0 <= v <= 1.0 + 1e-6 for v in out)


def test_dispatch_routes_correctly():
    fx = _make_fake_inputs()
    common = dict(
        video_feature=fx["video_feature"],
        video_indices=fx["video_indices"],
        num_input_frames=fx["num_frames"],
        video_token_per_frame=fx["tokens_per_frame"],
        num_groups=2,
    )
    a = dispatch_video_saliency("attn", attn_logits=fx["attn_logits"], **common)
    b = dispatch_video_saliency("l6_cached", l6_video_scores=fx["l6_video_scores"], **common)
    c = dispatch_video_saliency("sim_only", audio_feature=fx["audio_feature"], **common)
    assert len(a) == len(b) == len(c) == 2


def test_dispatch_rejects_missing_args():
    fx = _make_fake_inputs()
    common = dict(
        video_feature=fx["video_feature"],
        video_indices=fx["video_indices"],
        num_input_frames=fx["num_frames"],
        video_token_per_frame=fx["tokens_per_frame"],
        num_groups=2,
    )
    try:
        dispatch_video_saliency("attn", **common)
    except ValueError:
        pass
    else:
        raise AssertionError("attn without attn_logits should raise")


def test_map_retention_preserves_mean():
    target = 0.5
    ratios = _map_retention_to_ratios([0.1, 0.4, 0.7, 1.0], target_mean_ratio=target)
    assert len(ratios) == 4
    assert abs(sum(ratios) / len(ratios) - target) < 1e-3


def test_build_audio_groups_partitions():
    groups = _build_audio_groups(100, 4)
    assert groups[0] == (0, 25)
    assert groups[-1][1] == 100
    flat = [(s, e) for (s, e) in groups]
    for (s, e), (sn, en) in zip(flat, flat[1:]):
        assert e == sn


if __name__ == "__main__":
    test_attn_saliency_shape_and_range()
    test_l6_saliency_shape_and_range()
    test_l6_saliency_rejects_non_1d()
    test_simonly_saliency_shape_and_range()
    test_dispatch_routes_correctly()
    test_dispatch_rejects_missing_args()
    test_map_retention_preserves_mean()
    test_build_audio_groups_partitions()
    print("all smoke tests passed")
