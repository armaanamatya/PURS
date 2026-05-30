"""Smoke tests for the three video-saliency variants.

Run with: pytest videozip/tests/ -v
Or: python -m videozip.tests.test_video_saliency
"""

from __future__ import annotations

import torch

import videozip  # bootstraps sys.path
from omnizip_units import omnizip_istm
from videozip.src._utils import _build_audio_groups, _map_retention_to_ratios
from videozip.src.istm_audio_anchored import omnizip_istm_audio_anchored
from videozip.src.video_saliency import (
    dispatch_video_saliency,
    omnizip_video_saliency,
    omnizip_video_saliency_l6,
    omnizip_video_saliency_simonly,
)
from videozip.src.videozip import (
    _project_importance_to_audio_scores,
    omnizip_videozip,
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


def test_l6_saliency_uses_1d_cached_scores_directly():
    scores = torch.arange(1, 9, dtype=torch.float32)
    out = omnizip_video_saliency_l6(
        scores,
        num_input_frames=4,
        video_token_per_frame=2,
        num_groups=2,
    )
    expected = [2.5 / 6.5, 1.0]
    assert all(abs(a - b) < 1e-6 for a, b in zip(out, expected))


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


def test_project_importance_from_full_attention_to_audio_scores():
    T = 7
    audio_indices = torch.tensor([1, 4, 6])
    attn = torch.zeros(2, T, T)
    attn[:, :, 1] = 2.0
    attn[:, :, 4] = 5.0
    attn[:, :, 6] = 7.0
    out = _project_importance_to_audio_scores(attn, audio_indices)
    expected = attn.mean(dim=0).sum(dim=0)[audio_indices]
    assert torch.equal(out, expected)


def test_project_importance_accepts_audio_local_1d_scores():
    audio_indices = torch.tensor([2, 5, 9])
    scores = torch.tensor([0.1, 0.8, 0.3])
    out = _project_importance_to_audio_scores(scores, audio_indices)
    assert torch.equal(out, scores)


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


def test_non_divisible_frame_count_does_not_crash():
    num_frames = 6
    tokens_per_frame = 4
    audio_tokens = 12
    hidden_dim = 8
    N_v = num_frames * tokens_per_frame
    N_a = audio_tokens
    L = N_v + N_a
    video_token_id = 151656
    audio_token_id = 151646
    g = torch.Generator().manual_seed(4)
    input_embeds = torch.randn(1, L, hidden_dim, generator=g)
    input_ids = torch.cat([
        torch.full((N_v,), video_token_id),
        torch.full((N_a,), audio_token_id),
    ]).unsqueeze(0)
    attn_logits = torch.randn(2, L, L, generator=g).softmax(dim=-1)

    out_embeds, mask = omnizip_videozip(
        input_embeds,
        attn_logits,
        input_ids,
        audio_token_id=audio_token_id,
        video_token_id=video_token_id,
        num_input_frames=num_frames,
        merging_ratio_audio=0.3,
        merging_ratio_v=0.6,
        contextual_ratio=0.05,
        g=3,
        video_saliency_source="attn",
    )
    assert out_embeds.shape == input_embeds.shape
    assert mask.shape == (L,)


def test_audio_anchor_beta_zero_matches_omnizip_istm():
    g = torch.Generator().manual_seed(9)
    video_feature = torch.randn(32, 16, generator=g)
    audio_feature = torch.randn(10, 16, generator=g)
    ratios = (0.25, 0.5)

    original = omnizip_istm(
        video_feature,
        num_tokens_per_frame=8,
        merging_ratio=ratios,
    )
    anchored = omnizip_istm_audio_anchored(
        video_feature,
        audio_feature=audio_feature,
        num_tokens_per_frame=8,
        merging_ratio=ratios,
        audio_anchor_beta=0.0,
    )
    assert torch.equal(anchored, original)


if __name__ == "__main__":
    test_attn_saliency_shape_and_range()
    test_l6_saliency_shape_and_range()
    test_l6_saliency_uses_1d_cached_scores_directly()
    test_l6_saliency_rejects_non_1d()
    test_simonly_saliency_shape_and_range()
    test_dispatch_routes_correctly()
    test_project_importance_from_full_attention_to_audio_scores()
    test_project_importance_accepts_audio_local_1d_scores()
    test_dispatch_rejects_missing_args()
    test_map_retention_preserves_mean()
    test_build_audio_groups_partitions()
    test_non_divisible_frame_count_does_not_crash()
    test_audio_anchor_beta_zero_matches_omnizip_istm()
    print("all smoke tests passed")
