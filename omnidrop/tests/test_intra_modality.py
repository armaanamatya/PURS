"""Tests for intra-modality pre-LLM pruning (Sec. 3.4).

Trap (advisor): the 40% video retention is NOT a flat 20%-keep; it comes from
keeping a full anchor frame + 20% of each of the other 3 frames in a 4-frame
group: (1 + 3*0.2)/4 = 0.40. Combined with 70% audio over the paper's per-chunk
50 audio / 288 video, total ~= 45%.
"""
import numpy as np

from omnidrop.intra_modality import prune_audio_by_attention, prune_video_ttm


def test_audio_keeps_70_percent():
    saliency = np.arange(50, dtype=float)  # 50 audio tokens / chunk (App. A)
    keep = prune_audio_by_attention(saliency, prune_ratio=0.3)
    assert len(keep) == 35                 # 70% of 50
    # keeps the highest-saliency tokens.
    assert set(keep.tolist()) == set(range(15, 50))


def test_video_ttm_retains_40_percent():
    rng = np.random.default_rng(0)
    n_frames, tpf, d = 16, 288, 8          # 288 video tokens / chunk (App. A)
    video = rng.standard_normal((n_frames, tpf, d))
    keep = prune_video_ttm(video, group_size=4, prune_rate=0.8)
    total = n_frames * tpf
    ratio = len(keep) / total
    # (1 + 3*0.2)/4 = 0.40; rounding per-frame keep counts gives 0.401 for
    # tpf=288. Paper states "40%".
    assert abs(ratio - 0.40) < 0.01, ratio


def test_combined_total_retention_about_45_percent():
    # Per chunk: 50 audio @70% + 288 video @40%.
    audio_kept = 0.7 * 50
    video_kept = 0.4 * 288
    total_ratio = (audio_kept + video_kept) / (50 + 288)
    assert 0.43 < total_ratio < 0.46       # paper states ~45%


def test_anchor_frame_fully_kept():
    rng = np.random.default_rng(1)
    video = rng.standard_normal((4, 10, 4))
    keep = set(prune_video_ttm(video, group_size=4, prune_rate=0.8).tolist())
    # anchor = frame 0 -> flat indices 0..9 all present.
    assert set(range(0, 10)).issubset(keep)
