"""End-to-end pipeline test on synthetic ProbeInputs (no model)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnirefine import OmniRefineConfig, ProbeInputs, compress


def build_inputs(F=15, N=360, H=4, W=4, D=16, seed=0):
    rng = np.random.default_rng(seed)
    scenes = rng.normal(size=(3, D))
    frame_scene = np.repeat([0, 1, 2], F // 3)
    audio_scene = np.repeat([0, 1, 2], N // 3)

    video_grid_hidden, video_grid_ids = [], []
    tok = 0
    for s in frame_scene:
        grid = scenes[s] + 0.05 * rng.normal(size=(H, W, D))
        ids = np.arange(tok, tok + H * W).reshape(H, W)
        tok += H * W
        video_grid_hidden.append(grid)
        video_grid_ids.append(ids)

    audio_hidden = np.stack([scenes[s] + 0.05 * rng.normal(size=D) for s in audio_scene])
    audio_ids = list(range(10_000, 10_000 + N))
    audio_sal = rng.random(N)

    return ProbeInputs(
        video_grid_hidden=video_grid_hidden,
        video_grid_ids=video_grid_ids,
        frame_native_chunk=frame_scene.copy(),
        audio_hidden=audio_hidden,
        audio_global_ids=audio_ids,
        audio_native_chunk=audio_scene.copy(),
        audio_saliency=audio_sal,
    )


def test_end_to_end():
    cfg = OmniRefineConfig()
    inp = build_inputs()
    keep = compress(inp, cfg)

    # produced refined chunks
    assert len(keep.chunks) >= 1
    # kept ids are unique and reference real tokens
    assert len(keep.video_keep_ids) == len(set(keep.video_keep_ids))
    assert len(keep.audio_keep_ids) == len(set(keep.audio_keep_ids))
    # compression actually happened (fewer than input tokens)
    n_video_in = sum(g.size // g.shape[-1] for g in inp.video_grid_hidden)
    assert keep.n_video_kept < n_video_in
    assert keep.n_audio_kept < inp.audio_hidden.shape[0]

    # per-chunk video retention respects BOTH hard bounds (Appendix A)
    for rv in keep.chunk_video_retention:
        assert cfg.v_min - 1e-6 <= rv <= cfg.v_max + 1e-6, rv
    # audio retention is the budget-coupled value, bounded by 1 - a_min
    for ra in keep.chunk_audio_retention:
        assert 0.0 < ra <= 1.0 - cfg.a_min + 1e-6

    print(
        f"pipeline ok: {len(keep.chunks)} chunks | "
        f"video {keep.n_video_kept}/{n_video_in} | "
        f"audio {keep.n_audio_kept}/{inp.audio_hidden.shape[0]} | "
        f"R_v={[round(x,2) for x in keep.chunk_video_retention]} "
        f"R_a={[round(x,2) for x in keep.chunk_audio_retention]}"
    )


if __name__ == "__main__":
    test_end_to_end()
    print("ALL PIPELINE TESTS PASSED")
