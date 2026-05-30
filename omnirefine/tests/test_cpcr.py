"""CPCR invariant tests (no model required).

Checks the structural guarantees the paper states for the refined chunks:
monotonic, gap-free, non-overlapping, full coverage, and per-chunk sizes
inside [s_v_min, s_v_max] / [s_a_min, s_a_max].
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnirefine import OmniRefineConfig, correspondence_preserving_refinement
from omnirefine.cpcr import refine_chunks, similarity_field


def _synthetic(F=15, N=360, D=32, seed=0):
    """Build a block-structured audio/video sequence with 3 native chunks."""
    rng = np.random.default_rng(seed)
    # 3 latent "scenes"; frames and audio in the same scene look alike.
    scenes = rng.normal(size=(3, D))
    frame_scene = np.repeat([0, 1, 2], F // 3)
    audio_scene = np.repeat([0, 1, 2], N // 3)
    video_by_frame = [
        (scenes[s] + 0.05 * rng.normal(size=(20, D))) for s in frame_scene
    ]
    audio_hidden = np.stack(
        [scenes[s] + 0.05 * rng.normal(size=D) for s in audio_scene]
    )
    frame_native = frame_scene.copy()
    audio_native = audio_scene.copy()
    return video_by_frame, audio_hidden, frame_native, audio_native


def test_chunk_invariants():
    cfg = OmniRefineConfig()
    F, N = 15, 360
    vbf, ah, fn, an = _synthetic(F, N)
    chunks = correspondence_preserving_refinement(vbf, ah, fn, an, cfg, neighborhood_radius=2)

    assert len(chunks) >= 1
    # coverage + monotonic + gap-free + non-overlap
    assert chunks[0].frame_start == 0 and chunks[0].audio_start == 0
    assert chunks[-1].frame_end == F and chunks[-1].audio_end == N
    for a, b in zip(chunks, chunks[1:]):
        assert a.frame_end == b.frame_start, "frame gap/overlap"
        assert a.audio_end == b.audio_start, "audio gap/overlap"
    # size bounds
    for c in chunks:
        assert cfg.s_v_min <= c.n_frames <= cfg.s_v_max, (c.n_frames,)
        assert cfg.s_a_min <= c.n_audio <= cfg.s_a_max, (c.n_audio,)
    # totals
    assert sum(c.n_frames for c in chunks) == F
    assert sum(c.n_audio for c in chunks) == N
    print(f"CPCR ok: {len(chunks)} chunks, sizes "
          f"{[(c.n_frames, c.n_audio) for c in chunks]}")


def test_similarity_field_shapes():
    cfg = OmniRefineConfig()
    vbf, ah, fn, an = _synthetic()
    from omnirefine.cpcr import frame_embeddings, neighborhood_mask
    femb = frame_embeddings(vbf)
    nbr = neighborhood_mask(fn, an, radius=2)
    S_tilde, M = similarity_field(femb, ah, nbr)
    assert S_tilde.shape == (len(vbf), ah.shape[0])
    assert M.shape == S_tilde.shape
    assert np.all((M == 0) | (M == 1))
    # masked-out entries are zero
    assert np.all(S_tilde[M == 0] == 0)
    print("similarity field ok")


if __name__ == "__main__":
    test_similarity_field_shapes()
    test_chunk_invariants()
    print("ALL CPCR TESTS PASSED")
