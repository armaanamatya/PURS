"""PyTorch prefill bridge tests on synthetic Qwen-style tensors."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnirefine import OmniRefineConfig, omnirefine_qwen_prefill


def _synthetic_qwen_prefill(F=15, N=360, H=4, W=4, D=16, seed=0):
    rng = np.random.default_rng(seed)
    scenes = rng.normal(size=(3, D)).astype(np.float32)
    frame_scene = np.repeat([0, 1, 2], F // 3)
    audio_scene = np.repeat([0, 1, 2], N // 3)

    video = []
    for s in frame_scene:
        frame = scenes[s] + 0.05 * rng.normal(size=(H, W, D)).astype(np.float32)
        video.append(frame.reshape(-1, D))
    video = np.concatenate(video, axis=0)
    audio = np.stack(
        [scenes[s] + 0.05 * rng.normal(size=D).astype(np.float32) for s in audio_scene]
    )

    video_id = 200001
    audio_id = 200002
    text = rng.normal(size=(3, D)).astype(np.float32)
    embeds = np.concatenate([text[:1], video, text[1:2], audio, text[2:]], axis=0)
    ids = np.concatenate([
        np.array([11]),
        np.full(video.shape[0], video_id),
        np.array([12]),
        np.full(audio.shape[0], audio_id),
        np.array([13]),
    ])
    frame_native = frame_scene.copy()
    audio_native = audio_scene.copy()
    audio_saliency = rng.random(N)

    return (
        torch.tensor(embeds).unsqueeze(0),
        torch.tensor(ids).unsqueeze(0),
        video_id,
        audio_id,
        frame_native,
        audio_native,
        audio_saliency,
    )


def test_qwen_prefill_bridge_returns_compressed_mask():
    cfg = OmniRefineConfig(layer_probe=0)
    (
        embeds,
        ids,
        video_id,
        audio_id,
        frame_native,
        audio_native,
        audio_saliency,
    ) = _synthetic_qwen_prefill()

    compressed, mask, diag = omnirefine_qwen_prefill(
        embeds,
        ids,
        audio_token_id=audio_id,
        video_token_id=video_id,
        num_input_frames=15,
        cfg=cfg,
        video_grid_hw=(4, 4),
        frame_native_chunk=frame_native,
        audio_native_chunk=audio_native,
        audio_saliency=audio_saliency,
    )

    assert compressed.shape[0] == 1
    assert compressed.shape[1] == int(mask.sum().item())
    assert compressed.shape[1] < embeds.shape[1]
    assert diag.keep is not None
    assert diag.video_kept < diag.video_tokens
    assert diag.audio_kept < diag.audio_tokens
    assert mask[0].item() and mask[-1].item()


def test_qwen_prefill_bridge_short_sequence_identity():
    cfg = OmniRefineConfig(layer_probe=0)
    embeds = torch.randn(1, 8, 4)
    ids = torch.tensor([[1, 9, 9, 2, 10, 10, 10, 3]])

    compressed, mask, diag = omnirefine_qwen_prefill(
        embeds,
        ids,
        audio_token_id=10,
        video_token_id=9,
        num_input_frames=2,
        cfg=cfg,
        video_grid_hw=(1, 1),
    )

    assert compressed.shape == embeds.shape
    assert bool(mask.all().item())
    assert diag.keep is None
    assert diag.warnings
