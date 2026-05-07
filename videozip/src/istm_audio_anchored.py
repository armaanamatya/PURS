"""Audio-anchored ISTM. §4c of videozip_plan.md.

Forks OmniZip's `omnizip_istm` and adds an audio-similarity term to the dpcknn
diversity score. This is the "audio guides video anchor selection" direction that
makes VideoZip bidirectional — neither OmniZip nor OmniSIFT have it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


def dpcknn_audio_guided(
    tokens: torch.Tensor,
    audio: Optional[torch.Tensor],
    keep_rate: float = 0.5,
    k: int = 5,
    beta: float = 0.3,
) -> torch.Tensor:
    """DPC-KNN with an additive audio-similarity term.

    score = -diversity_score + beta * max_sim(v_token, audio_token)

    `beta = 0.0` recovers OmniZip's original dpcknn; higher beta = more audio-aligned
    selection. Returns indices to keep.
    """
    N = tokens.shape[0]
    num_keep = int(N * keep_rate)
    if num_keep >= N:
        return torch.arange(N, device=tokens.device)

    with torch.no_grad():
        normed = torch.nn.functional.normalize(tokens, dim=1)
        sim = torch.mm(normed, normed.T)
        sim.fill_diagonal_(-float("inf"))
        kk = min(k, max(1, N - 1))
        knn_vals, _ = torch.topk(sim, kk, dim=1)
        diversity_score = knn_vals.mean(dim=1)

        if audio is not None and audio.numel() > 0 and beta > 0:
            a_norm = torch.nn.functional.normalize(audio, dim=1)
            av_sim = torch.mm(normed, a_norm.T)
            audio_sim = av_sim.max(dim=1).values
            combined = -diversity_score + beta * audio_sim
        else:
            combined = -diversity_score

        selected = torch.topk(combined, min(num_keep, N), largest=True).indices
    return selected


def omnizip_istm_audio_anchored(
    video_feature: torch.Tensor,
    audio_feature: Optional[torch.Tensor],
    num_tokens_per_frame: int = 196,
    merging_ratio: Sequence[float] = (0.7, 0.7),
    audio_anchor_beta: float = 0.3,
) -> torch.Tensor:
    """Audio-anchored ISTM. Drop-in replacement for OmniZip's `omnizip_istm`.

    Returns a bool keep-mask over video tokens, same shape as the input video_feature.
    """
    num_frames = video_feature.shape[0] // num_tokens_per_frame
    mask = torch.zeros(
        video_feature.shape[0], dtype=torch.bool, device=video_feature.device
    )

    for t in range(num_frames):
        ratio_id = 0 if t < 2 else 1
        keep_ratio = 1.0 - merging_ratio[ratio_id]
        start_idx = t * num_tokens_per_frame
        end_idx = (t + 1) * num_tokens_per_frame
        tokens = video_feature[start_idx:end_idx]

        if t % 2 == 0:
            keep_idx = dpcknn_audio_guided(
                tokens, audio_feature, keep_rate=keep_ratio, beta=audio_anchor_beta,
            )
            mask[start_idx:end_idx][keep_idx] = True
        else:
            prev_tokens = video_feature[(t - 1) * num_tokens_per_frame : t * num_tokens_per_frame]
            prev_norm = torch.nn.functional.normalize(prev_tokens, p=2, dim=1)
            curr_norm = torch.nn.functional.normalize(tokens, p=2, dim=1)
            similarity = torch.nn.functional.cosine_similarity(curr_norm, prev_norm, dim=1)
            num_keep = int(num_tokens_per_frame * keep_ratio)
            if num_keep < num_tokens_per_frame:
                keep_idx = similarity.topk(num_keep, largest=False).indices
            else:
                keep_idx = torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx:end_idx][keep_idx] = True

    return mask
