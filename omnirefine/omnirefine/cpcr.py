"""Stage 1 -- Correspondence-Preserving Chunk Refinement (CPCR).

Paper: Sec 3.3 + Appendix B.2 / Algorithm 1 of arXiv:2605.12056v1.

Native temporal chunks in Qwen2.5-Omni are fixed-duration interleaved
audio/video partitions; they are coarse alignment priors that do not
always match local audio-visual correspondence (Sec 3.1).  CPCR refines
their boundaries into cross-modally aligned compression units by solving
a temporally-constrained joint segmentation over frames and audio tokens
with dynamic programming.

Equations implemented:
  (3)  S_{f,t} = cos( mean_p v_{f,p} , a_t )
  (4)  M_{f,t} = 1[ c_a(t) in N(c_v(f)) ]          (native-neighborhood mask)
  (5)  S~_{f,t} = M_{f,t} * S_{f,t}
  (6)  phi(i,u,j,q) = sum S~ over block / sum M over block
  (5')/Alg1 recurrence (source of truth -- see REPRODUCTION_NOTES #3):
       cost = -S_match + lambda_c * ChunkVariance ; minimize D (D init +inf)

NOTE: the main-text Eq (5) maximizes D[i,j]+phi-lambda_c (flat penalty),
while Algorithm 1 minimizes a cost with lambda_c * ChunkVariance.  We
implement Algorithm 1 (the appendix is the operational spec) and define
ChunkVariance explicitly as the variance of the masked similarities inside
the block.  ChunkVariance is INVENTED (paper leaves it undefined).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import OmniRefineConfig
from .utils import (
    as_numpy,
    cosine_matrix,
    integral_image,
    l2_normalize,
    rect_sum,
)


@dataclass
class RefinedChunk:
    """A refined compression unit.

    Frame interval is the half-open ``[frame_start, frame_end)`` and audio
    interval the half-open ``[audio_start, audio_end)`` (0-indexed).
    """

    frame_start: int
    frame_end: int
    audio_start: int
    audio_end: int

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def n_audio(self) -> int:
        return self.audio_end - self.audio_start


def frame_embeddings(video_hidden_by_frame: List[np.ndarray]) -> np.ndarray:
    """Aggregate per-frame video tokens into frame embeddings (Sec 3.1, Eq 3).

    ``video_hidden_by_frame[f]`` is an (P_f, D) array of the frame's video
    token hidden states.  Returns (F, D).
    """
    return np.stack(
        [as_numpy(f).reshape(-1, as_numpy(f).shape[-1]).mean(axis=0)
         for f in video_hidden_by_frame],
        axis=0,
    )


def neighborhood_mask(
    frame_native_chunk: np.ndarray,
    audio_native_chunk: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Eq (4): M_{f,t} = 1 if audio token t's native chunk is within
    ``radius`` of frame f's native chunk.

    ``frame_native_chunk[f]`` and ``audio_native_chunk[t]`` are the native
    (pre-refinement) chunk indices supplied by the model's temporal position
    indexing.  ``radius`` realises the neighborhood N(.) -- PARTIAL in the
    paper (see REPRODUCTION_NOTES); default 1 = the chunk and its immediate
    temporal neighbours.
    """
    fc = np.asarray(frame_native_chunk).reshape(-1, 1)
    ac = np.asarray(audio_native_chunk).reshape(1, -1)
    return (np.abs(fc - ac) <= radius).astype(np.float64)


def similarity_field(
    frame_emb: np.ndarray,
    audio_emb: np.ndarray,
    nbr_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the masked similarity field S~ (Eq 3 -> 5).

    Returns ``(S_tilde, M)`` both shaped (F, N).
    """
    S = cosine_matrix(as_numpy(frame_emb), as_numpy(audio_emb))  # Eq 3
    M = np.asarray(nbr_mask, dtype=np.float64)                   # Eq 4
    return S * M, M                                              # Eq 5


def refine_chunks(
    S_tilde: np.ndarray,
    M: np.ndarray,
    cfg: OmniRefineConfig,
) -> List[RefinedChunk]:
    """Constrained DP joint chunking (Algorithm 1).

    Returns the refined chunk sequence covering all F frames and N audio
    tokens, with per-chunk sizes inside [s_v_min, s_v_max] (frames) and
    [s_a_min, s_a_max] (audio).
    """
    S_tilde = np.asarray(S_tilde, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    F, N = S_tilde.shape

    # Integral images for O(1) block statistics (mean & variance of the
    # masked similarities -> phi (Eq 6) and ChunkVariance).
    P_s = integral_image(S_tilde)
    P_s2 = integral_image(S_tilde * S_tilde)
    P_m = integral_image(M)

    def block_stats(i: int, u: int, j: int, q: int):
        m = rect_sum(P_m, i, u, j, q)
        if m <= 0.0:
            return None
        s = rect_sum(P_s, i, u, j, q)
        s2 = rect_sum(P_s2, i, u, j, q)
        mean = s / m                         # phi(i,u,j,q)  (Eq 6)
        var = max(0.0, s2 / m - mean * mean)  # ChunkVariance (INVENTED)
        return mean, var

    INF = np.inf
    D = np.full((F + 1, N + 1), INF)
    D[0, 0] = 0.0
    Pi = np.full((F + 1, N + 1, 2), -1, dtype=np.int64)

    n_over_f = N / F
    band = max(float(cfg.dp_window), cfg.dp_band_ratio * n_over_f * cfg.s_v_max)

    for i in range(1, F + 1):
        j_hat = i * n_over_f
        for j in range(1, N + 1):
            # Admissible band mask (Algorithm 1, lines 4-6).
            if abs(j - j_hat) > band:
                continue
            i0_min = max(0, i - cfg.s_v_max)
            i0_max = i - cfg.s_v_min
            j0_min = max(0, j - cfg.s_a_max)
            j0_max = j - cfg.s_a_min
            if i0_max < i0_min or j0_max < j0_min:
                continue
            best = D[i, j]
            best_prev = (-1, -1)
            for pi in range(i0_min, i0_max + 1):
                row = D[pi]
                for pj in range(j0_min, j0_max + 1):
                    if not np.isfinite(row[pj]):
                        continue
                    stats = block_stats(pi, i, pj, j)
                    if stats is None:
                        continue
                    mean, var = stats
                    cost = -mean + cfg.lambda_c * var  # Algorithm 1
                    val = row[pj] + cost
                    if val < best:
                        best = val
                        best_prev = (pi, pj)
            if best_prev[0] >= 0:
                D[i, j] = best
                Pi[i, j] = best_prev

    if not np.isfinite(D[F, N]):
        raise ValueError(
            "CPCR DP found no feasible segmentation of F=%d frames into "
            "N=%d audio tokens under the size/band constraints. Check that "
            "N/F is roughly within [s_a/s_v] ratios and widen the DP band "
            "if needed." % (F, N)
        )

    # Traceback (Algorithm 1, line: optimal path P*).
    chunks: List[RefinedChunk] = []
    i, j = F, N
    while (i, j) != (0, 0):
        pi, pj = (int(Pi[i, j, 0]), int(Pi[i, j, 1]))
        if pi < 0:
            break
        chunks.append(RefinedChunk(pi, i, pj, j))
        i, j = pi, pj
    chunks.reverse()
    return chunks


def correspondence_preserving_refinement(
    video_hidden_by_frame: List[np.ndarray],
    audio_hidden: np.ndarray,
    frame_native_chunk: np.ndarray,
    audio_native_chunk: np.ndarray,
    cfg: OmniRefineConfig,
    neighborhood_radius: int = 1,
) -> List[RefinedChunk]:
    """End-to-end CPCR: embeddings -> masked similarity -> DP refinement."""
    f_emb = frame_embeddings(video_hidden_by_frame)
    a_emb = as_numpy(audio_hidden)
    nbr = neighborhood_mask(frame_native_chunk, audio_native_chunk, neighborhood_radius)
    S_tilde, M = similarity_field(f_emb, a_emb, nbr)
    return refine_chunks(S_tilde, M, cfg)
