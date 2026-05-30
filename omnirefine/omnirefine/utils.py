"""Small numeric helpers shared across OmniRefine modules.

Pure NumPy so the algorithm core has no torch dependency (training-free,
tensor-level).  Adapters may pass torch tensors; convert with
``as_numpy`` at the boundary.
"""
from __future__ import annotations

import numpy as np


def as_numpy(x) -> np.ndarray:
    """Accept numpy arrays or torch tensors; return a float64 ndarray."""
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype=np.float64)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``a`` (m,D) and ``b`` (n,D).

    Returns (m, n).
    """
    return l2_normalize(a) @ l2_normalize(b).T


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    return float(
        np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)
    )


def mean_pool(hidden: np.ndarray) -> np.ndarray:
    """z(.) of Eq 12: average-pooled hidden states of a region/block."""
    return np.asarray(hidden, dtype=np.float64).reshape(-1, hidden.shape[-1]).mean(axis=0)


def integral_image(mat: np.ndarray) -> np.ndarray:
    """2D prefix-sum with a zero top/left pad, shape (R+1, C+1)."""
    R, C = mat.shape
    P = np.zeros((R + 1, C + 1), dtype=np.float64)
    P[1:, 1:] = np.cumsum(np.cumsum(mat, axis=0), axis=1)
    return P


def rect_sum(P: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    """Sum over the half-open block rows [r0,r1), cols [c0,c1) using an
    integral image ``P`` produced by :func:`integral_image`."""
    return float(P[r1, c1] - P[r0, c1] - P[r1, c0] + P[r0, c0])
