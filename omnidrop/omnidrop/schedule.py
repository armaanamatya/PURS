"""Progressive Layer-wise Token Pruning (PLP) schedule.

Implements the depth-dependent pruning ratio of OmniDrop (arXiv 2605.14458,
Section 3.1, Eqs. 3-4 and Eq. 6) and the fixed-budget calibration of Appendix E.

Equations (reconstructed from the paper text; the rendered equation images were
not present in the extracted PDF, but the surrounding prose pins them exactly):

    Eq. 4 (sigmoid schedule):  s_l = sigmoid( beta * (l / L - t_mid) )
    Eq. 3 (PLP ratio):         p_l = p_init + (p_final - p_init) * s_l
    Eq. 6 (token count):       k_l = floor( p_l * (n^l_A + n^l_V) )

PLP is applied up to the penultimate layer L-2 (Section 3.1: "to preserve the
hidden states immediately preceding the language modeling head"), so layer L-1
performs no pruning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _sigmoid(x: float) -> float:
    # Numerically stable logistic sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class PLPSchedule:
    """Progressive Layer-wise Pruning schedule (OmniDrop Sec. 3.1).

    Args:
        p_init:  initial pruning ratio (shallow layers).            [Eq. 3]
        p_final: final pruning ratio (deep layers).                 [Eq. 3]
        L:       total number of decoder layers (28 for 7B, 36 for 3B).
        t_mid:   sigmoid midpoint as a fraction of depth.   default 0.5  [Sec. 4]
        beta:    sigmoid slope.                             default 20   [Sec. 4]
        kind:    "sigmoid" (default) or "exp" (ablation variant, Sec. 4.2).
    """

    p_init: float
    p_final: float
    L: int
    t_mid: float = 0.5
    beta: float = 20.0
    kind: str = "sigmoid"

    def pruning_ratio(self, l: int) -> float:
        """Return p_l for decoder layer index ``l`` (0-based).

        Layer L-1 is never pruned (PLP applied up to L-2, Sec. 3.1).
        """
        if l < 0 or l >= self.L:
            raise IndexError(f"layer {l} out of range [0, {self.L - 1}]")
        if l > self.L - 2:  # penultimate-layer cap (Sec. 3.1)
            return 0.0
        if self.kind == "sigmoid":
            s_l = _sigmoid(self.beta * (l / self.L - self.t_mid))  # Eq. 4
            return self.p_init + (self.p_final - self.p_init) * s_l  # Eq. 3
        if self.kind == "exp":
            # Ablation variant (Sec. 4.2): p_l = p_init * (p_final/p_init)^(l/(L-2)).
            # Undefined for p_init == 0; the sigmoid configs use p_init=0 so this
            # variant is only valid when p_init > 0.
            if self.p_init <= 0.0:
                raise ValueError("exp schedule requires p_init > 0")
            return self.p_init * (self.p_final / self.p_init) ** (l / (self.L - 2))
        raise ValueError(f"unknown schedule kind: {self.kind!r}")

    def num_to_prune(self, l: int, current_av_count: int) -> int:
        """k_l = floor(p_l * current AV-token count)  [Eq. 6].

        IMPORTANT: ``current_av_count`` is the number of audiovisual tokens that
        survive *entering* layer l, not the original count -- pruning is
        cumulative (Appendix E: r_l = r_0 * prod(1 - p_j)).
        """
        return int(math.floor(self.pruning_ratio(l) * current_av_count))

    def retained_trajectory(self, r0: float = 0.45) -> list[float]:
        """Cumulative retained ratio r_l entering each layer l = 0..L-1.

        r_0 = r0 (post intra-modality pruning, Appendix E);
        r_l = r0 * prod_{j<l} (1 - p_j).
        Returns a list of length L. Its arithmetic mean is the reported
        "average token retention ratio" used to compare against fixed-ratio
        baselines (Sec. 4).
        """
        traj = []
        r = r0
        for l in range(self.L):
            traj.append(r)          # ratio entering layer l
            r = r * (1.0 - self.pruning_ratio(l))
        return traj

    def mean_retained(self, r0: float = 0.45) -> float:
        """Mean of the cumulative retained-ratio trajectory (R-bar, App. E)."""
        traj = self.retained_trajectory(r0)
        return sum(traj) / len(traj)


def calibrate_pfinal(
    target_mean: float,
    L: int,
    p_init: float = 0.0,
    t_mid: float = 0.5,
    beta: float = 20.0,
    r0: float = 0.45,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> float:
    """Solve for p_final so the mean retained ratio equals ``target_mean``.

    Exact bisection on the cumulative-decay model of Appendix E. The paper's
    closed-form derivation (see ``calibrate_pfinal_paper_approx``) gives
    p_final ~= 0.146 for the canonical 7B / 30% setting; this numeric solver is
    the general tool.
    """
    lo, hi = 0.0, 1.0

    def mean_for(pf: float) -> float:
        return PLPSchedule(p_init, pf, L, t_mid, beta).mean_retained(r0)

    # mean_retained decreases as p_final increases -> monotone, bisection-safe.
    if mean_for(lo) < target_mean:
        return 0.0  # even no pruning cannot reach target (target > r0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = mean_for(mid)
        if abs(m - target_mean) < tol:
            return mid
        if m > target_mean:   # retaining too much -> prune harder
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_pfinal_paper_approx(
    target_mean: float = 0.30,
    L: int = 28,
    r0: float = 0.45,
) -> float:
    """Reproduce the closed-form derivation in Appendix E.2.

    Bisection schedule (p_init=0, t_mid=0.5): Phase 1 (first L/2 layers) keeps
    r0 constant; Phase 2 decays geometrically. Approximating Phase-2 mean by its
    midpoint layer (the (L/4)-th decay step):

        r_phase2 = 2 * target_mean - r0
        r0 * (1 - p_final)^steps = r_phase2,  steps = L/4

    For L=28, target=0.30, r0=0.45 this yields p_final ~= 0.146 (paper adopts a
    conservative 0.2).
    """
    steps = L // 4  # midpoint of Phase 2 = (L/2)/2 decay steps from start of Phase 2
    r_phase2 = 2.0 * target_mean - r0
    if r_phase2 <= 0:
        raise ValueError("target unreachable: 2*target_mean must exceed r0")
    return 1.0 - (r_phase2 / r0) ** (1.0 / steps)
