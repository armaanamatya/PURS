"""Configuration for OmniRefine.

All values are taken from the paper:
  OmniRefine: Alignment-Aware Cooperative Compression for Efficient
  Omnimodal Large Language Models (arXiv:2605.12056v1, 12 May 2026).

Sources:
  - Sec 4.1 "Implementation Details"
  - Appendix A "Hyperparameter Settings"
  - Appendix B, Algorithm 1 (DP bounds)

Every field carries an inline citation. Fields marked UNSPECIFIED or
PARTIAL are choices the paper does not fully pin down; see
REPRODUCTION_NOTES.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OmniRefineConfig:
    # ---- Global merging ratios (Sec 4.1 "Implementation Details") ----
    # rho_a, rho_v are the *base merging* (compression) ratios, so the base
    # retention levels are (1 - rho).  Eq 13/14 and Appendix B.1.
    rho_a: float = 0.3          # base audio merging ratio
    rho_v: float = 0.6          # base video merging ratio
    contextual_ratio: float = 0.05  # fraction of contextual anchors kept

    # ---- Video branch thresholds (Sec 4.1; Eq 6, Eq 7) ----
    tau_s: float = 0.82         # spatial similarity threshold (Eq 6)
    tau_t: float = 0.58         # temporal similarity threshold (Eq 7)

    # ---- Audio branch (Sec 4.1) ----
    beta: float = 0.4           # cross-modal budget coefficient (Eq 13)
    semantic_anchor_threshold: float = 0.4  # anchor split threshold (Sec 3.4)

    # ---- CPCR (Sec 4.1; Eq 5; Algorithm 1) ----
    lambda_c: float = 0.02      # chunk regularization term

    # ---- Cross-modal budget hard bounds (Appendix A) ----
    v_min: float = 0.18         # video retention lower bound
    v_max: float = 0.55         # video retention upper bound
    a_min: float = 0.10         # audio merging-ratio lower bound
    a_max: float = 0.90         # audio merging-ratio upper bound
    alpha: float = 0.15         # video budget modulation factor (Appendix A)

    # ---- Chunking / DP alignment constraints (Appendix A; Algorithm 1) ----
    s_v_min: int = 3            # min frames per video chunk
    s_v_max: int = 5            # max frames per video chunk
    s_a_min: int = 90           # min audio tokens per chunk
    s_a_max: int = 140          # max audio tokens per chunk
    dp_band_ratio: float = 2.0  # B in Algorithm 1
    dp_window: int = 48         # W in Algorithm 1 (min DP window)

    # ---- Probe layer (UNSPECIFIED by the paper) ----
    # The training-free similarity / saliency signals must come from a
    # partial forward pass at some layer L.  The paper's *method* never
    # states which layer; the motivating analysis (Sec 3.1) merely inspects
    # layers 0 and 8.  This is the single load-bearing free parameter.
    # Default mirrors the deepest analysis layer (8); tune per model.
    # See REPRODUCTION_NOTES.md item #2.
    layer_probe: int = 8

    def __post_init__(self) -> None:
        assert 0.0 < self.rho_a < 1.0
        assert 0.0 < self.rho_v < 1.0
        assert self.s_v_min <= self.s_v_max
        assert self.s_a_min <= self.s_a_max
        assert self.v_min <= self.v_max
        assert self.a_min <= self.a_max
