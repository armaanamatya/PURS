"""Cross-modal budget allocation.

Paper: Appendix B.1 "MACC Representation and Audio Budget Allocation" of
arXiv:2605.12056v1.

The audio budget is *referenced to the observed video retention* so the two
modalities stay coordinated within a refined chunk.

  rho_a, rho_v : global base merging ratios (config).
  R_v          : observed video retention in the current chunk.
  (1 - rho_v)  : base video retention level implied by rho_v.

  (13)  m_a = min( a_max, max( a_min, rho_a - beta * ( R_v - (1 - rho_v) ) ) )
  (14)  R_a = 1 - m_a

[a_min, a_max] are safety bounds on the audio *merging* ratio (Appendix A:
audio bounds [0.1, 0.9]) preventing over-pruning / under-compression.
"""
from __future__ import annotations

from .config import OmniRefineConfig


def audio_merge_ratio(r_v: float, cfg: OmniRefineConfig) -> float:
    """Eq (13): video-referenced audio merging ratio m_a."""
    raw = cfg.rho_a - cfg.beta * (r_v - (1.0 - cfg.rho_v))
    return min(cfg.a_max, max(cfg.a_min, raw))


def audio_retention(r_v: float, cfg: OmniRefineConfig) -> float:
    """Eq (14): R_a = 1 - m_a."""
    return 1.0 - audio_merge_ratio(r_v, cfg)
