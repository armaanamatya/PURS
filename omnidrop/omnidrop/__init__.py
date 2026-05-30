"""OmniDrop: training-free, layer-wise token pruning for Omni-modal LLMs.

Minimal, citation-anchored reproduction of arXiv 2605.14458
("OmniDrop: Layer-wise Token Pruning for Omni-modal LLMs via Query-Guidance").

Public API:
    PLPSchedule, calibrate_pfinal, calibrate_pfinal_paper_approx   (Sec. 3.1, App. E)
    query_guided_importance                                        (Sec. 3.2, Eq. 5)
    tds_select_to_prune, topk_lowest_to_prune                      (Sec. 3.3, Alg. 1)
    prune_audio_by_attention, prune_video_ttm                      (Sec. 3.4)
    OmniDropConfig, TokenLayout, OmniDropPruner                    (orchestrator)
"""
from .schedule import (
    PLPSchedule,
    calibrate_pfinal,
    calibrate_pfinal_paper_approx,
)
from .importance import query_guided_importance
from .tds import tds_select_to_prune, topk_lowest_to_prune
from .intra_modality import prune_audio_by_attention, prune_video_ttm
from .pruner import OmniDropConfig, TokenLayout, OmniDropPruner

__all__ = [
    "PLPSchedule",
    "calibrate_pfinal",
    "calibrate_pfinal_paper_approx",
    "query_guided_importance",
    "tds_select_to_prune",
    "topk_lowest_to_prune",
    "prune_audio_by_attention",
    "prune_video_ttm",
    "OmniDropConfig",
    "TokenLayout",
    "OmniDropPruner",
]

__version__ = "0.1.0"
