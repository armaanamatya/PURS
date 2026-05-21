"""Per-group audio compression using video-derived merging ratios. §4b of videozip_plan.md.

Wraps the existing OmniZip `omnizip_audio_attn` per group — video still guides anchor
selection within each group via the cross-modal sim_av term inside that function.
What changes here is that VIDEO now decides HOW MUCH to prune each audio group.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

# Imported via videozip/__init__.py sys.path bootstrap.
from omnizip_units import omnizip_audio_attn  # type: ignore[import-not-found]


def omnizip_audio_compress(
    audio_feature: torch.Tensor,
    video_feature: Optional[torch.Tensor],
    attn_logits: torch.Tensor,
    audio_groups: List[Tuple[int, int]],
    audio_merging_ratios: List[float],
    contextual_ratio: float = 0.05,
    g: int = 3,
) -> Tuple[torch.Tensor, Dict[int, List[int]]]:
    """Apply OmniZip's audio compression once per group with a different merging ratio.

    Returns:
        audio_mask: bool tensor [N_a] — True = keep
        merge_plan: anchor_idx -> [merged_idxs], indices in the global audio coord space
    """
    N_a = audio_feature.shape[0]
    audio_mask = torch.zeros(N_a, dtype=torch.bool, device=audio_feature.device)
    merge_plan: Dict[int, List[int]] = {}

    for (a_start, a_end), ratio in zip(audio_groups, audio_merging_ratios):
        if a_start >= a_end:
            continue
        group_feat = audio_feature[a_start:a_end]
        group_mask, group_plan = omnizip_audio_attn(
            audio_feature=group_feat,
            video_feature=video_feature,
            attn_logits=attn_logits,
            merging_ratio=ratio,
            contextual_ratio=contextual_ratio,
            g=g,
        )
        audio_mask[a_start:a_end] = group_mask
        for anchor, merges in group_plan.items():
            merge_plan[anchor + a_start] = [m + a_start for m in merges]

    return audio_mask, merge_plan
