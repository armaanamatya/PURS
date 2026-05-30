"""Shared helpers extracted from OmniZip's orchestration logic so VideoZip can reuse them."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def _map_retention_to_ratios(
    retention_scores: Sequence[float],
    target_mean_ratio: float,
    min_ratio: float = 0.35,
    max_ratio: float = 0.75,
) -> List[float]:
    """Map per-group retention scores in [0, 1] to per-group merging ratios in
    [min_ratio, max_ratio].

    High retention = this group is salient = prune LESS = small merging_ratio.
    Low retention  = this group is unimportant = prune MORE = large merging_ratio.

    The result is rescaled so that its mean equals `target_mean_ratio` (so the global
    compression budget is preserved across groups).
    """
    if not retention_scores:
        return []

    raw = [max_ratio + (min_ratio - max_ratio) * r for r in retention_scores]
    mean_raw = sum(raw) / len(raw)
    if mean_raw <= 0:
        return [target_mean_ratio] * len(retention_scores)
    scale = target_mean_ratio / mean_raw
    rescaled = [max(0.0, min(0.99, r * scale)) for r in raw]
    return rescaled


def _build_audio_groups(num_audio_tokens: int, num_groups: int) -> List[Tuple[int, int]]:
    """Partition [0, num_audio_tokens) into `num_groups` contiguous half-open intervals.

    The last group absorbs any remainder so that no audio token is dropped.
    """
    if num_groups <= 0 or num_audio_tokens <= 0:
        return []
    base = num_audio_tokens // num_groups
    groups: List[Tuple[int, int]] = []
    cursor = 0
    for g in range(num_groups):
        end = cursor + base if g < num_groups - 1 else num_audio_tokens
        groups.append((cursor, end))
        cursor = end
    return groups
