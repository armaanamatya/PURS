"""VideoZip orchestrator. §4d of videozip_plan.md.

Mirrors `omnizip_units.omnizip()` so it is a drop-in monkey-patch target with the
same call signature, but inverts the guidance direction:
    OmniZip:  audio saliency -> per-group video ratios; one global audio ratio.
    VideoZip: video saliency -> per-group audio ratios; one global video ratio,
              plus audio-anchored ISTM for video pruning.

Designed to be installed via `videozip/eval/eval_videozip.py`'s monkey-patch, the
same way `eval_qwen_omni_zip_cached.py` swaps in cached audio scores. Extra kwargs
beyond OmniZip's signature (l6_video_scores, video_saliency_source,
audio_anchor_beta) are bound at the patch site.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import warnings

# Resolved via videozip/__init__.py sys.path bootstrap.
from omnizip_units import omnizip_audio_attn  # type: ignore[import-not-found]

from videozip.src.istm_audio_anchored import omnizip_istm_audio_anchored
from videozip.src.video_saliency import dispatch_video_saliency

MIN_VIDEO_RATIO = 0.35
MAX_VIDEO_RATIO = 0.75


# ── Helpers ──────────────────────────────────────────────────────────────────


def _apply_merge_plan_into_embeds(
    flat_embeds: torch.Tensor,
    audio_indices: torch.Tensor,
    audio_feature: torch.Tensor,
    video_feature: torch.Tensor,
    merge_plan: Dict[int, List[int]],
    g: int,
) -> None:
    """In-place: replace anchor positions with merged audio. Mirrors lines 169-189
    of `omnizip_units.omnizip()`."""
    if g <= 0 or not merge_plan:
        return
    device = flat_embeds.device
    a_norm = audio_feature / (audio_feature.norm(dim=-1, keepdim=True) + 1e-6)
    v_norm = (
        video_feature / (video_feature.norm(dim=-1, keepdim=True) + 1e-6)
        if video_feature.numel() > 0
        else None
    )
    for anchor_rel, merge_rel_list in merge_plan.items():
        if not merge_rel_list:
            continue
        merge_rel = torch.tensor(merge_rel_list, device=device, dtype=torch.long)
        if v_norm is not None:
            scores = (a_norm[merge_rel] @ v_norm.T).max(dim=1).values
        else:
            scores = (a_norm[merge_rel] @ a_norm[anchor_rel].unsqueeze(-1)).squeeze(-1)
        w = torch.softmax(scores, dim=0)
        anchor_vec = audio_feature[anchor_rel]
        merged_vec = (audio_feature[merge_rel] * w.unsqueeze(-1)).sum(dim=0)
        new_anchor = (anchor_vec + merged_vec) / (1.0 + w.sum())
        anchor_global = audio_indices[anchor_rel]
        flat_embeds[anchor_global] = new_anchor


def _redistribute_to_target(
    base_vs: List[float],
    target_total: float,
    min_ratio: float = MIN_VIDEO_RATIO,
    max_ratio: float = MAX_VIDEO_RATIO,
) -> List[float]:
    """OmniZip's ratio redistribution heuristic — preserves the min and max group,
    rescales the rest so the sum hits target_total."""
    n = len(base_vs)
    base_total = sum(base_vs)
    if abs(base_total - target_total) <= 1e-3 or base_total <= 0 or n <= 2:
        return list(base_vs)
    min_idx = base_vs.index(min(base_vs))
    max_idx = base_vs.index(max(base_vs))
    fixed = list(base_vs)
    min_val, max_val = fixed[min_idx], fixed[max_idx]
    other = [i for i in range(n) if i != min_idx and i != max_idx]
    other_sum = sum(fixed[i] for i in other)
    if not other or other_sum <= 0:
        return list(base_vs)
    remain_target = target_total - min_val - max_val
    scaling = remain_target / other_sum
    out: List[float] = []
    for i in range(n):
        if i == min_idx:
            out.append(min_val)
        elif i == max_idx:
            out.append(max_val)
        else:
            out.append(max(min_ratio, min(max_ratio, fixed[i] * scaling)))
    diff = target_total - sum(out)
    if abs(diff) > 1e-3 and other:
        adj = diff / len(other)
        for i in other:
            out[i] = max(min_ratio, min(max_ratio, out[i] + adj))
    return out


def _retention_to_per_group_ratios(
    retention: List[float],
    target_mean: float,
    min_ratio: float = MIN_VIDEO_RATIO,
    max_ratio: float = MAX_VIDEO_RATIO,
) -> List[float]:
    """High retention -> low merging ratio (prune less); low retention -> high.
    Then rescale so the per-group ratios sum to target_mean * len."""
    base = []
    for r in retention:
        m = max_ratio + (min_ratio - max_ratio) * r
        base.append(max(min_ratio, min(max_ratio, m)))
    return _redistribute_to_target(base, target_mean * len(base), min_ratio, max_ratio)


def _aggregate_video_scores_to_groups(
    video_scores_1d: torch.Tensor,
    num_input_frames: int,
    video_token_per_frame: int,
    group_count: int,
) -> List[float]:
    """Spatial mean per frame, then frame mean per group. Returns normalized [0, 1]."""
    n_v = num_input_frames * video_token_per_frame
    if video_scores_1d.numel() != n_v:
        # Exact match required: slicing/padding here would misalign saliency with the
        # live video tokens and silently produce an invalid benchmark. Fail loud instead.
        raise ValueError(
            f"Cached L6 video saliency length ({video_scores_1d.numel()}) != live video "
            f"tokens (num_input_frames={num_input_frames} * video_token_per_frame="
            f"{video_token_per_frame} = {n_v}). The eval's video preprocessing "
            f"(fps/max_frames/max_pixels) must match the cache build config in "
            f"precompute_l6_saliency.py (defaults: fps=2.0, max_frames=64, "
            f"max_pixels=100800)."
        )
    frames_x_spatial = video_scores_1d.reshape(num_input_frames, video_token_per_frame)
    frame_imp = frames_x_spatial.mean(dim=1)
    frames_per_group = max(1, num_input_frames // group_count)
    group_imps: List[float] = []
    for gi in range(group_count):
        s = gi * frames_per_group
        e = s + frames_per_group if gi < group_count - 1 else num_input_frames
        group_imps.append(frame_imp[s:e].mean().item())
    mx = max(group_imps) + 1e-8
    return [v / mx for v in group_imps]


def _build_frame_token_groups(
    num_input_frames: int,
    video_token_per_frame: int,
    group_count: int,
) -> List[Tuple[int, int]]:
    """Partition video tokens by frame groups; the final group absorbs the tail."""
    frames_per_group = max(1, num_input_frames // group_count)
    groups: List[Tuple[int, int]] = []
    for gi in range(group_count):
        f_start = gi * frames_per_group
        f_end = f_start + frames_per_group if gi < group_count - 1 else num_input_frames
        groups.append((f_start * video_token_per_frame, f_end * video_token_per_frame))
    return groups


def _audio_anchored_istm_keep_tail(
    video_feature: torch.Tensor,
    audio_anchors: Optional[torch.Tensor],
    num_tokens_per_frame: int,
    merging_ratio: Tuple[float, float],
    audio_anchor_beta: float,
) -> torch.Tensor:
    """Run ISTM on complete frame pairs and keep an incomplete tail unpruned."""
    if num_tokens_per_frame <= 0 or video_feature.numel() == 0:
        return torch.ones(video_feature.size(0), dtype=torch.bool, device=video_feature.device)

    full_len = (video_feature.size(0) // num_tokens_per_frame) * num_tokens_per_frame
    if full_len == 0:
        return torch.ones(video_feature.size(0), dtype=torch.bool, device=video_feature.device)

    full_mask = omnizip_istm_audio_anchored(
        video_feature=video_feature[:full_len],
        audio_feature=audio_anchors,
        num_tokens_per_frame=num_tokens_per_frame,
        merging_ratio=merging_ratio,
        audio_anchor_beta=audio_anchor_beta,
    )
    if full_len == video_feature.size(0):
        return full_mask

    tail = torch.ones(
        video_feature.size(0) - full_len,
        dtype=torch.bool,
        device=video_feature.device,
    )
    return torch.cat([full_mask, tail], dim=0)


def _project_importance_to_audio_scores(
    attn_logits: Optional[torch.Tensor],
    audio_indices: torch.Tensor,
) -> torch.Tensor:
    """Project OmniZip's importance channel into audio-local coordinates.

    VideoZip calls `omnizip_audio_attn()` per audio group. That function expects
    a 1D importance vector aligned to the local audio slice, so full-sequence
    attention must be reduced once and indexed at the exact audio token positions.
    """
    device = audio_indices.device
    if attn_logits is None:
        return torch.ones(audio_indices.numel(), device=device)

    scores = attn_logits.to(device)
    if scores.dim() == 3:
        scores = scores.mean(dim=0).sum(dim=0)
    elif scores.dim() == 2:
        scores = scores.sum(dim=0)
    elif scores.dim() != 1:
        raise ValueError(f"attn_logits must be 1D, 2D, or 3D; got {tuple(scores.shape)}")

    if scores.numel() == audio_indices.numel():
        return scores

    if audio_indices.numel() == 0:
        return torch.zeros(0, device=device, dtype=scores.dtype)

    max_idx = int(audio_indices.max().item())
    if scores.numel() <= max_idx:
        padded = torch.zeros(max_idx + 1, device=device, dtype=scores.dtype)
        padded[: scores.numel()] = scores
        scores = padded

    return scores[audio_indices]


# ── Main entry ───────────────────────────────────────────────────────────────


def omnizip_videozip(
    input_embeds: torch.Tensor,
    attn_logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    num_input_frames: int,
    merging_ratio_audio: float = 0.5,
    merging_ratio_v: float = 0.5,
    contextual_ratio: float = 0.05,
    g: int = 3,
    *,
    l6_video_scores: Optional[torch.Tensor] = None,
    video_saliency_source: str = "l6_cached",
    audio_anchor_beta: float = 0.3,
):
    """Drop-in replacement for `omnizip_units.omnizip()`.

    Positional args match OmniZip exactly so the eval-side monkey-patch can
    forward them through. Keyword-only args after `*` are bound by the patch
    wrapper; modeling code does not pass them.
    """
    device = input_embeds.device
    is_batched = input_embeds.dim() == 3
    if is_batched:
        B, L, D = input_embeds.shape
        flat_embeds = input_embeds.reshape(-1, D)
        flat_ids = input_ids.reshape(-1)
    else:
        L, D = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids

    video_indices = torch.nonzero((flat_ids == video_token_id).to(device), as_tuple=True)[0]
    audio_indices = torch.nonzero((flat_ids == audio_token_id).to(device), as_tuple=True)[0]

    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]
    N_v = video_feature.shape[0]
    N_a = audio_feature.shape[0]

    if N_v == 0 or N_a == 0:
        return input_embeds, torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)

    video_token_per_frame = max(1, N_v // num_input_frames)

    group_count = max(1, (num_input_frames + 3) // 4)
    video_groups = _build_frame_token_groups(
        num_input_frames, video_token_per_frame, group_count,
    )
    num_audio_per_group = max(1, N_a // group_count)

    # Build audio groups (uniform partition; tail group absorbs remainder).
    audio_groups: List[Tuple[int, int]] = []
    for i in range(group_count):
        s = i * num_audio_per_group
        e = (i + 1) * num_audio_per_group if i < group_count - 1 else N_a
        audio_groups.append((s, e))

    # Step 1: video saliency -> per-group retention.
    # Cached L6 scores must positionally align with the live video tokens. Frame-decode
    # jitter (decord returning a slightly different frame count than the precompute pass)
    # can make the cached length differ by <1% on the odd clip. In that case, fall back to
    # online saliency for THIS video rather than misaligning scores or hard-erroring the
    # row -- the exact-equality guard in _aggregate_video_scores_to_groups stays as a
    # backstop for gross (config-level) mismatches.
    n_v_expected = num_input_frames * video_token_per_frame
    cached_available = l6_video_scores is not None
    cached_aligned = cached_available and int(l6_video_scores.numel()) == n_v_expected
    if cached_available and not cached_aligned:
        warnings.warn(
            f"VideoZip: cached L6 saliency length {int(l6_video_scores.numel())} != live "
            f"video tokens {n_v_expected} (num_input_frames={num_input_frames} * "
            f"video_token_per_frame={video_token_per_frame}); likely frame-decode jitter. "
            f"Falling back to online saliency for this video.",
            stacklevel=2,
        )
    if cached_aligned:
        scores_t = l6_video_scores.to(device).float()
        if scores_t.dim() != 1:
            raise ValueError(
                f"l6_video_scores must be 1D (length N_v); got shape {tuple(scores_t.shape)}"
            )
        video_group_retention = _aggregate_video_scores_to_groups(
            scores_t, num_input_frames, video_token_per_frame, group_count,
        )
    else:
        if cached_available:
            fallback_source = "attn" if attn_logits is not None else "sim_only"
        else:
            fallback_source = video_saliency_source
        video_group_retention = dispatch_video_saliency(
            fallback_source,
            video_feature=video_feature,
            video_indices=video_indices,
            num_input_frames=num_input_frames,
            video_token_per_frame=video_token_per_frame,
            num_groups=group_count,
            attn_logits=attn_logits,
            audio_feature=audio_feature,
        )

    # Step 2: video retention -> per-group audio merging ratios (the swap).
    audio_merging_ratios = _retention_to_per_group_ratios(
        video_group_retention, target_mean=merging_ratio_audio,
    )

    # Step 3: per-group audio compression. Project full-sequence attention into
    # audio-local scores, then slice the score vector for each group.
    audio_scores = _project_importance_to_audio_scores(attn_logits, audio_indices)
    audio_mask = torch.zeros(N_a, dtype=torch.bool, device=device)
    merge_plan: Dict[int, List[int]] = {}
    for (a_start, a_end), ratio in zip(audio_groups, audio_merging_ratios):
        if a_start >= a_end:
            continue
        group_feat = audio_feature[a_start:a_end]
        group_logits = audio_scores[a_start:a_end]
        group_mask, group_plan = omnizip_audio_attn(
            audio_feature=group_feat,
            video_feature=video_feature,
            attn_logits=group_logits,
            merging_ratio=ratio,
            contextual_ratio=contextual_ratio,
            g=g,
        )
        audio_mask[a_start:a_end] = group_mask
        for anchor_loc, merges_loc in group_plan.items():
            merge_plan[anchor_loc + a_start] = [m + a_start for m in merges_loc]

    # Step 4: apply merge plan in place (anchor positions absorb merged tokens).
    _apply_merge_plan_into_embeds(
        flat_embeds, audio_indices, audio_feature, video_feature, merge_plan, g,
    )

    # Step 5: audio-anchored ISTM, run pairwise on (group, group+1) like OmniZip.
    # Re-read from flat_embeds so anchors include any merge-plan updates.
    audio_kept_features = flat_embeds[audio_indices][audio_mask]
    audio_anchors: Optional[torch.Tensor] = (
        audio_kept_features if audio_mask.any() else None
    )
    video_group_masks: List[torch.Tensor] = []
    tokens_per_istm_frame = video_token_per_frame * 2
    for i in range(0, group_count, 2):
        v_start = video_groups[i][0]
        v_end = video_groups[i + 1][1] if i + 1 < group_count else video_groups[i][1]
        group_feat = video_feature[v_start:v_end]
        group_len = group_feat.size(0)
        if group_len == 0:
            video_group_masks.append(torch.zeros(0, dtype=torch.bool, device=device))
            continue
        gm = _audio_anchored_istm_keep_tail(
            video_feature=group_feat,
            audio_anchors=audio_anchors,
            num_tokens_per_frame=tokens_per_istm_frame,
            merging_ratio=(merging_ratio_v, merging_ratio_v),
            audio_anchor_beta=audio_anchor_beta,
        )
        video_group_masks.append(gm)

    video_mask = torch.cat(video_group_masks, dim=0) if video_group_masks else torch.zeros(0, dtype=torch.bool, device=device)
    if video_mask.shape[0] < N_v:
        pad = torch.ones(N_v - video_mask.shape[0], dtype=torch.bool, device=device)
        video_mask = torch.cat([video_mask, pad], dim=0)
    elif video_mask.shape[0] > N_v:
        video_mask = video_mask[:N_v]

    # Step 6: assemble global keep mask.
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    global_mask[video_indices] = video_mask
    global_mask[audio_indices] = audio_mask

    if is_batched:
        input_embeds_out = flat_embeds.reshape(B, L, D)
    else:
        input_embeds_out = flat_embeds
    return input_embeds_out, global_mask
