"""Torch runtime helper for experimental Qwen2.5-Omni OmniDrop integration.

This module implements the layer-wise token-selection part of OmniDrop on live
torch tensors. It is intentionally model-light: the Qwen model owns hidden-state
slicing and KV-cache creation, while this helper owns token metadata, scoring,
and per-layer keep masks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from .schedule import PLPSchedule


def audio_importance_from_attn(attn_logits: torch.Tensor, n_audio: int) -> torch.Tensor:
    """Reduce audio-encoder attention to one saliency score per audio token."""
    if attn_logits.dim() == 3:
        importance = attn_logits.float().mean(dim=0).sum(dim=0)
    elif attn_logits.dim() == 2:
        importance = attn_logits.float().sum(dim=0)
    else:
        importance = attn_logits.float().flatten()
    if importance.numel() > n_audio:
        importance = importance[:n_audio]
    elif importance.numel() < n_audio:
        pad = torch.zeros(n_audio - importance.numel(), device=importance.device, dtype=importance.dtype)
        importance = torch.cat([importance, pad], dim=0)
    return importance


def video_ttm_keep_mask(
    video_feature: torch.Tensor,
    num_input_frames: int,
    group_size: int = 4,
    prune_rate: float = 0.8,
) -> torch.Tensor:
    """Dycoke-style TTM keep mask used by OmniDrop Sec. 3.4.

    The first frame in each 4-frame group is kept fully. For every other frame,
    keep the least-similar ``1 - prune_rate`` fraction of tokens relative to the
    anchor frame at the same spatial/token index.
    """
    n_video = int(video_feature.shape[0])
    if n_video == 0:
        return torch.zeros(0, dtype=torch.bool, device=video_feature.device)
    if num_input_frames <= 0 or n_video % max(num_input_frames, 1) != 0:
        return torch.ones(n_video, dtype=torch.bool, device=video_feature.device)

    tokens_per_frame = n_video // num_input_frames
    if tokens_per_frame <= 0:
        return torch.ones(n_video, dtype=torch.bool, device=video_feature.device)

    frames = video_feature.view(num_input_frames, tokens_per_frame, -1)
    keep = torch.zeros((num_input_frames, tokens_per_frame), dtype=torch.bool, device=video_feature.device)
    n_keep = int(round(tokens_per_frame * (1.0 - prune_rate)))
    n_keep = max(0, min(tokens_per_frame, n_keep))

    for g0 in range(0, num_input_frames, group_size):
        g1 = min(g0 + group_size, num_input_frames)
        keep[g0, :] = True
        anchor = frames[g0]
        anchor_norm = torch.nn.functional.normalize(anchor.float(), p=2, dim=-1)
        for f in range(g0 + 1, g1):
            if n_keep <= 0:
                continue
            cur_norm = torch.nn.functional.normalize(frames[f].float(), p=2, dim=-1)
            sim = (cur_norm * anchor_norm).sum(dim=-1)
            local = torch.topk(sim, n_keep, largest=False, sorted=False).indices
            keep[f, local] = True
    return keep.flatten()


def omnidrop_pre_prune(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    audio_token_id: int,
    video_token_id: int,
    attn_logits: torch.Tensor,
    num_input_frames: int,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply OmniDrop Sec. 3.4 pre-LLM intra-modality pruning."""
    device = input_embeds.device
    is_batched = input_embeds.dim() == 3
    if is_batched:
        batch, seq_len, dim = input_embeds.shape
        if batch != 1:
            return input_embeds, torch.ones(seq_len, dtype=torch.bool, device=device), {
                "pre_prune_applied": False,
                "pre_prune_skip_reason": "batch",
            }
        flat_embeds = input_embeds.reshape(seq_len, dim)
        flat_ids = input_ids.reshape(-1)
    else:
        seq_len, dim = input_embeds.shape
        flat_embeds = input_embeds
        flat_ids = input_ids.reshape(-1)

    audio_indices = torch.nonzero(flat_ids == audio_token_id, as_tuple=False).flatten()
    video_indices = torch.nonzero(flat_ids == video_token_id, as_tuple=False).flatten()
    global_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
    audio_pruned = False
    video_pruned = False
    video_skip_reason = None
    video_tokens_per_frame = None

    if audio_indices.numel() > 0 and attn_logits is not None:
        scores = audio_importance_from_attn(attn_logits.to(device), int(audio_indices.numel()))
        keep_n = int(round(float(config.get("audio_keep_ratio", 0.7)) * int(audio_indices.numel())))
        keep_n = max(0, min(int(audio_indices.numel()), keep_n))
        audio_keep = torch.zeros(audio_indices.numel(), dtype=torch.bool, device=device)
        if keep_n > 0:
            audio_keep[torch.topk(scores, keep_n, largest=True, sorted=False).indices] = True
        global_mask[audio_indices] = audio_keep
        audio_pruned = keep_n < int(audio_indices.numel())

    if video_indices.numel() > 0:
        video_feature = flat_embeds[video_indices]
        if num_input_frames <= 0:
            video_skip_reason = "no_frame_count"
        elif int(video_feature.shape[0]) % int(num_input_frames) != 0:
            video_skip_reason = "token_frame_mismatch"
        else:
            video_tokens_per_frame = int(video_feature.shape[0]) // int(num_input_frames)
        video_keep = video_ttm_keep_mask(
            video_feature,
            num_input_frames=int(num_input_frames),
            group_size=int(config.get("video_group_size", 4)),
            prune_rate=float(config.get("video_prune_rate", 0.8)),
        )
        if video_keep.numel() == video_indices.numel():
            global_mask[video_indices] = video_keep
            video_pruned = int(video_keep.sum().item()) < int(video_indices.numel())

    stats = {
        "pre_prune_applied": bool(audio_pruned or video_pruned),
        "pre_audio_has_attention": bool(attn_logits is not None),
        "pre_audio_before": int(audio_indices.numel()),
        "pre_audio_after": int(global_mask[audio_indices].sum().item()) if audio_indices.numel() else 0,
        "pre_video_before": int(video_indices.numel()),
        "pre_video_after": int(global_mask[video_indices].sum().item()) if video_indices.numel() else 0,
        "pre_video_frame_count": int(num_input_frames),
        "pre_video_tokens_per_frame": video_tokens_per_frame,
        "pre_seq_before": int(seq_len),
        "pre_seq_after": int(global_mask.sum().item()),
    }
    if video_skip_reason is not None:
        stats["pre_video_skip_reason"] = video_skip_reason
    if is_batched:
        return flat_embeds[global_mask].view(1, -1, dim), global_mask, stats
    return flat_embeds[global_mask], global_mask, stats


@dataclass
class OmniDropTorchState:
    """Mutable pruning state for one prefill forward pass."""

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    audio_token_id: int
    video_token_id: int
    config: dict[str, Any]
    current_pos: torch.Tensor = field(init=False)
    audio_global: torch.Tensor = field(init=False)
    video_global: torch.Tensor = field(init=False)
    av_global: torch.Tensor = field(init=False)
    query_global: torch.Tensor = field(init=False)
    chunk_by_global: torch.Tensor = field(init=False)
    schedule: PLPSchedule = field(init=False)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = self.input_ids[0]
        device = ids.device
        seq_len = int(ids.numel())
        self.current_pos = torch.arange(seq_len, device=device, dtype=torch.long)

        self.audio_global = torch.nonzero(ids == self.audio_token_id, as_tuple=False).flatten()
        self.video_global = torch.nonzero(ids == self.video_token_id, as_tuple=False).flatten()
        self.av_global = torch.sort(torch.cat([self.audio_global, self.video_global], dim=0)).values

        is_av = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if self.av_global.numel() > 0:
            is_av[self.av_global] = True
            last_av = int(self.av_global.max().item())
            query = torch.arange(seq_len, device=device) > last_av
            query = query & (~is_av)
            if int(query.sum().item()) == 0:
                query = ~is_av
            self.query_global = torch.nonzero(query, as_tuple=False).flatten()
        else:
            self.query_global = torch.empty(0, device=device, dtype=torch.long)

        self.chunk_by_global = self._build_chunk_ids(seq_len, device)
        self.schedule = PLPSchedule(
            p_init=float(self.config.get("p_init", 0.0)),
            p_final=float(self.config.get("p_final", 0.2)),
            L=int(self.config.get("L", self.config.get("num_layers", 28))),
            t_mid=float(self.config.get("t_mid", 0.5)),
            beta=float(self.config.get("beta", 20.0)),
            kind=str(self.config.get("kind", "sigmoid")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True)) and self.av_global.numel() > 0 and self.query_global.numel() > 0

    def _build_chunk_ids(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Assign a stable temporal chunk id to each global sequence position.

        Prefer Qwen's temporal mRoPE coordinate because it is available after
        ``get_rope_index``. Fall back to paper per-chunk AV-token counts if the
        coordinate is degenerate.
        """
        out = torch.zeros(seq_len, device=device, dtype=torch.long)
        if self.av_global.numel() == 0:
            return out

        temporal = self.position_ids[0, 0].to(device=device, dtype=torch.long)
        av_temporal = temporal[self.av_global]
        unique = torch.unique(av_temporal, sorted=True)
        if int(unique.numel()) > 1:
            dense = torch.searchsorted(unique, av_temporal)
            out[self.av_global] = dense.to(torch.long)
            return out

        per_chunk = int(
            self.config.get(
                "av_tokens_per_chunk",
                int(self.config.get("audio_tokens_per_chunk", 50))
                + int(self.config.get("video_tokens_per_chunk", 288)),
            )
        )
        per_chunk = max(per_chunk, 1)
        ranks = torch.arange(self.av_global.numel(), device=device, dtype=torch.long)
        out[self.av_global] = torch.div(ranks, per_chunk, rounding_mode="floor")
        return out

    def select_keep_mask(self, layer_idx: int, attn_weights: torch.Tensor | None) -> torch.Tensor:
        """Return a bool mask over the current sequence after layer ``layer_idx``.

        ``attn_weights`` must be post-softmax attention with shape
        ``(batch, heads, seq, seq)``. If attention is unavailable, no pruning is
        performed; this makes the integration fail-soft for unsupported kernels.
        """
        n_current = int(self.current_pos.numel())
        keep = torch.ones(n_current, dtype=torch.bool, device=self.current_pos.device)
        av_local_mask = torch.isin(self.current_pos, self.av_global)
        av_local = torch.nonzero(av_local_mask, as_tuple=False).flatten()
        n_av_before = int(av_local.numel())

        if attn_weights is None or n_av_before == 0:
            self.trace.append(self._trace_row(layer_idx, n_current, n_current, n_av_before, n_av_before, 0, "no_attn"))
            return keep

        k = self.schedule.num_to_prune(layer_idx, n_av_before)
        k = max(0, min(int(k), n_av_before))
        if k == 0:
            self.trace.append(self._trace_row(layer_idx, n_current, n_current, n_av_before, n_av_before, 0, "schedule"))
            return keep

        query_local_mask = torch.isin(self.current_pos, self.query_global)
        query_local = torch.nonzero(query_local_mask, as_tuple=False).flatten()
        if int(query_local.numel()) == 0:
            self.trace.append(self._trace_row(layer_idx, n_current, n_current, n_av_before, n_av_before, 0, "no_query"))
            return keep

        # Mean over heads, then over text-query rows: Eq. 5.
        attn = attn_weights[0].float().mean(dim=0)
        scores = attn.index_select(0, query_local).index_select(1, av_local).mean(dim=0)
        prune_rel = self._select_prune_rel(layer_idx, scores, self.current_pos[av_local], k)
        prune_local = av_local[prune_rel]
        keep[prune_local] = False

        n_av_after = n_av_before - int(prune_local.numel())
        self.trace.append(
            self._trace_row(
                layer_idx,
                n_current,
                int(keep.sum().item()),
                n_av_before,
                n_av_after,
                int(prune_local.numel()),
                "tds" if layer_idx >= int(self.config.get("tds_start_layer", 14)) else "topk",
            )
        )
        return keep

    def _select_prune_rel(self, layer_idx: int, scores: torch.Tensor, av_global_pos: torch.Tensor, k: int) -> torch.Tensor:
        if k >= int(scores.numel()):
            return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)

        tds_start = int(self.config.get("tds_start_layer", 14))
        if layer_idx < tds_start:
            return torch.topk(scores, k, largest=False, sorted=False).indices

        n_cand = min(2 * k, int(scores.numel()))
        cand = torch.topk(scores, n_cand, largest=False, sorted=False).indices
        key_rel = int(torch.argmax(scores).item())
        key_chunk = self.chunk_by_global[av_global_pos[key_rel]]
        cand_chunks = self.chunk_by_global[av_global_pos[cand]]
        denom = torch.clamp(self.chunk_by_global[self.av_global].max().float(), min=1.0)
        dist = (cand_chunks.float() - key_chunk.float()).abs() / denom
        boosted = scores[cand] + float(self.config.get("lambda_div", 0.2)) * dist
        return cand[torch.topk(boosted, k, largest=False, sorted=False).indices]

    def apply_keep(self, keep: torch.Tensor) -> None:
        self.current_pos = self.current_pos[keep]

    def _trace_row(
        self,
        layer_idx: int,
        seq_before: int,
        seq_after: int,
        av_before: int,
        av_after: int,
        pruned: int,
        mode: str,
    ) -> dict[str, Any]:
        return {
            "layer": int(layer_idx),
            "mode": mode,
            "seq_before": int(seq_before),
            "seq_after": int(seq_after),
            "av_before": int(av_before),
            "av_after": int(av_after),
            "pruned": int(pruned),
            "ratio": float(self.schedule.pruning_ratio(layer_idx)),
        }
