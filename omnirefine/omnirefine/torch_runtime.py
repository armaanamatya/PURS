"""PyTorch prefill bridge for OmniRefine on Qwen2.5-Omni-style inputs.

Paper: arXiv:2605.12056v1.

This module keeps the cited NumPy implementation as the source of truth and
adds the missing runtime boundary: convert a Qwen2.5-Omni prefill tensor into
``ProbeInputs``, call CPCR+MACC, write merged representatives back to anchor
positions, and return a sequence keep mask.

It intentionally implements a layer-0/prefill bridge.  The paper does not
specify the probe layer L; applying the method after an internal decoder layer
also requires pruning K/V cache state for already-run layers, which is model
version specific and remains outside this bridge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import OmniRefineConfig
from .pipeline import KeepMask, ProbeInputs, compress


@dataclass
class TorchOmniRefineDiagnostics:
    """Runtime diagnostics for the prefill bridge.

    Sec. 3.2 describes compression before LLM prefill; these fields expose the
    concrete token counts and the model-free ``KeepMask`` used to produce the
    returned sequence mask.
    """

    keep: Optional[KeepMask]
    input_tokens: int
    output_tokens: int
    video_tokens: int
    audio_tokens: int
    video_kept: int
    audio_kept: int
    warnings: List[str] = field(default_factory=list)

    @property
    def token_retention(self) -> float:
        return self.output_tokens / max(1, self.input_tokens)

    @property
    def av_retention(self) -> float:
        kept = self.video_kept + self.audio_kept
        total = self.video_tokens + self.audio_tokens
        return kept / max(1, total)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "omnirefine.torch_runtime requires torch. Install torch or use the "
            "model-free NumPy core in omnirefine.pipeline."
        ) from exc
    return torch


def _as_single_batch(input_embeds, input_ids):
    """Validate the batch shape used by the runtime bridge.

    [UNSPECIFIED] The paper reports Qwen2.5-Omni inference but does not define
    batched compression semantics.  Using: batch size 1, matching the existing
    local OmniZip/OmniDrop inference paths.  Alternatives: run independent
    per-sample compression and pad, or flatten batches with per-sample masks.
    """
    if input_embeds.dim() == 2:
        input_embeds = input_embeds.unsqueeze(0)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_embeds.dim() != 3 or input_ids.dim() != 2:
        raise ValueError("expected input_embeds (B,L,D) and input_ids (B,L)")
    if input_embeds.shape[:2] != input_ids.shape:
        raise ValueError("input_embeds and input_ids must share B,L")
    if input_embeds.shape[0] != 1:
        raise ValueError("OmniRefine prefill bridge currently supports batch size 1")
    return input_embeds, input_ids


def _infer_grid_hw(
    tokens_per_frame: int,
    video_grid_thw=None,
    spatial_merge_size: Optional[int] = None,
    video_grid_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    """Infer the frame-wise 2D grid required by the TSST video branch.

    Sec. 3.4 reorganizes video tokens into frame-wise 2D grids using original
    spatial position encoding.  Qwen supplies that as ``video_grid_thw`` before
    spatial merge; when it is absent, the square-ish factorization fallback is
    [UNSPECIFIED] and should be replaced by explicit grid metadata.
    """
    if video_grid_hw is not None:
        h, w = int(video_grid_hw[0]), int(video_grid_hw[1])
        if h * w != tokens_per_frame:
            raise ValueError("video_grid_hw product must equal tokens_per_frame")
        return h, w

    if video_grid_thw is not None and spatial_merge_size is not None:
        thw = video_grid_thw[0].detach().cpu().tolist()
        h = int(thw[1]) // int(spatial_merge_size)
        w = int(thw[2]) // int(spatial_merge_size)
        if h * w == tokens_per_frame:
            return h, w

    root = int(np.sqrt(tokens_per_frame))
    for h in range(root, 0, -1):
        if tokens_per_frame % h == 0:
            return h, tokens_per_frame // h
    return 1, tokens_per_frame


def _rank_chunks(n_items: int, n_chunks: int) -> np.ndarray:
    """Fallback native chunk ids by rank.

    Eq. 4 uses native temporal neighborhoods ``c_v``/``c_a``.  If the caller
    does not pass those ids from Qwen's temporal positions, this fallback
    creates equal-rank native chunks so CPCR still has a coarse temporal prior.
    """
    n_chunks = max(1, min(int(n_chunks), max(1, n_items)))
    edges = np.linspace(0, n_items, n_chunks + 1)
    out = np.zeros(n_items, dtype=np.int64)
    for i in range(n_chunks):
        out[int(edges[i]): int(edges[i + 1])] = i
    return out


def _importance_from_attention(attn_logits, audio_indices, n_audio: int):
    """Build audio saliency from attention, matching Sec. 3.4.

    Sec. 3.4 says dominant audio tokens are selected by fused attention-based
    importance, but does not specify the fusion.  Using: mean over heads and
    sum incoming attention/logit mass per key token.  Alternatives: query-only
    attention from text tokens, last-layer attention, or normalized attention
    after softmax.
    """
    if attn_logits is None:
        return np.ones(n_audio, dtype=np.float64)

    torch = _require_torch()
    x = attn_logits.detach()
    if x.dim() == 4:
        x = x[0].float().mean(dim=0)  # (B,H,T,T) -> (T,T)
    elif x.dim() == 3:
        x = x.float().mean(dim=0)     # (H,T,T) -> (T,T)
    else:
        x = x.float()

    if x.dim() == 2:
        score = x.sum(dim=0)
    elif x.dim() == 1:
        score = x
    else:
        return np.ones(n_audio, dtype=np.float64)

    if score.numel() == n_audio:
        sal = score
    elif score.numel() >= int(audio_indices.max().item()) + 1:
        sal = score.index_select(0, audio_indices.to(score.device))
    else:
        pad = torch.ones(n_audio, dtype=score.dtype, device=score.device)
        pad[: min(n_audio, score.numel())] = score[: min(n_audio, score.numel())]
        sal = pad
    return sal.detach().cpu().numpy().astype(np.float64)


def build_probe_inputs_from_qwen_prefill(
    input_embeds,
    input_ids,
    audio_token_id: int,
    video_token_id: int,
    num_input_frames: int,
    cfg: Optional[OmniRefineConfig] = None,
    *,
    attn_logits=None,
    video_grid_thw=None,
    spatial_merge_size: Optional[int] = None,
    video_grid_hw: Optional[Tuple[int, int]] = None,
    frame_native_chunk: Optional[np.ndarray] = None,
    audio_native_chunk: Optional[np.ndarray] = None,
    audio_saliency: Optional[np.ndarray] = None,
) -> ProbeInputs:
    """Convert Qwen2.5-Omni prefill tensors into ``ProbeInputs``.

    Sec. 3.3 requires frame/audio hidden states plus native temporal chunk ids;
    Sec. 3.4 requires frame-wise video grids and audio saliency.  This function
    extracts those fields from the already-merged prefill embedding sequence.
    """
    torch = _require_torch()
    cfg = cfg or OmniRefineConfig(layer_probe=0)
    input_embeds, input_ids = _as_single_batch(input_embeds, input_ids)

    ids = input_ids[0]
    hidden = input_embeds[0]
    video_indices = torch.nonzero(ids == int(video_token_id), as_tuple=False).flatten()
    audio_indices = torch.nonzero(ids == int(audio_token_id), as_tuple=False).flatten()
    n_video = int(video_indices.numel())
    n_audio = int(audio_indices.numel())
    if n_video == 0 or n_audio == 0:
        raise ValueError("OmniRefine requires both video and audio tokens")
    if num_input_frames <= 0 or n_video % int(num_input_frames) != 0:
        raise ValueError("video tokens must divide evenly by num_input_frames")

    tokens_per_frame = n_video // int(num_input_frames)
    grid_h, grid_w = _infer_grid_hw(
        tokens_per_frame,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        video_grid_hw=video_grid_hw,
    )

    video_hidden = hidden.index_select(0, video_indices).detach().float().cpu().numpy()
    video_ids = video_indices.detach().cpu().numpy().astype(np.int64)
    video_hidden = video_hidden.reshape(int(num_input_frames), grid_h, grid_w, -1)
    video_ids = video_ids.reshape(int(num_input_frames), grid_h, grid_w)
    video_grid_hidden = [video_hidden[i] for i in range(int(num_input_frames))]
    video_grid_ids = [video_ids[i] for i in range(int(num_input_frames))]

    audio_hidden = hidden.index_select(0, audio_indices).detach().float().cpu().numpy()
    audio_global_ids = audio_indices.detach().cpu().numpy().astype(np.int64).tolist()

    if frame_native_chunk is None or audio_native_chunk is None:
        n_chunks = max(1, int(round(num_input_frames / max(1, cfg.s_v_max))))
        frame_native_chunk = _rank_chunks(int(num_input_frames), n_chunks)
        audio_native_chunk = _rank_chunks(n_audio, n_chunks)
    if audio_saliency is None:
        audio_saliency = _importance_from_attention(attn_logits, audio_indices, n_audio)

    return ProbeInputs(
        video_grid_hidden=video_grid_hidden,
        video_grid_ids=video_grid_ids,
        frame_native_chunk=np.asarray(frame_native_chunk, dtype=np.int64),
        audio_hidden=audio_hidden,
        audio_global_ids=audio_global_ids,
        audio_native_chunk=np.asarray(audio_native_chunk, dtype=np.int64),
        audio_saliency=np.asarray(audio_saliency, dtype=np.float64),
    )


def apply_keep_to_prefill(input_embeds, keep: KeepMask, prunable_ids=None):
    """Scatter merged OmniRefine reps and return ``(compressed, global_mask)``.

    Sec. 3.4 Eq. 11 updates retained anchor representations; Sec. 3.2 then
    continues inference on the compressed sequence.  This function performs the
    tensor-side scatter and sequence slicing for a prefill-level integration.
    """
    torch = _require_torch()
    if input_embeds.dim() == 2:
        input_embeds = input_embeds.unsqueeze(0)
    if input_embeds.shape[0] != 1:
        raise ValueError("apply_keep_to_prefill currently supports batch size 1")

    out = input_embeds.clone()
    seq_len = int(out.shape[1])
    device = out.device
    dtype = out.dtype
    global_mask = torch.ones(seq_len, dtype=torch.bool, device=device)

    keep_ids = set(int(i) for i in keep.video_keep_ids) | set(int(i) for i in keep.audio_keep_ids)
    if prunable_ids is not None:
        for pos in prunable_ids:
            global_mask[int(pos.item())] = False
    else:
        # [UNSPECIFIED] KeepMask intentionally stores retained anchors, not the
        # full original AV id set.  Without prunable_ids we can only guarantee
        # anchor scatter, so no additional tokens are dropped.
        for pos in keep_ids:
            global_mask[int(pos)] = True
    for pos in keep_ids:
        global_mask[int(pos)] = True

    for pos, rep in keep.video_reps.items():
        out[0, int(pos)] = torch.as_tensor(rep, dtype=dtype, device=device)
    for pos, rep in keep.audio_reps.items():
        out[0, int(pos)] = torch.as_tensor(rep, dtype=dtype, device=device)

    return out[:, global_mask, :], global_mask


def omnirefine_qwen_prefill(
    input_embeds,
    input_ids,
    audio_token_id: int,
    video_token_id: int,
    num_input_frames: int,
    cfg: Optional[OmniRefineConfig] = None,
    *,
    attn_logits=None,
    video_grid_thw=None,
    spatial_merge_size: Optional[int] = None,
    video_grid_hw: Optional[Tuple[int, int]] = None,
    frame_native_chunk: Optional[np.ndarray] = None,
    audio_native_chunk: Optional[np.ndarray] = None,
    audio_saliency: Optional[np.ndarray] = None,
):
    """Run OmniRefine on a Qwen2.5-Omni prefill embedding sequence.

    Returns ``(compressed_embeds, global_mask, diagnostics)``.  The signature is
    intentionally close to the local OmniZip helper so Qwen integration can
    slice ``attention_mask`` and ``position_ids`` with the returned mask.
    """
    torch = _require_torch()
    cfg = cfg or OmniRefineConfig(layer_probe=0)
    input_embeds, input_ids = _as_single_batch(input_embeds, input_ids)
    ids = input_ids[0]
    video_tokens = int(torch.sum(ids == int(video_token_id)).item())
    audio_tokens = int(torch.sum(ids == int(audio_token_id)).item())
    warnings: List[str] = []

    if (
        video_tokens == 0
        or audio_tokens == 0
        or num_input_frames < cfg.s_v_min
        or audio_tokens < cfg.s_a_min
    ):
        global_mask = torch.ones(input_embeds.shape[1], dtype=torch.bool, device=input_embeds.device)
        warnings.append("sequence too short or missing a modality; returned identity mask")
        diag = TorchOmniRefineDiagnostics(
            keep=None,
            input_tokens=int(input_embeds.shape[1]),
            output_tokens=int(input_embeds.shape[1]),
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            video_kept=video_tokens,
            audio_kept=audio_tokens,
            warnings=warnings,
        )
        return input_embeds, global_mask, diag

    probe = build_probe_inputs_from_qwen_prefill(
        input_embeds=input_embeds,
        input_ids=input_ids,
        audio_token_id=audio_token_id,
        video_token_id=video_token_id,
        num_input_frames=num_input_frames,
        cfg=cfg,
        attn_logits=attn_logits,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        video_grid_hw=video_grid_hw,
        frame_native_chunk=frame_native_chunk,
        audio_native_chunk=audio_native_chunk,
        audio_saliency=audio_saliency,
    )

    try:
        keep = compress(probe, cfg)
    except ValueError as exc:
        global_mask = torch.ones(input_embeds.shape[1], dtype=torch.bool, device=input_embeds.device)
        warnings.append(f"CPCR infeasible under current bounds; returned identity mask: {exc}")
        diag = TorchOmniRefineDiagnostics(
            keep=None,
            input_tokens=int(input_embeds.shape[1]),
            output_tokens=int(input_embeds.shape[1]),
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            video_kept=video_tokens,
            audio_kept=audio_tokens,
            warnings=warnings,
        )
        return input_embeds, global_mask, diag
    prunable_ids = torch.nonzero(
        (ids == int(video_token_id)) | (ids == int(audio_token_id)),
        as_tuple=False,
    ).flatten()
    compressed, global_mask = apply_keep_to_prefill(input_embeds, keep, prunable_ids=prunable_ids)
    diag = TorchOmniRefineDiagnostics(
        keep=keep,
        input_tokens=int(input_embeds.shape[1]),
        output_tokens=int(compressed.shape[1]),
        video_tokens=video_tokens,
        audio_tokens=audio_tokens,
        video_kept=keep.n_video_kept,
        audio_kept=keep.n_audio_kept,
        warnings=warnings,
    )
    return compressed, global_mask, diag
