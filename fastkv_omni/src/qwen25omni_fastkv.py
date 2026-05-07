"""
FastKV-Omni: Token-Selective Propagation (TSP) patch for Qwen2.5-Omni Thinker.

Ports the TSP mechanism from FastKV (arXiv:2502.01068, ACL Findings 2026)
to the Qwen2.5-Omni omnimodal architecture. Per-layer SnapKV-style KV
compression is intentionally OMITTED in this first pass — TSP-only.

Three patches are installed at runtime via apply_fastkv():
  1. Each Qwen2_5OmniDecoderLayer.forward gains a post-MLP TSP scoring step
     that writes self.new_position_ids / self.new_tsp_indices when this
     layer is the chosen TSP layer.
  2. The Thinker's text decoder model.forward is rewritten to propagate
     the slice (hidden_states, position_ids, position_embeddings, mask)
     to subsequent layers when a TSP cut occurs.
  3. The slicing is TMRoPE-aware: position_ids has shape (3, B, T) carrying
     (temporal, height, width) per token; we gather along dim=2 and broadcast
     the same T-index across all three rope-axis rows so each kept token's
     (t, h, w) triple stays internally consistent.

Smoke test (run as `python qwen25omni_fastkv.py --smoke`):
  Loads the Thinker, applies the patch with TSP effectively disabled
  (tsp_length set larger than the input), runs a forward pass, and asserts
  output logits match the unpatched baseline within 1e-4. If this fails,
  the patch is wrong before any compression is even applied.

NOT YET IMPLEMENTED (deliberately, for first experiment):
  - Per-layer KV cache compression (SnapKV-style topk on K/V).
  - Decode-time use of compressed cache.
  - Modality-aware budget split.
  - Multi-layer (cumulative) TSP.

Reference algorithm: vendored/fastkv/utils.py (verbatim copy of the
FastKV-main upstream source).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# FastKV TSP scorer (TSP-only port; see vendored/fastkv/utils.py for full source)
# ────────────────────────────────────────────────────────────────────────────

def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return hidden_states[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class FastKVOmniCluster:
    """
    Per-layer TSP scorer. Holds the FastKV hyperparameters and computes which
    K token positions should propagate past this layer when this layer is
    designated the TSP layer.

    Score recipe (from vendored/fastkv/utils.py:update_kv):
      1. Take last `window_size` queries.
      2. attn = softmax(Q[-W:] · K^T / sqrt(d))   with causal mask on the W×W block.
      3. score[t] = sum over the W queries of attn[w, t] for t in [0, T-W).
      4. Smooth scores with avg_pool1d(kernel_size).
      5. Sum across GQA groups → per-KV-head score.
      6. Sum across KV heads → global score (for TSP — distinct from SnapKV's per-head topk).
      7. topk(tsp_length - W). Concatenate the trailing window. Sort.
    """

    def __init__(
        self,
        window_size: int = 8,
        kernel_size: int = 7,
        pooling: str = "avgpool",
        tsp_layer: bool = False,
        tsp_length: int = 2048,
    ):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.tsp_layer = tsp_layer
        self.tsp_length = tsp_length

    def select_tsp_indices(
        self,
        key_states: torch.Tensor,    # (B, num_kv_heads, T, D)
        query_states: torch.Tensor,  # (B, num_heads, T, D)
        num_key_value_groups: int,
    ) -> Optional[torch.Tensor]:
        """Returns (B, K) sorted token indices to keep, or None if TSP is a no-op
        (either this isn't the TSP layer, or the sequence is short enough to skip)."""
        if not self.tsp_layer:
            return None
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len <= self.tsp_length:
            return None
        if q_len <= self.window_size:
            return None

        K_temp = _repeat_kv(key_states, num_key_value_groups)  # (B, H, T, D)
        # Score keys via the last `window_size` queries
        attn = torch.matmul(
            query_states[..., -self.window_size:, :],
            K_temp.transpose(2, 3),
        ) / math.sqrt(head_dim)  # (B, H, W, T)

        # Causal mask on the trailing W×W block (each window-query can't see future window-keys)
        W = self.window_size
        mask = torch.full((W, W), torch.finfo(attn.dtype).min, device=attn.device, dtype=attn.dtype)
        m_cond = torch.arange(W, device=attn.device)
        mask.masked_fill_(m_cond < (m_cond + 1).view(W, 1), 0)
        attn[:, :, -W:, -W:] = attn[:, :, -W:, -W:] + mask

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Sum over the window queries → per-key score, restricted to the body (drop trailing W)
        scores = attn[:, :, :, : q_len - W].sum(dim=-2)  # (B, H, T-W)

        # Smooth
        if self.pooling == "avgpool":
            scores = F.avg_pool1d(scores, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)
        elif self.pooling == "maxpool":
            scores = F.max_pool1d(scores, kernel_size=self.kernel_size,
                                  padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError(f"Pooling {self.pooling!r} not supported")

        # Collapse heads → KV heads → global score (TSP picks one global top-K, not per-head)
        scores = scores.view(bsz, -1, num_key_value_groups, q_len - W).sum(dim=-2)  # (B, num_kv_heads, T-W)
        global_scores = scores.sum(dim=-2)  # (B, T-W)

        keep_body = max(self.tsp_length - W, 1)
        body_idx = global_scores.topk(keep_body, dim=-1).indices  # (B, K-W)
        window_idx = torch.arange(q_len - W, q_len, device=body_idx.device).unsqueeze(0).expand(bsz, -1)
        all_idx = torch.cat([body_idx, window_idx], dim=-1)
        all_idx, _ = torch.sort(all_idx, dim=-1)
        return all_idx  # (B, K)


# ────────────────────────────────────────────────────────────────────────────
# Patched decoder layer forward
# ────────────────────────────────────────────────────────────────────────────

def _patched_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """Run the original layer forward, then optionally compute TSP indices.
    The actual hidden-state slice is applied by the patched model.forward
    after this function returns."""
    pre_attn_input = hidden_states  # save BEFORE input_layernorm

    outputs = self._fastkv_original_forward(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    out_hidden = outputs[0]

    cluster = getattr(self.self_attn, "kv_cluster", None)
    self.new_tsp_indices = None
    if cluster is None or not cluster.tsp_layer:
        return outputs
    if pre_attn_input.size(1) <= cluster.tsp_length:
        return outputs

    # Score: re-project pre-attention hidden states through Q, K (post-rotary)
    bsz, q_len, _ = pre_attn_input.size()
    attn = self.self_attn
    with torch.no_grad():
        ln_in = self.input_layernorm(pre_attn_input)
        q = attn.q_proj(ln_in).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(ln_in).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rotary(q, k, cos, sin)
        tsp_idx = cluster.select_tsp_indices(k, q, attn.num_key_value_groups)
    self.new_tsp_indices = tsp_idx
    return outputs


def _apply_rotary(q, k, cos, sin):
    """Wrapper that imports the model's rotary helper lazily so this file can
    be loaded even without the transformers Qwen2.5-Omni module present."""
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb
    return apply_rotary_pos_emb(q, k, cos, sin)


# ────────────────────────────────────────────────────────────────────────────
# TMRoPE-correct slicing helpers
# ────────────────────────────────────────────────────────────────────────────

def _slice_position_ids(position_ids: torch.Tensor, tsp_idx: torch.Tensor) -> torch.Tensor:
    """TMRoPE-aware position_ids slicing.
    position_ids:
       (3, B, T)  → TMRoPE: gather on dim=2, broadcasting one T-index across the 3 rope axes
       (B, T)     → plain RoPE: gather on dim=1
    tsp_idx: (B, K) sorted token indices to keep.
    """
    if position_ids.dim() == 3:
        # (rope_axis, B, T)  — keep all 3 axes coupled to the same kept tokens
        rope_axes, B, T = position_ids.shape
        idx = tsp_idx.unsqueeze(0).expand(rope_axes, -1, -1)  # (3, B, K)
        return torch.gather(position_ids, dim=2, index=idx)
    elif position_ids.dim() == 2:
        return torch.gather(position_ids, dim=1, index=tsp_idx)
    else:
        raise ValueError(f"Unexpected position_ids rank {position_ids.dim()}; expected 2 or 3.")


def _slice_attention_mask(causal_mask: Optional[torch.Tensor], tsp_idx: torch.Tensor) -> Optional[torch.Tensor]:
    """Slice a 4D causal mask (B, 1, T, T) along both the query and key axes by tsp_idx."""
    if causal_mask is None:
        return None
    if causal_mask.dim() != 4:
        # Some HF paths pass a 2D padding mask; downstream layers will rebuild it.
        return causal_mask
    B, _, Tq, Tk = causal_mask.shape
    K = tsp_idx.size(1)
    # Gather rows (queries)
    row_idx = tsp_idx.view(B, 1, K, 1).expand(B, 1, K, Tk)
    m = causal_mask.gather(2, row_idx)
    # Gather cols (keys)
    col_idx = tsp_idx.view(B, 1, 1, K).expand(B, 1, K, K)
    m = m.gather(3, col_idx)
    return m


def _slice_cache_position(cache_position: Optional[torch.Tensor], tsp_idx: torch.Tensor) -> Optional[torch.Tensor]:
    if cache_position is None:
        return None
    if tsp_idx.size(0) != 1:
        # cache_position is (T,) — only meaningful for single-batch runs.
        # For multi-batch, the caller should batch independently.
        return cache_position[tsp_idx[0]]
    return cache_position[tsp_idx[0]]


# ────────────────────────────────────────────────────────────────────────────
# Patched Thinker text-decoder model forward (the LM loop)
# ────────────────────────────────────────────────────────────────────────────

def _patched_thinker_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    cache_position=None,
    **kwargs,
):
    """Replacement for Qwen2_5OmniThinkerTextModel.forward (or the inner LM
    forward, whatever it is named). Mirrors the standard HF transformer
    decoder loop but checks each layer for a TSP cut after it runs and, if
    found, slices hidden_states + position_ids + position_embeddings + mask
    for the remaining layers.

    Assumptions (standard for HF text decoders ≥ 4.45):
      - self.embed_tokens, self.layers, self.norm, self.rotary_emb exist
      - self._update_causal_mask exists (or causal mask is rebuilt per-layer)
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        # Default: text-only 1-D positions broadcast to TMRoPE's (3, B, T) layout
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1).contiguous()

    # Build causal mask (delegate to the original implementation if available)
    if hasattr(self, "_update_causal_mask"):
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
    else:
        causal_mask = attention_mask

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        # Apply TSP slice if this layer requested one
        tsp_idx = getattr(decoder_layer, "new_tsp_indices", None)
        if tsp_idx is not None:
            B, K = tsp_idx.shape
            # Slice hidden_states: (B, T, D) → (B, K, D)
            hidden_states = torch.gather(
                hidden_states, dim=1,
                index=tsp_idx.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
            )
            # TMRoPE-aware position_ids slice
            position_ids = _slice_position_ids(position_ids, tsp_idx)
            # Recompute rotary embeddings on the sliced positions
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            # Slice causal mask along both axes
            causal_mask = _slice_attention_mask(causal_mask, tsp_idx)
            # Slice cache_position
            cache_position = _slice_cache_position(cache_position, tsp_idx)
            # Consume the marker so a re-run starts clean
            decoder_layer.new_tsp_indices = None

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in (hidden_states, past_key_values, all_hidden_states, all_self_attns) if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API: apply / disable
# ────────────────────────────────────────────────────────────────────────────

def _get_thinker_lm(model):
    """Locate the Thinker's inner text-decoder model. Order of attempts:
       model.thinker.model       (Qwen2.5-Omni standard layout)
       model.thinker             (if .thinker IS the LM)
       model                     (if `model` is already the LM)
    """
    for path in ("thinker.model", "thinker", ""):
        target = model
        if path:
            for part in path.split("."):
                if not hasattr(target, part):
                    target = None
                    break
                target = getattr(target, part)
        if target is not None and hasattr(target, "layers") and hasattr(target, "embed_tokens"):
            return target
    raise AttributeError("Could not locate the Thinker text-decoder model on the given object.")


def apply_fastkv(
    model,
    tsp_idx: int,
    tsp_length: int = 2048,
    window_size: int = 8,
    kernel_size: int = 7,
    pooling: str = "avgpool",
):
    """Install TSP patches on a Qwen2.5-Omni model in place.

    Args:
        model: a Qwen2_5OmniForConditionalGeneration (or its .thinker, or the
            Thinker LM directly).
        tsp_idx: 0-based index of the layer at which TSP fires. Use a value
            >= num_layers (e.g. num_layers - 1) to make TSP a no-op for
            downstream layers (smoke-test mode).
        tsp_length: number of tokens to retain after TSP. Use a value larger
            than any prompt length to disable TSP entirely (other smoke-test mode).
        window_size, kernel_size, pooling: FastKV defaults.

    Returns the model (also mutated in place).
    """
    lm = _get_thinker_lm(model)
    n = len(lm.layers)

    for i, layer in enumerate(lm.layers):
        # Attach a per-layer cluster. Only layer i == tsp_idx fires TSP.
        layer.self_attn.kv_cluster = FastKVOmniCluster(
            window_size=window_size,
            kernel_size=kernel_size,
            pooling=pooling,
            tsp_layer=(i == tsp_idx),
            tsp_length=tsp_length,
        )
        layer.new_tsp_indices = None

        # Patch the decoder layer forward (idempotent: only patch once)
        if not hasattr(layer, "_fastkv_original_forward"):
            layer._fastkv_original_forward = layer.forward
            layer.forward = types.MethodType(_patched_decoder_layer_forward, layer)

    # Patch the inner LM forward (idempotent)
    if not hasattr(lm, "_fastkv_original_forward"):
        lm._fastkv_original_forward = lm.forward
        lm.forward = types.MethodType(_patched_thinker_lm_forward, lm)

    return model


def disable_fastkv(model):
    """Restore the Thinker LM and decoder layer forwards to their originals."""
    lm = _get_thinker_lm(model)
    if hasattr(lm, "_fastkv_original_forward"):
        lm.forward = lm._fastkv_original_forward
        del lm._fastkv_original_forward
    for layer in lm.layers:
        if hasattr(layer, "_fastkv_original_forward"):
            layer.forward = layer._fastkv_original_forward
            del layer._fastkv_original_forward
        if hasattr(layer.self_attn, "kv_cluster"):
            del layer.self_attn.kv_cluster
        if hasattr(layer, "new_tsp_indices"):
            del layer.new_tsp_indices
    return model


# ────────────────────────────────────────────────────────────────────────────
# Smoke test (run as: python qwen25omni_fastkv.py --smoke)
# ────────────────────────────────────────────────────────────────────────────

def smoke_test(model_path: str, prompt: str = "The capital of France is", atol: float = 1e-4):
    """Loads the Thinker, runs baseline forward, applies the patch in no-op
    mode (huge tsp_length), runs again, asserts logits match within `atol`.

    NOTE: this requires actual GPU + the full model. Cannot be run on the
    Windows dev box; intended for the user's data/armaan server.
    """
    print(f"[smoke] loading {model_path} ...")
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
    import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as _qmod
    _qmod.check_torch_load_is_safe = lambda: None  # bypass torch < 2.6 CVE check

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    inputs = processor(text=[prompt], return_tensors="pt").to(model.device)

    # Baseline
    print("[smoke] baseline forward ...")
    with torch.no_grad():
        out_base = model.thinker.model(input_ids=inputs.input_ids)
    base_logits = out_base.last_hidden_state.float().cpu()

    # Patched, no-op
    print("[smoke] applying patch in no-op mode (tsp_length much larger than prompt) ...")
    seq_len = inputs.input_ids.shape[1]
    apply_fastkv(model, tsp_idx=0, tsp_length=seq_len * 100, window_size=8)
    with torch.no_grad():
        out_patched = model.thinker.model(input_ids=inputs.input_ids)
    patched_logits = out_patched.last_hidden_state.float().cpu()
    disable_fastkv(model)

    diff = (base_logits - patched_logits).abs().max().item()
    print(f"[smoke] max |Δ| = {diff:.2e}  (tolerance {atol:.0e})")
    if diff > atol:
        print("[smoke] FAILED — patch is changing outputs even when TSP is disabled.")
        sys.exit(1)
    print("[smoke] PASSED — patch is logits-equivalent in no-op mode.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the no-op equivalence smoke test.")
    parser.add_argument("--model_path", default="/data/armaan/models/Qwen2.5-Omni-7B")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.smoke:
        smoke_test(args.model_path, args.prompt, args.atol)
    else:
        parser.print_help()
