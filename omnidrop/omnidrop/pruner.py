"""OmniDrop orchestrator: cumulative, query-guided, layer-wise pruning.

Ties together the PLP schedule (Sec. 3.1), query-guided importance (Sec. 3.2),
and TDS (Sec. 3.3). The pruning is *cumulative*: at each layer the budget k_l is
computed on the number of AV tokens currently alive, and dropped tokens never
return (Appendix E: r_l = r_0 * prod(1 - p_j)).

This module is model-agnostic. In a real Qwen2.5-Omni integration the
``attn_provider`` would yield each decoder layer's post-softmax attention and
the keep-mask would be used to slice hidden states / KV-cache / position ids;
here we operate on indices so the logic is unit-testable without the full model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .importance import query_guided_importance
from .schedule import PLPSchedule
from .tds import tds_select_to_prune, topk_lowest_to_prune


@dataclass
class TokenLayout:
    """Immutable per-token metadata for the input sequence (Eq. 2 layout).

    Positions are absolute indices into the full sequence
    [T_sys, (V_1,A_1), ..., (V_m,A_m), T_query].
    """

    text_query_idx: np.ndarray   # positions of text-query tokens (the guide)
    av_idx: np.ndarray           # positions of audiovisual tokens (prunable)
    av_chunk_ids: np.ndarray     # chunk index per AV token (same order as av_idx)
    n_chunks: int                # total chunks m (for max(C) stability)

    def __post_init__(self):
        self.text_query_idx = np.asarray(self.text_query_idx, dtype=np.int64)
        self.av_idx = np.asarray(self.av_idx, dtype=np.int64)
        self.av_chunk_ids = np.asarray(self.av_chunk_ids, dtype=np.int64)
        if self.av_idx.shape != self.av_chunk_ids.shape:
            raise ValueError("av_idx and av_chunk_ids must align")


@dataclass
class OmniDropConfig:
    p_init: float
    p_final: float
    L: int
    t_mid: float = 0.5
    beta: float = 20.0
    lambda_div: float = 0.2
    tds_start_layer: int = 14    # 14 for 7B, 19 for 3B (Sec. 4)
    kind: str = "sigmoid"


class OmniDropPruner:
    """Applies OmniDrop layer-by-layer over the audiovisual token set."""

    def __init__(self, config: OmniDropConfig, layout: TokenLayout):
        self.cfg = config
        self.layout = layout
        self.schedule = PLPSchedule(
            config.p_init, config.p_final, config.L,
            config.t_mid, config.beta, config.kind,
        )
        # Mutable survival state: positions of AV tokens still alive, and their
        # chunk ids in the same order.
        self.alive_pos = layout.av_idx.copy()
        self.alive_chunk = layout.av_chunk_ids.copy()
        self.history: list[dict] = []

    @property
    def n_alive(self) -> int:
        return int(self.alive_pos.shape[0])

    def step(self, layer: int, attn: np.ndarray) -> np.ndarray:
        """Prune at one decoder layer given its post-softmax attention.

        Args:
            layer: 0-based decoder layer index.
            attn: (n_heads, seq, seq) or (seq, seq) attention for the *current*
                  (already-reduced) sequence. Column indices must refer to the
                  original absolute positions -- callers pass the full-width
                  attention and we index by ``alive_pos`` / ``text_query_idx``.

        Returns:
            The array of AV positions that remain alive after this layer.
        """
        k = self.schedule.num_to_prune(layer, self.n_alive)  # Eq. 6 on CURRENT count
        if k <= 0 or self.n_alive == 0:
            self.history.append({"layer": layer, "k": 0, "n_alive": self.n_alive})
            return self.alive_pos

        # Eq. 5: importance of each currently-alive AV token.
        S = query_guided_importance(attn, self.layout.text_query_idx, self.alive_pos)

        if layer >= self.cfg.tds_start_layer:
            local_prune = tds_select_to_prune(
                S, self.alive_chunk, k,
                lambda_div=self.cfg.lambda_div,
                max_chunk=self.layout.n_chunks - 1,
            )
        else:
            local_prune = topk_lowest_to_prune(S, k)

        keep_mask = np.ones(self.n_alive, dtype=bool)
        keep_mask[local_prune] = False
        self.alive_pos = self.alive_pos[keep_mask]
        self.alive_chunk = self.alive_chunk[keep_mask]
        self.history.append(
            {"layer": layer, "k": int(k), "n_alive": self.n_alive,
             "ratio": self.schedule.pruning_ratio(layer)}
        )
        return self.alive_pos

    def run(self, attn_provider) -> list[np.ndarray]:
        """Run the full L-layer loop.

        Args:
            attn_provider: callable ``layer -> attn`` returning the post-softmax
                attention for that layer (full sequence width). In tests this is
                a stub; in production it is a hook on the decoder forward pass.

        Returns:
            List (length L) of surviving AV-position arrays after each layer.
        """
        survivors = []
        for layer in range(self.cfg.L):
            survivors.append(self.step(layer, attn_provider(layer)).copy())
        return survivors

    def mean_retained(self, n_av_total: int | None = None, r0: float | None = None) -> float:
        """Empirical mean retained ratio across layers from the run history.

        Uses ``r0`` (post intra-modality, default 0.45) as the entering ratio so
        the result is directly comparable to the paper's reported average
        retention. If no run has happened, falls back to the analytic schedule.
        """
        r0 = 0.45 if r0 is None else r0
        if not self.history:
            return self.schedule.mean_retained(r0)
        n_total = n_av_total if n_av_total is not None else self.layout.av_idx.shape[0]
        ratios = []
        n = n_total
        for h in self.history:
            ratios.append(r0 * n / n_total)  # ratio entering this layer
            n = h["n_alive"]
        return float(np.mean(ratios))
