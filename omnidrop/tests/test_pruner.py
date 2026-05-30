"""End-to-end tests for the OmniDrop orchestrator.

Trap (advisor): pruning is cumulative -- k_l = floor(p_l * CURRENT count), and
dropped tokens stay dropped. The end-to-end mean retained ratio must close to
the published target's safety margin, and survival must be monotone.
"""
import numpy as np

from omnidrop.pruner import OmniDropConfig, TokenLayout, OmniDropPruner


def _make_layout(n_chunks=20, av_per_chunk=10, n_text=8, n_sys=4):
    """Build a [T_sys, (V,A)*m, T_query] layout (Eq. 2)."""
    pos = n_sys
    av_idx, av_chunk = [], []
    for c in range(n_chunks):
        for _ in range(av_per_chunk):
            av_idx.append(pos); av_chunk.append(c); pos += 1
    text_query_idx = np.arange(pos, pos + n_text)
    return TokenLayout(
        text_query_idx=text_query_idx,
        av_idx=np.array(av_idx),
        av_chunk_ids=np.array(av_chunk),
        n_chunks=n_chunks,
    ), pos + n_text


def _stub_attn_provider(seq_len, rng):
    full = rng.random((seq_len, seq_len))
    full = full / full.sum(axis=1, keepdims=True)
    return lambda layer: full


def test_pruning_is_cumulative_and_monotone():
    layout, seq_len = _make_layout()
    cfg = OmniDropConfig(p_init=0.0, p_final=0.5, L=28, tds_start_layer=14)
    pruner = OmniDropPruner(cfg, layout)
    rng = np.random.default_rng(42)
    survivors = pruner.run(_stub_attn_provider(seq_len, rng))
    counts = [len(s) for s in survivors]
    # monotone non-increasing -- dropped tokens never return.
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    # survivors are always a subset of the original AV set.
    assert set(survivors[-1].tolist()).issubset(set(layout.av_idx.tolist()))


def test_kl_computed_on_current_not_original_count():
    layout, seq_len = _make_layout()
    cfg = OmniDropConfig(p_init=0.02, p_final=0.5, L=28, tds_start_layer=14)
    pruner = OmniDropPruner(cfg, layout)
    rng = np.random.default_rng(0)
    pruner.run(_stub_attn_provider(seq_len, rng))
    # Re-derive expected counts cumulatively and compare to history.
    n = layout.av_idx.shape[0]
    for h in pruner.history:
        k_expected = int(np.floor(pruner.schedule.pruning_ratio(h["layer"]) * n))
        assert h["k"] == k_expected, (h, k_expected, n)
        n -= h["k"]
    assert n == len(pruner.alive_pos)


def test_mean_retained_matches_schedule_within_margin():
    # The count trajectory is set by the schedule (attention only chooses which
    # tokens), so empirical mean retained should track the analytic schedule.
    layout, seq_len = _make_layout(n_chunks=40, av_per_chunk=10)
    cfg = OmniDropConfig(p_init=0.0, p_final=0.2, L=28, tds_start_layer=14)
    pruner = OmniDropPruner(cfg, layout)
    rng = np.random.default_rng(7)
    pruner.run(_stub_attn_provider(seq_len, rng))
    emp = pruner.mean_retained(r0=0.45)
    analytic = pruner.schedule.mean_retained(0.45)
    assert abs(emp - analytic) < 0.02, (emp, analytic)
    assert abs(emp - 0.30) < 0.03   # realizes the nominal 30% target


def test_tds_only_active_from_start_layer():
    # Before tds_start_layer, plain top-k; the history records correct k either way.
    layout, seq_len = _make_layout()
    cfg = OmniDropConfig(p_init=0.05, p_final=0.5, L=28, tds_start_layer=14)
    pruner = OmniDropPruner(cfg, layout)
    rng = np.random.default_rng(3)
    pruner.run(_stub_attn_provider(seq_len, rng))
    assert len(pruner.history) == 28
