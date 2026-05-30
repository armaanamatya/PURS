# OmniDrop (minimal reproduction)

Citation-anchored reproduction of **"OmniDrop: Layer-wise Token Pruning for
Omni-modal LLMs via Query-Guidance"** (arXiv 2605.14458, Park et al., Samsung
Research).

OmniDrop is a **training-free** token-pruning framework for Omni-modal LLMs
(Qwen2.5-Omni). Unlike input-level methods (OmniZip, DASH) that prune on
audio↔video similarity, OmniDrop prunes **progressively inside the decoder
layers** using **text-query→audiovisual attention** as the importance signal,
plus a **temporal diversity score** to preserve global context.

## What this repo implements (paper → code)

| Paper | Module | Function |
|---|---|---|
| §3.1, Eqs. 3–4, App. E | `omnidrop/schedule.py` | `PLPSchedule`, `calibrate_pfinal*` |
| §3.2, Eq. 5 | `omnidrop/importance.py` | `query_guided_importance` |
| §3.3, Algorithm 1 | `omnidrop/tds.py` | `tds_select_to_prune` |
| §3.4 | `omnidrop/intra_modality.py` | `prune_audio_by_attention`, `prune_video_ttm` |
| orchestration | `omnidrop/pruner.py` | `OmniDropPruner` (cumulative layer loop) |

## The method in four steps (per decoder layer `l`)

1. **PLP ratio** `p_l = p_init + (p_final − p_init)·σ(β·(l/L − t_mid))` (sigmoid, β=20, t_mid=0.5).
2. **Budget** `k_l = floor(p_l · current_AV_token_count)` — *cumulative*; dropped tokens stay dropped.
3. **Importance** `S_j = mean over text-query tokens of attention to AV token j`.
4. **TDS** (from layer 14 on 7B / 19 on 3B): re-rank the bottom `2k_l` candidates by adding `λ_div·D` where `D` is normalized temporal distance to the key chunk, then drop the bottom `k_l`.

Intra-modality pruning (audio top-attention → 70%; video Dycoke-TTM → 40%) runs
once before the LLM, giving ~45% retention entering layer 0.

## Quickstart

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest tests/ -q          # 23 tests
PYTHONPATH=. python -c "from omnidrop import calibrate_pfinal_paper_approx as f; print(f())"  # ~0.146
```

```python
import numpy as np
from omnidrop import OmniDropConfig, TokenLayout, OmniDropPruner

cfg = OmniDropConfig(p_init=0.0, p_final=0.2, L=28, tds_start_layer=14)  # 7B / 30%
layout = TokenLayout(text_query_idx=..., av_idx=..., av_chunk_ids=..., n_chunks=m)
pruner = OmniDropPruner(cfg, layout)
survivors = pruner.run(lambda layer: attention_at(layer))   # post-softmax attn
print(pruner.mean_retained(r0=0.45))                        # ~0.30
```

## Scope

Minimal mode: the pruning **mechanism** is fully implemented and unit-tested in
pure NumPy. It is model-agnostic — the same index/mask logic plugs into the
Qwen2.5-Omni decoder forward pass on torch tensors. **Not** included: the Qwen
model itself, KV-cache rewiring, FlashAttention plumbing, or the VideoMME /
WorldSense / AVUT benchmark harnesses and their reported numbers.

See `REPRODUCTION_NOTES.md` for every assumption, ambiguity, and a notable
discrepancy found in the paper's Appendix-E calibration.
