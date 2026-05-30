# Reproduction Notes — OmniDrop (arXiv 2605.14458)

Honest record of what is pinned by the paper, what was assumed, and one
numerical discrepancy found in the paper itself. The equation **images** were
absent from the extracted PDF; every formula here was reconstructed from the
surrounding prose + Algorithm 1 + Appendix E, which pin them unambiguously.

## ⚠️ Notable finding: Appendix-E calibration undershoots

The paper derives `p_final ≈ 0.146` (Appendix E.2) to hit a 30% mean retained
ratio for the 7B model (`L=28, p_init=0, t_mid=0.5, r_0=0.45`), then
"conservatively" adopts `p_final = 0.2`, claiming it keeps the mean *strictly
below* 0.30.

Computing the **exact** cumulative mean `R̄ = (1/L)·Σ r_l` with `r_l = r_0·Π(1−p_j)`:

| `p_final` | exact mean retained |
|---|---|
| 0.146 (paper's closed form) | **0.324** |
| 0.2 (paper's adopted value) | **0.302** |
| 0.206 (this repo's exact solve) | 0.300 |

So the paper's Taylor/linear-approximation closed form **undershoots**: 0.146
actually realizes ~32.4% retention, and the exact `p_final` for 30% is ~0.206.
The adopted 0.2 gives 0.302 — essentially 30%, but marginally *above*, not below,
the threshold the paper claims. This does not affect the method, only the
calibration arithmetic. `calibrate_pfinal_paper_approx()` reproduces the paper's
0.146; `calibrate_pfinal()` is the exact numeric solver (0.206). Tests assert
both. (The penultimate-layer cap of §3.1 is irrelevant to this mean — it only
changes `r_L`, which lies outside the averaging window.)

## SPECIFIED (taken verbatim)
- Sigmoid schedule β=20, t_mid=0.5 (0.55 for 3B@20%); PLP up to layer L−2. §3.1, §4
- `k_l = floor(p_l·(n^l_A+n^l_V))`, cumulative on current count. Eq. 6, App. E
- Importance `S` = mean text-query attention to each AV token. Eq. 5
- TDS Algorithm 1 verbatim; `λ_div=0.2`; TDS from layer 14 (7B) / 19 (3B). §3.3, §4
- Retention configs `(p_init,p_final)`: 7B 30%→(0,0.2), 20%→(0.02,0.5);
  3B 30%→(0,0.15), 20%→(0.02,0.5). §4
- Per-chunk 50 audio + 288 video tokens; intra-modality → 70% audio / 40% video
  / ~45% total. §2.1, §3.4, App. A

## PARTIALLY SPECIFIED — defaults chosen (see `ambiguity_audit.md`)
- **Attention used for `S`:** current layer `l`'s post-softmax attention, **mean
  over heads**, query rows = text-query positions, key cols = alive AV tokens.
  (Paper says "attention from text token q" without pinning heads/layer.)
- **Modality balance:** single **global** ranking over the A+V union — no
  per-modality budget. This is what produces the task-adaptive modality balance
  reported in §4.3.
- **`max(C)` (TDS denominator):** max chunk index in the sequence, clamped ≥1.
- **Chunk id after pruning:** fixed at input construction; survivors keep their
  original `chunk_id`.

## UNSPECIFIED — documented assumptions
- **Head aggregation = mean** (OmniZip / FastKV precedent).
- **Text / system tokens are never pruned** (they are the guidance signal).
- **Tie-breaking:** stable, lowest index first (NumPy `kind="stable"`).
- **Dycoke TTM** (§3.4, video): faithful re-implementation of the *described*
  mechanism — keep a full anchor frame per 4-frame group, drop the 80% of each
  other frame's tokens most cosine-similar to the anchor → `(1+3·0.2)/4 = 0.40`.
  Not a port of the Dycoke [24] codebase. (Rounding gives 0.401 for tpf=288.)
- **OmniZip audio selection** (§3.4, audio): we select top-attention tokens given
  a saliency vector; we do not re-derive the audio encoder.

## Out of scope (minimal mode)
- Qwen2.5-Omni model integration, KV-cache/position-id rewiring, FlashAttention.
- VideoMME / WorldSense / AVUT harnesses and the paper's reported accuracy /
  latency / memory numbers (Tables 2, 6, 7) — not reproduced.
- Motivation figures (PCA, cosine-similarity KDE, attention-recall; Figs. 1–2).

## What IS empirically verified by the test suite (23 tests)
- Sigmoid midpoint, penultimate cap, monotone exp variant + `p_init>0` guard.
- Appendix-E closed form ≈ 0.146; exact solver hits target by construction.
- Eq. 5 = mean text→AV attention; head averaging.
- TDS spares temporally-distant low-score tokens; candidate buffer = 2k; edge cases.
- Intra-modality: 70% audio / 40% video / ~45% total; anchor frame fully kept.
- Orchestrator: cumulative & monotone pruning; `k_l` on current (not original)
  count; realized mean retention matches the schedule and the nominal target.
