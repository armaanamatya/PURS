# Reproduction Notes — OmniRefine (arXiv:2605.12056v1)

Honest accounting of every implementation-relevant detail, classified as
**SPECIFIED** (paper states it), **PARTIAL** (paper gestures at it but leaves
a real choice), or **UNSPECIFIED / INVENTED** (we had to decide). Nothing
below was silently filled in.

---

## SPECIFIED — taken directly from the paper

| Item | Value | Source |
|------|-------|--------|
| Base merging ratios ρ_a, ρ_v | 0.3, 0.6 | Sec 4.1 |
| Contextual ratio | 0.05 | Sec 4.1 |
| Spatial / temporal thresholds τ_s, τ_t | 0.82, 0.58 | Sec 4.1 |
| Cross-modal coeff β, anchor threshold | 0.4, 0.4 | Sec 4.1 |
| CPCR penalty λ_c | 0.02 | Sec 4.1 |
| Video chunk size [S_Vmin, S_Vmax] | [3, 5] frames | Appendix A / Alg 1 |
| Audio chunk size [S_Amin, S_Amax] | [90, 140] tokens | Appendix A / Alg 1 |
| DP band ratio B, min window W | 2.0, 48 | Alg 1 |
| Video retention bounds | [0.18, 0.55] | Appendix A |
| Audio bounds | [0.1, 0.9] | Appendix A |
| Video budget modulation α | 0.15 | Appendix A |
| Eq 13 audio budget | m_a = min(a_max, max(a_min, ρ_a − β·(R_v−(1−ρ_v)))) | Appendix B.1 |
| Eq 14 | R_a = 1 − m_a | Appendix B.1 |
| z(·) | mean-pooled hidden states of the region (Eq 12) | Appendix B.1 |
| Eq 6 spatial keep rule | cos(z(R),z(R_c)) ≥ τ_s ∀ child → keep parent | Sec 3.4 |
| Eq 7 temporal merge | cos(z(n_i^{t-1}),z(n_j^{t})) ≥ τ_t → merge | Sec 3.4 |
| Eq 9 anchor assignment | π(t)=argmax_h cos(a_t, a_h) | Sec 3.4 |
| Eq 11 anchor fusion | ã_h = (a_h + Σ w_t a_t)/(1 + Σ w_t) | Sec 3.4 |

## #1 — The probe layer L is the architecture (UNSPECIFIED)

Training-free ⇒ all signals come from a partial forward pass, but the
**method never says at which layer**. Sec 3.1's analysis inspects layers 0
and 8, but that is analysis, not the method spec. We expose it as
`OmniRefineConfig.layer_probe` (default **8**, the deepest analysis layer)
and do **not** hardcode it anywhere else. *This is the load-bearing knob* —
any accuracy reproduction must sweep it. (Connects to the project's existing
L6 cache finding for Qwen2.5-Omni.)

## #2 — DP: main text vs. appendix conflict (we implement the appendix)

Main-text **Eq 5** is a *maximization* with a flat penalty:
`D[u,q] = max_{i,j} [D[i,j] + φ(i,u,j,q) − λ_c]`.
**Algorithm 1** is a *minimization* (`D` init `+∞`,
`cost = −S_match + λ_c·ChunkVariance`). These are **not the same optimum**:
the sign flip is consistent (minimizing −score = maximizing score), but
`λ_c·ChunkVariance` ≠ a flat `λ_c`. We implement **Algorithm 1** (the
appendix is the operational spec). `S_match` = φ (Eq 6, mean masked
similarity in the block).

## #3 — `ChunkVariance` is INVENTED

Algorithm 1 references `ChunkVariance` but never defines it. We define it as
the **variance of the masked frame-audio similarities S̃ inside the
candidate block** (computed via integral images alongside the mean). Plain
reading of "variance" + "discourages over-fragmentation". Mark as invented.

## #4 — Neighborhood N(·) in Eq 4 (PARTIAL)

`M_{f,t} = 1[c_a(t) ∈ N(c_v(f))]` restricts similarity to a native
neighborhood, but the neighborhood *radius* is not given. Exposed as
`neighborhood_radius` (default 1 = the native chunk ± its immediate temporal
neighbors).

## #5 — "Fused attention-based importance" / audio saliency (PARTIAL)

Sec 3.4 selects "dominant audio tokens according to fused attention-based
importance scores" but does not define the fusion. We make `audio_saliency`
an **explicit input** to `compress_audio_chunk` (the adapter computes it).
A reasonable choice: mean attention mass each audio token *receives* at the
probe layer. Likewise video saliency (for bound clamping) is an optional
input, not invented internally.

## #6 — Merge weight w_t (PARTIAL)

Eq 11 says `w_t` is obtained by "normalizing the relevance scores within
M(h)" but does not define *relevance*. We use **cross-modal cosine to the
nearest retained video rep** when video reps are available (the section
stresses cross-modal guidance), falling back to **audio cosine to the
anchor** otherwise; then L1-normalize within M(h).

## #7 — Dominant vs. contextual split (PARTIAL)

The retained budget (R_a tokens) is split into "dominant" (top saliency) and
a small set of "contextual" anchors (contextual_ratio = 0.05). The exact
selection of contextual anchors is unspecified; we take the next-highest
saliency tokens after the dominant set. An equally valid reading is "one
contextual anchor per under-covered semantic interval".

## #8 — Audio bounds: merging ratio vs. retention (minor ambiguity)

Appendix A calls [0.1, 0.9] "audio retention bounds"; Appendix B.1 calls
[a_min, a_max] bounds on the audio *merging* ratio. We apply them to **m_a**
exactly as Eq 13 is written. Numerically symmetric (R_a = 1 − m_a), so the
admissible R_a range is [0.1, 0.9] either way.

## #9 — Hard retention-bound enforcement (PARTIAL mechanism)

The bounds [0.18, 0.55] (video) are stated, but the *mechanism* to enforce
them when the threshold-driven merge over/under-shoots is not. We clamp by
dropping lowest-saliency nodes (above v_max) or re-adding highest-saliency
individual tokens (below v_min). Documented in `pipeline._enforce_video_bounds`.

---

## #10 - Runtime bridge scope (UNSPECIFIED / implementation boundary)

The paper describes OmniRefine as training-free and applied before LLM prefill,
but it does not specify an executable Qwen2.5-Omni interface, batching
semantics, or how to prune already-materialized K/V cache entries if probing at
an internal layer L. `torch_runtime.py` therefore implements only a conservative
prefill-level bridge (`layer_probe=0`): it converts already-merged Qwen
`inputs_embeds` into `ProbeInputs`, writes Eq 11 merged anchor reps back to the
kept positions, and returns a boolean `global_mask` for slicing
`inputs_embeds`, `attention_mask`, and `position_ids`. Internal layer-L pruning
remains a model-version-specific adapter task.

Fallbacks in this bridge are explicitly not paper claims:

- batch size is restricted to 1;
- if Qwen native chunk ids are not provided, equal-rank native chunks are used;
- if Qwen grid metadata is not provided, a square-ish frame grid is inferred;
- if attention logits are not provided, audio saliency is uniform;
- sequences shorter than the paper's CPCR bounds, or infeasible under those
  bounds, return an identity mask with a diagnostic warning.

## Accuracy reproduction — OUT OF SCOPE (minimal mode)

We do **not** reproduce the reported numbers (e.g. WorldSense 46.7% @ 44%
token retention on Qwen2.5-Omni-7B). That requires the Qwen2.5-Omni
checkpoints, the WorldSense / VideoMME / AVUT benchmarks, the LMMs-Eval
harness, and a working `adapter.probe/apply`. What this repo *does* verify
(in `tests/`) are the paper's structural invariants on synthetic input:

- CPCR traceback yields monotonic, gap-free, non-overlapping chunks covering
  all F frames / N audio tokens, with sizes in [3,5] / [90,140].
- Retention respects the hard bounds (video [0.18,0.55]; audio coupled via
  Eq 13/14).
- Output token count matches the target ratio; every pruned audio token is
  assigned to a retained anchor.
- A uniform frame collapses to one node; a diverse frame does not.
