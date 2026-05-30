# Tri-Modal Pruning for Streaming Qwen2.5-Omni — Design

> Companion to `streamingtom_omnizip_composition.md`. Resolves the 9 blockers (3 hard, 2 soft, 4 silent) raised there.
> Sources: NotebookLM 736d820f. New sources added this pass: EchoingPixels 2512.10324 (`46bc33d8`), Video-SALMONN-S 2510.11129 (`03785de2`), STC 2512.00891 (`d1145bb4`). Existing: OmniZip (`6d47c43b`), StreamingTOM (`41d40947`), OmniSift (`3e4a1de4`), Survey (`62ed2215`), DyCoke (`e78d80b1`), HoliTom (`9e375c1d`).

---

## 1. What "tri-modal" means here — committing a decomposition

"Tri-modal" is overloaded. Four candidate interpretations, ordered worst → best for *this* training-free streaming setting:

**Interpretation 1: Audio + Video + Text-question.** Treat the question as a third pruning axis. *Rejected.* Under the RVS protocol the question arrives **after** the stream, so it cannot drive prefill-time pruning. Worse, in Qwen2.5-Omni's omnimodal flow the question is short (≪50 tokens) — there is no compression to spend.

**Interpretation 2: Audio + Video + Past-KV-memory tier.** Three pruners: OmniZip-causal at prefill, plus a separate evictor over OQM at decode. *Rejected as a "tri-modal" framing.* This is really two-modal × two-stage — the third "axis" is a temporal-memory mechanism, not a modality. It is good engineering (and is implicit in our pipeline) but mislabelled as tri-modal.

**Interpretation 3: Single-pool joint reduction (EchoingPixels style).** One score, one budget, mixed audio+video. *Rejected for our setting.* EchoingPixels (2512.10324) explicitly co-pools audio and video and lets a trained CS2 module arbitrate ("reduces tokens from an entire combined pool... rather than using fixed budgets per modality"; "co-design Sync-RoPE to maintain temporal relationships for the sparsely selected tokens"). It is *trained* — not training-free — and joint pooling forfeits the asymmetric statistics that make audio scores in Qwen2.5-Omni cheap to compute (50-token sequences) and video scores expensive (288-token sequences per chunk). Single-pool also collapses our diagnostic surface: we can no longer ablate "audio pruning hurt X, spatial hurt Y."

**Interpretation 4 (committed): Audio (A) × Video-spatial-within-frame (V_s) × Video-temporal-frame-selection (V_t).** Three orthogonal axes, each with its own statistics, signal, and budget rule. Justified because:

1. *Statistical orthogonality.* The three axes have distinct redundancy structure: audio is dense per-token (already pre-pooled to ~50/window — Qwen-Omni §TMRoPE; OmniZip §4.1), video-spatial is patch-grid redundant (288/frame mostly background), video-temporal is frame-redundant (adjacent frames near-identical at 0.5–2 fps). One score function does not subsume the other two.
2. *Existing primitives.* OmniZip already does (A) and (V_s, jointly with audio guidance); StreamingTOM CTR does (V_s, V_t weakly); DyCoke TTM does (V_t) but offline. There is no training-free, streaming-causal method that does (A, V_s, V_t) as **independently controllable** axes today. That is the gap.
3. *Causality factorization.* (V_t) — "should this frame survive at all?" — is the most causality-hostile decision because it concerns *future* unseen frames. Splitting it out lets us bound its lookahead separately (we use 1 chunk = 2 s) without forcing the same lookahead onto (A) and (V_s).
4. *Compatibility with OQM.* OQM stores fixed-G groups. With three axes, each axis has its own group budget — `G_A, G_Vs, G_Vt-summary` — and OQM extends from dual to **quad-structure** (sys-FP16 / audio-q4 / video-spatial-q4 / video-temporal-summary-FP16) cleanly because the int4 packing is mathematically agnostic to G value (NotebookLM probe of StreamingTOM §3.4: "the int4 packing algorithm itself does not mathematically assume G=50... as long as G is fixed").

**Defense against the obvious objection** ("V_t is just sparse video"): no — frame-selection drops *whole frames* before patch tokenization, saving ViT-encode FLOPs (~30% of prefill in Qwen2.5-Omni at 0.5 fps). Spatial pruning operates *post-ViT*, on patch tokens. They attack different bottlenecks. STC (2512.00891) validates this hierarchical split: STC-Cacher (≈V_t) saves 24.5% ViT latency; STC-Pruner (≈V_s) saves 45.3% LLM-prefill latency. **Independent, multiplicative.**

---

## 2. Pipeline diagram (per TMRoPE chunk, t = 2 s)

```
                Streaming chunk t :  audio_t (~50 raw windows) + video_t (≤4 frames @ 2 fps)
                                              │
        ┌─────────────────────── Stage 0 (pre-ViT) ──────────────────────────────┐
        │  Axis V_t : Frame-selection                                            │
        │     signal = ‖φ_low(f_i) − φ_low(f_{i-1})‖₂ (low-res perceptual diff,  │
        │              cached from prev chunk, 8×8 thumbnail) + audio-energy     │
        │              gate g_a(t) from raw mel features (zero-cost).            │
        │     budget : drop frame i if diff_i < τ_t AND g_a(t) below quantile q. │
        │              τ_t adapts via running quantile (t-digest) over diff_i.   │
        │     output : keep ≤K_t frames out of |video_t|; emit a 1-token         │
        │              "frame-summary" (mean-pool of dropped patches) per drop   │
        │              for V_t-summary group → OQM FP16 tier.                    │
        └────────────────────────────────────────────────────────────────────────┘
                                              ▼ (selected frames only)
                              Qwen2.5-Omni ViT encodes audio + selected video frames
                                              │
        ┌─────────────────────── Stage 1 (post-encode) ──────────────────────────┐
        │  Axis A : Audio-token pruning (OmniZip-causal)                         │
        │     signal = sliding-window self-attn over last W=3 chunks → S_a(t)    │
        │              + cross-modal merge S_cross with V_s (already per-window) │
        │     budget : ρ_a · 50  with ρ_a = clip(0.3 + α·zscore(S_a; t-digest),  │
        │                                       [0.2, 0.7])                     │
        │     output : G_A audio anchor tokens for chunk t                       │
        │                                                                        │
        │  Axis V_s : Video spatial pruning (audio-guided)                       │
        │     signal = OmniZip's per-window ρ'_v(t) driven by S_a(t),            │
        │              normalized by streaming quantile (NOT global mean)        │
        │     budget : G_Vs (FIXED) tokens per surviving frame; OmniZip selects  │
        │              WHICH G_Vs to keep, OQM gets predictable layout           │
        │     output : G_Vs · |kept frames in chunk| spatial tokens              │
        └────────────────────────────────────────────────────────────────────────┘
                                              ▼
                         Qwen2.5-Omni Thinker prefill (chunk t — variable size only across t)
                                              │ writes K,V for selected tokens
                                              ▼
        ┌─────────────────────── Stage 2 (post-LLM) ─────────────────────────────┐
        │  OQM, quad-structure :                                                  │
        │     M_t = {sys FP16}                                                    │
        │         ∪ {(uint4(K_A_t), s_A,t, m_A,t, k̄_A,t)}      audio group       │
        │         ∪ {(uint4(K_Vs_t), s_Vs,t, m_Vs,t, k̄_Vs,t)}  video-spatial grp │
        │         ∪ {(FP16 V_t-summary token, k̄_Vt,t)}         frame-summary     │
        │  Per-modality top-K_m retrieval at decode (NOT joint top-K).            │
        └────────────────────────────────────────────────────────────────────────┘
                              query q arrives → top-K per axis → cross-axis fuse → decode
```

**Lookahead budget**: exactly 1 TMRoPE chunk = 2 s, forced by Qwen2.5-Omni's audio encoder receptive field (we cannot compute `S_a(t)` for chunk *t* without buffering chunk *t*). No extra lookahead added by the design.

---

## 3. Per-axis design

### 3.1 Axis V_t — Video temporal / frame-selection

**Signal (causal-streaming).** Two cheap signals fused, both available *before* ViT encode:

1. **Low-resolution perceptual diff**: 8×8 thumbnail (downsampled by ViT preprocessor), `d_i = ‖f_i_thumb − f_{i-1}_thumb‖₂ / d_dim`. Cost: ~negligible (0.5 ms/frame on a single GPU thread). Causal by construction (uses only prev frame, cached).
2. **Audio-energy gate**: `g_a(t) = log(1 + Σ |mel_t|²)`, mean over the 2 s window. Computed from the raw audio that *already must arrive* for the audio encoder. Zero marginal cost. Provides a "is anything happening?" prior — drops a still frame held for 4 s of silence faster than a still frame held for 4 s of speech (because the latter still receives audio-anchor tokens that may semantically reference it via OmniZip's S_cross).

**Fusion.** `score_Vt(i) = d_i + λ · g_a(chunk(i))` with `λ ≈ 0.3` (cheap to tune; see §6 ablation 6).

**Budget rule.** Adaptive threshold from a t-digest streaming quantile over the last *T*=200 frames of `score_Vt`: drop frame *i* iff `score_Vt(i) < quantile(0.4)`. Lower bound: never drop more than 50% of frames in a chunk (ensures ≥1 frame survives every 2 s — bounds worst-case retrieval starvation).

**Frame-summary emission.** Each dropped frame emits a single 1-token mean-pool of its (un-tokenized) patch grid into `V_t-summary` group, stored FP16 in OQM. Cost: 1 token per dropped frame; over a 1 h stream at 0.5 fps with 40% drop rate this is ~720 FP16 tokens. Cheap. The point is *not* to recover the visual content — it is to give the retrieval head a fingerprint (`k̄`) so the decoder can detect "you dropped a frame relevant to this question" and (optionally) re-encode on demand. **Defense against silent skip-bias** (Failure Mode 5).

**Failure modes addressed.**
- *Static-camera lectures*: pure visual-diff drops every frame after frame 1; the audio-energy gate prevents this from being a no-op when the lecturer is talking (audio anchors carry the signal; visual frames legitimately redundant).
- *Silent action sequences* (chase, sport): visual-diff fires, audio-energy gate idle → high frame retention. Correct.
- *Drift at scene cuts*: t-digest's online quantile lags. Mitigation — when `d_i > 5 · running_mean`, force-keep frame and mark a "scene-cut event" that resets the quantile.

### 3.2 Axis A — Audio-token pruning

**Signal (streaming-causal).** Sliding-window OmniZip score: replace OmniZip's global self-attention `A ∈ ℝ^{B×N_a×N_a}` (OmniZip §3.2) with windowed attention over the last *W*=3 chunks (≈150 audio tokens). Score per token = mean column of the windowed attention matrix. Plus the per-window `S_cross = Ĥ_a Ĥ_v^T`, which is **already chunk-local** in OmniZip.

**Budget rule.** Replace OmniZip's *global* normalization (which requires whole-clip mean of `ρ'_v`) with a streaming **t-digest quantile**: maintain online estimate of `S_a` quantiles, set `ρ_a(t) = 0.3 + 0.4 · sigmoid(zscore(S_a(t)))`, clipped to `[0.2, 0.7]`. This preserves OmniZip's intent ("dense windows keep more, sparse keep less") with O(1) state.

**Justification of W=3.** Audio events at 2 s granularity have 4–6 s typical context (speech phonemes ≤ word ≤ phrase). W=3 covers this without buffering any *future* — pure left-only. We expect minimal degradation vs. OmniZip's global score because the 50-token-per-window pre-pooling already absorbs short-range structure (per-author OmniZip §4.1). Validate in ablation 7.

**Failure modes addressed.**
- *One-off audio anomaly* (a single discriminative cough in a long lecture): sliding-window score may rank it lower than global would. Mitigation — emit `S_a(t)` to the t-digest *before* applying the quantile gate, so a token whose absolute score exceeds the 95th percentile of the historical distribution gets force-kept regardless of local rank. Cheap, principled.
- *Bursty audio events overflowing G_A*: cap per-chunk audio anchors at `G_A` (default 25) and overflow into a "spillover" group at lower priority. Logs the overflow count for diagnostic.

### 3.3 Axis V_s — Video spatial pruning

**Signal.** OmniZip's audio-guided `ρ'_v(t)` (OmniZip §3.3 Eq. 5) restricted to the surviving frames from V_t. Cross-modal merge `S_cross = Ĥ_a Ĥ_v^T` is per-window in OmniZip and stays per-window — no change.

**Budget rule.** **Fixed `G_Vs` per surviving frame** (default G_Vs=50 — match StreamingTOM CTR for direct comparability, NotebookLM §3.4 says int4 packing tolerates any consistent G). OmniZip's adaptivity is preserved as *which* G_Vs tokens to keep, not *how many*. The chunks where OmniZip wants ρ'_v=0.9 (dense) will silently truncate — log this overflow rate per the diagnostic in §5 Blocker 2. The chunks where OmniZip wants ρ'_v=0.1 (sparse) **also keep G_Vs**, padded with the next-best tokens by score; this is cheap insurance against under-budgeting at low-saliency moments that the question may later target.

**Optional variant (V_s-elastic).** `G_Vs ∈ [25, 100]` with an elastic budget normalized by t-digest over surviving frames; sacrifices OQM latency predictability for OmniZip adaptivity. We test both in ablation 4 vs 5.

**Failure modes addressed.**
- *Quant-distortion of audio-elevated tokens* (the silent killer from prior doc Failure Mode 1): we now have a knob — store top-`r%` of OmniZip-elevated spatial tokens at FP16 in a dedicated `Vs-anchor` sub-group. Default r=0 (matches baseline); test r∈{5, 10, 20} in ablation 3.

---

## 4. Cross-axis coordination

**Order of operations (within chunk t).** Sequential:
1. V_t fires first (pre-ViT). Cuts frame count; bounds downstream cost.
2. A and V_s fire **simultaneously** (post-encode), since OmniZip's S_cross requires both audio and video tokens encoded. A drives V_s via S_a → ρ'_v as in OmniZip.
3. OQM commits all three groups.

**Who guides whom.**
- A → V_s: yes (OmniZip's audio-guided spatial). Inherited.
- A → V_t: yes (audio-energy gate). Cheap, asymmetric, defended above.
- V_s → A: **no.** OmniSift would say yes (vision-anchored audio scoring), but OmniSift is *trained* — the VGAS module has 4.85M parameters optimized end-to-end (NotebookLM probe of OmniSift §3.4). We are training-free; cannot invert OmniZip's selector to V_s → A without retraining. **Honest limitation.**
- V_t → A: no. Frame-selection happens before audio encoding finishes; the dependency would be circular.
- V_t → V_s: yes by construction (V_s only operates on frames V_t kept).

**Tie-breaking under contention.** Two contention modes:
1. *Token-budget pressure across axes.* If the per-chunk total `G_A + |kept frames|·G_Vs + |dropped frames|·1` exceeds a global budget `B_chunk`, we trim in this priority order: V_t-summary (drop entire summaries first — cheapest semantic loss), then V_s overflow (truncate to G_Vs), then A overflow (cap to G_A). Audio anchors are protected because they drive the cross-modal score that everything else depends on at retrieval time.
2. *Retrieval contention at decode.* Per-modality top-K with `K_A=4, K_Vs=12, K_Vt=8` groups (tunable). Joint top-K over a flat pool would let abundant V_s groups crowd out audio — this is the modality-collapse failure mode. Per-modality top-K is the standard fix (and matches multi-tier retrieval practice from RAG literature).

**Cross-axis bidirectional saliency (open).** EchoingPixels' single-pool argument — that joint co-attention finds combinations no axis-decomposed scorer can — has merit. Without retraining we cannot match it, but we can approximate by computing a **single post-hoc consistency check**: after all three axes commit, recompute `S_cross` over the *committed* tokens; if any audio anchor has zero high-similarity V_s neighbor, it is likely orphaned (audio talks about something we pruned away) — log and optionally lift the V_s budget for the next chunk by 10%. Speculative; not in the MVE.

---

## 5. Blocker resolution table

The 9 blockers from `streamingtom_omnizip_composition.md`. (Numbering: Q1 = Causality of audio selector; Q2 = OQM modality assumptions; Q3 = Sync tension; Q4 = Where audio compression fits; F1–F4 = Failure modes; OmniZip *global budget normalization* = soft Q1b.)

| # | Blocker | Resolution | Residual risk |
|---|---|---|---|
| Q1 | OmniZip's global self-attention over all audio tokens (non-causal) | Sliding-window self-attn over last W=3 chunks for `S_a` (§3.2). Restores strict causality. | Loss of long-range audio anomaly ranking. **Mitigated** by force-keep-on-absolute-95th-percentile from t-digest (§3.2). |
| Q1b | OmniZip's global normalization of per-window ρ'_v | Streaming t-digest quantile estimator replaces global mean (§3.3 budget rule). O(1) state, sublinear error. | Drift at hard scene cuts; mitigated by quantile reset on visual-diff outlier (§3.1). |
| Q2 | OQM's dual-structure memory + fixed G assumption | Quad-structure {sys, A, V_s, V_t-summary} with per-axis G_m. Verified by NotebookLM probe: int4 packing math is agnostic to G value (only requires G fixed *per group type*). Per-modality top-K at retrieval. | Larger metadata overhead per group (4 scalar (s, m) per axis × N groups). Negligible. |
| Q3 | Audio/video clock-sync tension | TMRoPE 2 s chunk is the natural quantum; **forced** lookahead of 1 chunk. No additional lookahead. Documented as "near-causal with chunk-bounded lookahead" (per prior doc §Q3). | This is now a property, not a bug. |
| Q4 | Where audio compression fits | Option (b) from prior doc — full OmniZip audio path per chunk, audio anchors stored as their own OQM group. Required so S_cross is preservable. | None — committed. |
| F1 | Quant-distortion of audio-elevated spatial tokens | Optional `Vs-anchor` FP16 tier for top-r% audio-elevated tokens (§3.3). Tested at r ∈ {0, 5, 10, 20} in ablation 3. | If r>0 helps materially, OQM's clean int4-only story complicates. Acceptable tradeoff. |
| F2 | Variable-budget × fixed-G silent truncation | Fixed G_Vs is now an explicit design choice with overflow logging (§3.3); elastic V_s tested as ablation 5. | Quality leak in dense windows is *measured*, not silent. |
| F3 | Bursty audio events break per-frame budgets | Audio overflow → spillover group (§3.2 failure modes). | Spillover group quality unstudied; treat as instrumentation in v1. |
| F4 | Streaming quantile drifts at scene cuts | Quantile reset on `d_i > 5·running_mean` (§3.1). | Threshold 5× is a heuristic; could be hyperparameter-tuned per benchmark. |

**Score: 6 cleanly resolved (Q1, Q1b, Q2, Q3, Q4, F2), 3 mitigated-with-instrumentation (F1, F3, F4).** Honest reading: F1 is the one that decides the paper — see §6 ablation 3.

---

## 6. Open design tensions left unresolved

1. **No bidirectional cross-modal saliency.** Without retraining, V_s → A scoring is unavailable. EchoingPixels' single-pool joint approach (2512.10324) likely beats us on benchmarks where the discriminative cue is visual but the question is phrased as audio (e.g., "what made that sound?" — the answer lives in V_s but the query targets A). We cannot test this without their model. Documented limitation; do not over-claim.
2. **V_t-summary FP16 tokens are an unaudited memory leak.** At 1 token per dropped frame they are small per-chunk but accumulate linearly in stream length. After 4 h of streaming at 50% drop rate this is ~7200 FP16 tokens, comparable to OQM's quantized payload. If they don't materially improve retrieval recall (ablation 8), drop them.
3. **Per-modality top-K weights `K_A:K_Vs:K_Vt`** are unprincipled. Defaulting to 4:12:8 is an educated guess. A learned router (training-free) would test fixed ratios on a held-out dev set; we just declare "v1 = 4:12:8."
4. **OQM int4 quant of audio K,V is unvalidated.** StreamingTOM never quantized audio. Audio K vectors may have different per-channel statistics (post-Whisper-style encoder) than visual K. The min/max packing might silently underflow audio. Detect by comparing reconstruction error per axis.
5. **`λ` audio-gate weight in V_t is one number, fixed across content types.** Lectures vs. movies vs. cooking-videos likely want different λ. v1 keeps it fixed; v2 could adapt λ from a content-type classifier.

---

## 7. Minimum viable experiment (delta vs prior MVE)

The prior doc's MVE had 6 ablations on a single composition. The tri-axis design adds **4 axis-specific ablations** without expanding the headline benchmark set:

**Inherited from prior MVE** (still required):
- (1) CTR + OQM (StreamingTOM-as-is, no audio) — control.
- (2) OmniZip-causal + OQM, audio muted — isolates "does causal-OmniZip ≥ CTR for video?"
- (3) FP16 anchor sweep r ∈ {0, 5, 10, 20} — isolates F1 (quant distortion of audio-elevated tokens).
- (4) Fixed-G_Vs vs (5) elastic-G_Vs.
- (6) Lookahead L ∈ {0, 1, 2, 4} TMRoPE chunks.

**New for tri-axis design**:
- (7) Sliding window W ∈ {1, 3, 5, ∞} for audio attention scope. Tests whether causal-windowed attention loses anything vs. global. Decisive for whether the streaming claim costs accuracy.
- (8) V_t on/off × V_t-summary on/off (2×2). Validates that frame selection helps prefill cost without tanking QA, and quantifies whether summary tokens earn their FP16 bytes.
- (9) λ ∈ {0, 0.15, 0.3, 0.6} — does audio-energy gating help frame-drop decisions?
- (10) Per-modality top-K (4:12:8 default vs flat top-K of 24) — validates Question 4 of cross-axis coordination.

**Benchmarks** (unchanged from prior doc):
- *streaming-AVUT* (replay AVUT at 0.5 fps, RVS-style post-stream question) — primary.
- *RVS-Ego + RVS-Movie* at 0.5 fps — video-only sanity (audio muted).
- *AVUT + WorldSense* offline — confirms tri-axis pipeline doesn't tank OmniZip-offline numbers.

**Cost.** Prior MVE = ~12 GPU-hr. Adding 4 ablations × 1k clips × 10 s ≈ 11 GPU-hr. Total ~23 GPU-hr on a single 7B GPU. Tractable.

**New competitor to beat (this round).** **EchoingPixels (2512.10324)**, which did not exist when the prior doc was written. They report 2–3× speedup at 5–20% token retention. We claim a comparable speedup *without* training — that is the honest pitch. If we cannot match within 2 pts on AVUT, the value proposition shifts to "training-free + composable with OQM" rather than "quality-matching."

**New kill criterion**: if (W=∞) beats (W=3) by >2 pts on streaming-AVUT, the streaming-causal claim doesn't hold up empirically — we either accept a 1-chunk additional lookahead (W=5) or kill.

---

## 8. Verdict

**More publishable than the two-axis composition; same risk profile.** The tri-axis decomposition is the right framing because (a) it exposes a clean diagnostic surface — every ablation row maps to a single design choice rather than a tangled compound, (b) it factorizes causality cleanly: V_t lookahead = chunk; A causality = sliding-window; V_s causality = inherited from A, (c) OQM extends to a quad-structure with no algorithmic change, only a per-axis G, and (d) it positions us against EchoingPixels (joint single-pool) and Video-SALMONN-S (TTT memory) as orthogonal alternatives in the omnimodal-streaming design space, not as a strict subset of either.

The honest weak point remains F1 (quant distortion of audio-elevated tokens) and the lack of bidirectional V_s → A guidance. The first is testable in ablation 3; the second is a true limitation we cannot close without retraining. Acknowledging both is what makes this design a paper rather than a benchmark hack.

The single most novel piece is **per-axis OQM with per-modality top-K retrieval** — to our knowledge no streaming KV-cache compression today separates audio / video-spatial / video-temporal-summary into distinct retrievable banks with per-axis budgets. Everything else (sliding-window audio attn, t-digest quantile, audio-energy frame gate) is principled but plumbing; the per-modality cache structure is the contribution.

---

## Appendix A — Sources cited inline

- **OmniZip** (NotebookLM `6d47c43b`): §3.2 audio attention global, §3.3 ρ'_v normalization Eq. 5, §4.1 token counts.
- **StreamingTOM** (NotebookLM `41d40947`): §3.4 OQM dual-structure, int4 packing, cosine top-K. Probe confirms G is configurable; per-modality top-K not in original.
- **OmniSift** (NotebookLM `3e4a1de4`): §3.4 VGAS architecture; trained 4.85M params; "modality-asymmetric" defended explicitly. Inversion requires retraining.
- **EchoingPixels** (arXiv 2512.10324, NotebookLM `46bc33d8`): joint single-pool reduction with trained CS2 + Sync-RoPE. Not training-free.
- **Video-SALMONN-S** (arXiv 2510.11129, NotebookLM `03785de2`): TTT memory is parametric (not KV); orthogonal alternative to OQM.
- **STC** (arXiv 2512.00891, NotebookLM `d1145bb4`): hierarchical ViT-cache + LLM-pre-prune validates V_t × V_s split as multiplicative.
- **DyCoke / HoliTom** (NotebookLM `e78d80b1`, `9e375c1d`): both offline; not directly portable to streaming-causal V_t.
- **Survey 2507.20198** (NotebookLM `62ed2215`): confirms StreamingTOM-CTR is the *only* prior streaming-causal frame-selector in this neighborhood.

## Appendix B — Saved NotebookLM notes (this session)

- "EchoingPixels CS2 architecture and training/streaming status" (`2ac856e6`)
- "Video-SALMONN-S TTT memory vs OQM compatibility" (`9f113bdd`)
- "STC hierarchical ViT-cache + LLM-pruner architecture" (`d57400e9`)
