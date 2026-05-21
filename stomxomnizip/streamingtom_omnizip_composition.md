# StreamingTOM × OmniZip Composition — Feasibility Analysis

> Sources used (NotebookLM 736d820f): OmniZip full PDF (`6d47c43b`), StreamingTOM full PDF (`41d40947`), OmniSift PDF (`3e4a1de4`), survey 2507.20198 (`62ed2215`). Section/equation cites refer to those PDFs as indexed.

## Hypothesis

Replace StreamingTOM's CTR (causal-temporal frame-difference + spatial-attention saliency) with OmniZip's audio-guided selector, and reuse StreamingTOM's OQM (4-bit quantized KV memory + group-level retrieval) verbatim, to obtain a training-free, *omnimodal*, *streaming*, bounded-memory pipeline for Qwen2.5-Omni — currently a literature gap (StreamingTOM is video-only/streaming; OmniZip is omnimodal/offline by the authors' own admission, OmniZip Appendix F).

## Architecture sketch (proposed)

```
                                    streaming chunk t (audio_t, video_t)
                                              │
            ┌─────────────────────────────────┼─────────────────────────────────┐
            │                                 ▼                                 │
            │     OmniZip-causal selector (per-window, fwd-only audio attn)     │
            │     -> S_a(t)  -> ρ'_v(t) ∈ [ρ_min, ρ_max]                        │
            │     -> top-G_t video tokens; audio anchors + cross-modal merges    │
            └─────────────────────────────────┬─────────────────────────────────┘
                                              ▼  (variable size per chunk)
                                  Qwen2.5-Omni Thinker prefill (chunk t)
                                              │ writes K,V for selected tokens
                                              ▼
                            StreamingTOM OQM:  group-quantize K,V to int4
                            store group g_t with rep key  k̄_t = mean(K_t)
                                              │
                              query q arrives ─┼─> top-K group retrieval
                                              ▼
                                       decode (FP16 active KV)
```

Net effect: OmniZip replaces CTR's frame-difference + visual-saliency pruner; OQM is unchanged downstream. Three layers of friction listed below.

## Question 1: Causality of the audio-guided selector

OmniZip's per-token audio score uses **non-causal global self-attention**: `A = softmax(QKᵀ/√d) ∈ ℝ^{B×N_a×N_a}` over the **entire** audio sequence, taking each token's importance as the *mean column* — i.e., "mean attention each audio token receives from **all other** audio tokens" (OmniZip §3.2). There is no causal mask. Two further global ops then run:

1. **Global top-ρ_a anchor selection** across all windows (OmniZip §3.2 — anchors are picked by global rank).
2. **Global budget normalization** of the per-window video ratios `ρ'_v(i)` so they average to the target `ρ_v` (OmniZip §3.3, Eq. 5 + the normalization step).

Both require the whole clip. The cross-modal merge `S_cross = Ĥ_a Ĥ_vᵀ` is *already* per-window (OmniZip §3.2), so it is the only causal-friendly piece out of the box.

**Minimal modification to make it streaming-safe.** Three options, in increasing cost:

- *Sliding-window audio attention* (cheapest). Replace global `A` with attention over the last *W* windows. Loses global ranking; `S_a(t)` becomes "how salient relative to recent past" rather than "relative to whole clip." Likely fine when audio events have local temporal structure (speech, music beats, action sounds), bad when the discriminative event is a one-off contrast far apart (e.g., a single anomalous cough in a long lecture).
- *Per-chunk recompute with running quantile* (medium). Compute `A` only within chunk *t* of size `W_chunk`, then convert to `S_a(t)` by mapping to a **streaming quantile estimator** (e.g., t-digest) over historical scores. Replaces global ranking with online ranking. Trivially preserves Eq. 5's intent.
- *Causal lookahead of `L` windows* (most invasive). Buffer `L` future audio windows before committing — breaks pure causality, costs `L · window_duration` of TTFT. StreamingTOM allows zero lookahead by construction (§3.3, "strict causality with a 2-frame window").

**Quality preservation is unverified**: OmniZip reports no ablation on local vs. global attention scope, and the paper's own audio token count is small (50 tokens per window — §4.1, "For each time window, it has 50 audio tokens and 288 video tokens"). Speculation: at this scale, local attention probably keeps most of the signal because semantic chunks of audio are already pooled into the 50 tokens; the brittleness lies in the global *normalization*, not the global *attention*.

## Question 2: OQM's modality assumptions

OQM is **mostly** modality-agnostic mechanically but **not** as currently structured.

Mechanical agnosticism (StreamingTOM §3.4):

- The representative key is just `k̄_t = mean(K_t)` over whatever tokens land in the group — "obtained by averaging the keys before quantization." No spatial pooling, no CLS token.
- Retrieval is `R = TopK { sim(q, k̄_i) }` (Eq. 12) — flat cosine top-K over all stored groups. No frame/spatial assumption.

Structural assumption that breaks modality-mixing:

- §3.4 explicitly defines a **dual-structure** memory: `M_t = { FP16 system tokens } ∪ { Q_4(G_i), k̄_i for visual groups }`. Audio is unmentioned. To compose with OmniZip, you must extend to **tri-structure** (sys / visual / audio) **or** fuse audio anchors into the visual group's `G` budget.

**Hard constraint that conflicts with OmniZip**: OQM requires a **fixed per-group budget G** (StreamingTOM §3.3: "fixed per-frame budget G to stabilize latency and ensure predictable compute and memory consumption across all frames"; default `G = 50`, yielding the 15.7× number from `4 · 196/50 ≈ 15.7×`). OmniZip's whole point is that `ρ'_v(i)` is **dynamic across windows** — high-saliency windows keep more, low-saliency keep less. Two ways to reconcile:

- (a) **Fix G, let OmniZip choose *which* G tokens** (give up dynamic budget; lose OmniZip's adaptivity but gain OQM's predictable memory). This is the cleanest fit.
- (b) **Variable G per group with the same retrieval mechanism**. OQM's storage doesn't actually require uniform `G` — only the per-frame *latency budget* does. Storing variable-length groups is a one-line change; the cost is loss of latency predictability and a tweak to the int4 packing layout.

The TMRoPE interleaved layout (Qwen2.5-Omni interleaves audio/video at a 2 s granularity) maps cleanly onto OQM's per-window grouping if you commit one group per modality per 2 s window. Audio rep keys would be queried by the same cosine top-K, and modality-balanced retrieval is achievable by per-modality top-K rather than joint.

## Question 3: Synchronization tension

OmniZip's `S_cross` is local to a window (OmniZip §3.2: "for each time window... paired video segment"), so cross-modal alignment is already chunked. The synchronization tension does **not** come from `S_cross`; it comes from the fact that OmniZip computes `S_a(t)` from the audio encoder *first*, then uses it to drive `ρ'_v(t)`. In Qwen2.5-Omni, audio is encoded at ~50 Hz post-pooling and video at the frame rate — they don't share a clock.

In streaming, the cleanest protocol:

1. Buffer 2 s of audio + the matching ≤2 s of video frames (matches the TMRoPE chunk granularity — *not* additional lookahead, just the model's own native chunk boundary).
2. Run audio-encoder forward over the chunk → `S_a(t)`.
3. Use `S_a(t)` to set `ρ'_v(t)` for the same chunk's video tokens.
4. Commit the variable-or-fixed-G group(s) to OQM.

So a **2 s "lookahead" is forced** by Qwen2.5-Omni's own TMRoPE chunking, not by the composition. StreamingTOM tolerates this — its "strict causality with a 2-frame window" (§3.3) already caches one frame back; allowing a 2 s forward window is a strictly larger but still bounded latency. This is the cleanest design point.

Speculation: pure per-frame causality (the StreamingTOM ideal) is impossible to keep — *any* audio guidance signal needs a window long enough for the audio encoder's receptive field. The honest framing is "near-causal with chunk-bounded lookahead = TMRoPE chunk size."

## Question 4: Where audio token compression fits

OmniZip prunes audio to `ρ_a` of original (OmniZip §3.2 + Table 6: typical `ρ_a ∈ {0.3, 0.4, 0.5, 0.55}`); StreamingTOM does *nothing* to audio (§3.3 — visual only). OmniZip's audio-token count is small relative to video (§4.1: 50 vs. 288 per window — ~6× video-heavy), which makes audio compression less critical for prefill but still useful for KV memory.

Three options for the composition:

- **(a)** Skip OmniZip's audio pruning, let OQM 4-bit quantize all audio K,V post-LLM. — Simplest. Audio is small enough that the prefill cost is fine; OQM still bounds memory. Loses OmniZip's audio anchor-and-merge logic, which means `S_cross` cannot be computed (because that step requires the merged audio anchors). **This breaks the very mechanism that makes the composition novel.**
- **(b)** Run OmniZip's full two-stage audio path (intra-modal anchor + cross-modal merge) per TMRoPE chunk → emit `ρ_a · 50` audio tokens per chunk → group those into OQM as a third structure. — Preserves the omnimodal selector. **Recommended.**
- **(c)** Audio anchors live FP16 alongside system tokens; only video lives 4-bit. — Hybrid. Costs ~50·ρ_a · d FP16 per chunk indefinitely; over a 1 h stream at 0.5 fps-equivalent this is meaningful. Probably wrong.

Principled choice given Qwen2.5-Omni's audio:video ratio: **(b)**. The selector's whole edge is the audio→video saliency mapping; cutting audio out of the cache structure leaks that signal at retrieval time.

## Question 5: Quantitative plausibility

Headlines (verbatim from sources):

| Metric | StreamingTOM | OmniZip |
|---|---|---|
| KV/cache compression | **15.7×** (§3.2 footnote) | 1.4× memory red. (§4.1) |
| Speedup | 2× TTFT vs. LiveVLM (§4) | 2.51–3.42× wall-clock; 3.42× prefill on 7B (§4.1) |
| Accuracy (anchor benchmark) | 55.8% / 3.7 score on RVS (§4) | 99.1% retention vs. baseline on AVUT/VideoMME/ShortVid-Bench at 45% retention (§4.1) |
| Modality | video only | audio+video, offline |
| Stage primarily optimized | **decode** (OQM bounds active KV) + prefill (CTR) | **prefill** (pre-LLM token reduction) |

**Compounding vs. competing.** OmniZip cuts the *number of tokens* entering the LLM at prefill. OQM cuts the *bit-width* and *active set* of stored KV at decode. They operate on orthogonal axes:

- **Prefill speedup**: dominated by OmniZip's pre-LLM reduction. OQM's prefill cost (CTR replacement) is now zero because OmniZip provides the selection. Expect ≈OmniZip's 3.42×.
- **Decode TTFT under long history**: dominated by OQM's group retrieval bounding the active KV. Expect ≈StreamingTOM's 2× over a no-OQM baseline.
- **Memory**: multiplicative in principle. OmniZip retains ~45% of tokens; OQM 4-bit quantizes them and only loads top-K groups. Naïvely, `0.45 × 0.25 × top-K-fraction ≈ 5–10%` of the FP16 baseline, *if* nothing degrades.

**Where they compete**: OmniZip's audio score elevates tokens that — by definition — have high cross-modal saliency. Those tokens are the ones most likely to be **queried later** in long-form QA. If OQM's top-K retrieval misses them (because cosine sim on a 4-bit-quantized representative key is noisy), you've spent compression budget on tokens you can no longer find. The audio score and the retrieval score are uncorrelated; that's the tax.

**Where they redundantly compress**: if OmniZip aggressively prunes a low-saliency window, the resulting group is small and well-represented by its mean key. OQM's quantization adds little error here. Compounding works *best* in low-saliency regions and *worst* in dense, important regions — the opposite of what you'd hope.

## Question 6: Failure modes

1. **Quant-distortion of audio-elevated tokens.** OmniZip's selector picks tokens whose information density is high *because* of cross-modal contrast. After 4-bit quantization in OQM and group-mean retrieval, those tokens are stored at coarser precision than their selection score implies. The quality drop will be largest on benchmarks with fine-grained audio-grounded QA (AVUT, DailyOmni). Detection: ablate by storing OmniZip-selected tokens at FP16 and comparing.
2. **Variable-budget × fixed-G mismatch silently truncates.** If you keep OQM's `G=50` per group but OmniZip wants ρ'_v=0.9 (dense window) → 259 tokens, you must drop 209. The dropped tokens are *exactly the ones OmniZip judged worth keeping*. Detection: log per-chunk count(OmniZip-kept) − G; non-zero means quality leak.
3. **Bursty audio events break per-frame budgets.** A 200 ms door slam pushes `S_a(t)` to ceiling for one chunk. With fixed-G this is a no-op (already capped); with variable-G it inflates one group's storage and may starve subsequent chunks under a global memory cap. Detection: long-tail of group sizes; correlate with audio energy spikes.
4. **(Bonus) Streaming quantile estimator drifts at scene cuts.** Replacing OmniZip's global rank with an online quantile is fine in stationary regimes; at hard scene cuts (movie chapters) the running stats are stale, and `ρ'_v(t)` ranks the new regime against the old. Detection: spike in QA error after detected scene boundaries.

## Question 7: Minimum viable experiment

**Goal**: validate that audio-guided selection + OQM is *not* worse than StreamingTOM's CTR + OQM on streaming video, *and* better than StreamingTOM (no audio) on streaming omnimodal QA.

**Checkpoint**: `Qwen2.5-Omni-7B` (matches OmniZip §4.1; rules out 3B confounds).

**Benchmarks**.

- *Streaming-omnimodal*: no native benchmark exists (confirmed across all four sources). Construct **streaming-AVUT** by replaying AVUT clips at 0.5 fps with question-after-end-timestamp protocol borrowed from RVS (StreamingTOM §4 protocol). This is the bare minimum to claim "streaming + omnimodal."
- *Streaming-video-only sanity*: RVS-Ego + RVS-Movie at 0.5 fps. Validates that swapping CTR → OmniZip (with audio muted) at least matches CTR.
- *Offline omnimodal sanity*: WorldSense + AVUT, run OmniZip-only and OmniZip+OQM offline; ensures OQM doesn't tank OmniZip's own numbers.

**Mandatory ablations** (refuses the easy paper):

1. CTR + OQM (StreamingTOM as-is, no audio) — control.
2. OmniZip-causal + OQM, audio muted — isolates "is causal-OmniZip ≥ CTR for video?"
3. OmniZip-causal + OQM, audio on, audio anchors stored FP16 — isolates Failure Mode 1 (quant distortion of audio-elevated tokens).
4. OmniZip-causal + OQM, fixed-G — isolates Failure Mode 2.
5. OmniZip-causal + OQM, variable-G — alternative to (4).
6. Lookahead sweep `L ∈ {0, 1, 2, 4}` TMRoPE chunks — quantifies the causality cost.

**Baselines to beat**: StreamingTOM (video-only on streaming-AVUT — should lose because it ignores audio); LiveVLM (video-only streaming SOTA); OmniZip-offline (upper bound at unconstrained latency).

**Kill criterion**: if OmniZip-causal + OQM with audio on cannot beat CTR + OQM by ≥3 pts on streaming-AVUT *and* match it within 1 pt on RVS, the composition adds complexity without payoff — kill.

**Estimated cost**: ~1k clips × 4 ablation cells × 10 s/clip on a single 7B GPU ≈ 12 GPU-hours. Tractable.

## Verdict

**Risky but feasible, and worth doing precisely because the gap it fills is real.** The mechanical pieces compose: OQM's representative-key averaging + cosine top-K is modality-agnostic, OmniZip's `S_cross` is already per-window, and Qwen2.5-Omni's TMRoPE chunking forces a natural 2 s near-causal boundary that both methods can live with. The two non-trivial blockers are (i) OmniZip's *global* attention + *global* budget normalization, which require principled streaming substitutes (running quantile is the cheapest), and (ii) the dynamic-ρ vs. fixed-G mismatch, which forces a design choice that costs either OmniZip's adaptivity or OQM's latency predictability. Neither is fatal; both are publishable design decisions. The biggest **silent** risk is Failure Mode 1: the selector rewards audio-grounded tokens precisely so a 4-bit-with-mean-key cache can later mis-retrieve them — this is the experiment that decides the paper.

## Open questions for the authors

- **OmniZip authors**: how much of the reported gain comes from *global* top-ρ_a vs. local? Any ablation with windowed attention?
- **OmniZip authors**: section 3.3 normalization — what happens if `S_a` is replaced with a per-window quantile from a streaming estimator? Approximation error?
- **StreamingTOM authors**: was OQM ever stress-tested with variable-G groups, or does the int4 packing assume `G=50`?
- **StreamingTOM authors**: representative key is `mean(K)` — was a learned/PCA pooling considered? Cosine sim on 4-bit-quantized means is the retrieval bottleneck.
- **Both**: any experience with Qwen2.5-Omni's TMRoPE chunk size as the natural quantum for streaming omnimodal compression?

---

*NotebookLM saved note*: "StreamingTOM x OmniZip composition - blockers" (id `35b298d7...`).
