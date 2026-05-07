# Professor Progress Updates

**Owner:** Armaan Amatya · **Doc type:** Rolling advisor-facing update log · **Most recent entry:** 2026-05-07

This file consolidates all professor updates for the OmniZip / L6 cache project. **Most recent entry is at the top.** Earlier entries are preserved verbatim below for context.

**Important reframe (2026-05-07):** the 118-question accuracy numbers in Updates 1 and 2 (Baseline 29.7% / OmniZip 30.5% / OmniZip+L6 28.8%) are within run-to-run noise on the 10× repeat matrix at `10x/qwen25_matrix_gpu7_all7_snapkv/`. The matrix shows OmniZip and baseline are statistically tied at T=0.1 (0.312 vs 0.311, within 0.1σ). The L6 mechanistic findings (question-invariance, AUC vs OmniZip mask, L6 ≫ L14) remain valid; the *contribution framing* changes — see Update 3.

For authoritative current numbers see `docs/research_log.md §Key Results Summary` and `docs/PROJECT_BRIEFING.md §2a`.

---

## Update 3 — 2026-05-07 (post 10× matrix)

### Short Version

Hi Professors,

One important reframe and a revised 1-week plan.

**What changed.** I mined the 10× repeat benchmark matrix at `10x/qwen25_matrix_gpu7_all7_snapkv/` (10 runs × 2 temperatures × 5 methods, full benchmark, prefill + per-question VRAM logged). It shows three things that change how I should be framing the project:

1. **OmniZip and baseline are statistically tied on accuracy** — 0.312 vs 0.311 at T=0.1, within 0.1σ. The earlier "OmniZip 30.5% > baseline 29.7%" gap from the 118-Q snapshot was single-run noise. OmniZip's real contribution is **35% prefill speedup + 38% VRAM reduction at parity accuracy**, not an accuracy win. The "beat OmniZip" target is therefore a Pareto target on speed/memory/compression-ratio — the L6-cache regression of `28.8% vs 30.5%` from Update 1 also lives within this noise band.
2. **MixKV at budget=256 is broken on Qwen-Omni** — −11 accuracy points vs baseline (0.20 vs 0.31) with only 7% prefill speedup. Audio+video context overflows the 256-token KV cache. MixKV is unusable as a comparator or as a stacking partner until I sweep budget upward (512 / 1024 / 2048).
3. **ReDiPrune > DivPrune at matched compression** — 0.218 vs 0.203 at T=0.1 with the same 0.5 keep ratio. +1.5 pts from text-relevance over pure visual diversity. Validates text-query relevance as a useful signal worth incorporating into any future fusion.

**The new gating experiment is the audio-guidance-OFF ablation in OmniZip** — replace the audio-derived per-group video budget with random / uniform selection at fixed ρ_v=0.6, all else identical. Two outcomes, both decisive for the rest of the project:

- random ≈ OmniZip → audio guidance is a story not a contribution; pivot to adaptive ρ + diversity (the budget allocator wins).
- random ≈ baseline → audio signal carries the gain; pursue 3-signal fusion (L6 + encoder + ReDiPrune-style query relevance).

**Revised 1-week plan:** (1) audio-OFF ablation, (2) MixKV budget sweep, (3) ReDiPrune α sweep at fixed subset=0.5. After that the original three threads (B Hybrid + adaptive, A VideoZip, C Compression × spec-decoding) stay queued, but with adjusted priorities — see detail below.

The L6 mechanistic story from Update 2 is unchanged. What shifts is the contribution framing: L6's value is no longer "preserves OmniZip's accuracy gain" (there is no such gain over baseline), it is "a question-invariant, video-level surrogate for OmniZip's audio mask that enables offline precomputation and cross-query amortization." The multi-turn caching demonstration becomes more important under this framing, not less, because it's the cleanest setting where L6 caching genuinely beats OmniZip on serving cost.

Best,
Armaan

### Detailed Version

#### Authoritative numbers — 10× repeat matrix

10 runs × 2 temperatures (0.1 / 0.9) per method on the full benchmark. Eval config: `fps=2.0`, `max_pixels=100352`, `max_frames_videomme=768`, `max_frames_other=128`, `dtype=bfloat16`.

| Method | Acc T=0.1 (mean ± std) | Acc T=0.9 (mean ± std) | Prefill ms | E2E ms | Process Peak GB | Frame keep |
|---|---|---|---:|---:|---:|---:|
| baseline | **0.311 ± 0.013** | 0.333 ± 0.030 | 2481 | 2592 | 34.1 | 1.000 |
| omnizip (ρ_a=0.3, ρ_v=0.6, g=3) | **0.312 ± 0.013** | 0.328 ± 0.034 | 1614 | 1802 | 21.0 | 1.000 |
| mixkv (budget=256, snapkv) | 0.200 ± 0.011 | 0.220 ± 0.034 | 2304 | 2391 | 24.1 | 1.000 |
| divprune (subset=0.5, frame mode) | 0.203 ± 0.012 | 0.245 ± 0.034 | 1200 | 1295 | 21.4 | 0.500 |
| rediprune (α=0.5, subset=0.5, frame) | 0.218 ± 0.009 | 0.264 ± 0.017 | 1257 | 1356 | 21.4 | 0.500 |

#### Revised priority stack

The 10× matrix changed the priority stack. Relative to Update 2's three threads:

- **B: Hybrid + adaptive extensions** — the linear hybrid `α·L6 + (1−α)·encoder` part is now lower priority (L6 and encoder are correlated, and the encoder's accuracy gain over baseline is within noise so a hybrid can't unlock something that isn't there). **Higher priority for adaptive ρ** if the audio-OFF ablation comes back as outcome H1 (random ≈ OmniZip — i.e. the budget allocator is the real contribution). The multi-turn caching demonstration is independent of the ablation outcome and remains queued — likely now more important since L6's contribution narrative leans on it.
- **A: VideoZip implementation** (~2 weeks). Reframe the baseline as **OmniSIFT** (training-free vs trained 4.85M params), not OmniZip on accuracy.
- **C: Compression × speculative decoding** (3+ weeks): unchanged.

**Earliest possible Pareto-improvement-over-OmniZip win:** OmniZip + (working) MixKV stacked, once the MixKV budget is fixed. Pre-LLM × post-LLM compression on orthogonal axes; expected ~1.5× further latency reduction at parity accuracy. This is now the cleanest publishable single-method result on the path.

#### Three things I'd value your read on (carried forward from Update 2, still open)

- Should the L6 result and VideoZip be one paper (training-free omni compression, two directions, mechanistic story) or two?
- Is the multi-turn caching experiment compelling enough on its own to be a standalone result, or only useful as a section?
- For the mechanistic framing, would a TransformerLens-style wrapper of the Qwen Thinker (residual streams + per-layer hooks exposed) be a deliverable in itself, given how little tooling currently exists for omni models?

---

## Update 2 — May 2026 (post-L6 result, mechanistic framing + VideoZip plan)

### Short Version

Hi Professors,

Following up on the previous update on the Layer-6 cache result. Three things have happened since then that I think are worth flagging.

**1. The L6 finding now has a mechanistic story behind it, not just numbers.** I wrote a layer-by-layer interpretability analysis of the Qwen2.5/Qwen3 Thinker–Talker stack (`Mechanistic Layer-Level Analysis... .md`) that frames the L6 result as one observation inside a broader picture: which Thinker depths carry low-level perceptual structure, which carry cross-modal fusion, and which carry semantic reasoning. The writeup ports TransformerLens-style probing, activation patching, and causal tracing onto Qwen-Omni and lays out what to probe at which depth. This gives the L6 cache a "why" — early layers express geometric content saliency aligned with the LLM's input space, deeper layers (L14+) express reasoning structure that no longer looks like the pruning mask. The interpretability framing turns the compression result into a probe of the model's information flow.

**2. VideoZip — the training-free inverse of OmniZip — is fully specified.** I drafted `videozip_plan.md` with the full algorithm: video saliency drives the audio compression ratios, with audio anchoring the video selection (bidirectional cross-modal anchoring). The literature gap is concrete: OmniSIFT (Feb 2026) does video-guided audio but trains 4.85 M parameters; VideoZip would be the training-free counterpart. Four functions are spec'd at code level (`omnizip_video_saliency`, `omnizip_audio_compress`, `omnizip_istm_audio_anchored`, `omnizip_videozip`), wired into a `guide_mode: audio | video | adaptive` config switch on `modeling_qwen2_5_omni.py`. Implementation order, ablation grid (direction of guidance, audio-anchor β sweep, per-group vs global, attention-entropy adaptive routing), and risk list are all written.

**3. The repo is now navigable.** I consolidated all notes into `docs/` and wrote `docs/PROJECT_BRIEFING.md` as a single-page entry point with the headline numbers, repo map, verified citations (OmniZip arXiv 2511.14582, OmniSIFT 2602.04804, FastKV 2502.01068, DivPrune 2503.02175, ReDiPrune 2603.24680, MixKV 2510.20707, AngelSlim 2602.21233, Qwen2.5-Omni 2503.20215, Qwen3-Omni 2509.17765, WorldSense 2502.04326, Daily-Omni 2505.17862), and the three live research threads.

**Where I'd value your input:** I have three viable next-step threads and want guidance on ordering.

- **(A) VideoZip implementation** — Highest novelty (training-free inverse of OmniSIFT), but a bigger build (~2 weeks to first ablation). Best fit if the goal is a standalone paper.
- **(B) Hybrid + adaptive extensions to the L6 result** — `α·L6 + (1−α)·encoder` audio score, saliency-entropy-driven adaptive ρ, video-side importance+similarity. Smaller, faster wins, all building on the existing L6 infrastructure. Best fit if the goal is to strengthen the L6 paper.
- **(C) Compression × speculative decoding** — Audio-guided dual masks (drafter vs verifier), modality-aware acceptance. Higher upside but earliest-stage; needs a baseline against MSD ("Speculative Decoding Reimagined for Multimodal LLMs") first.

My instinct is **B before A** — finish telling the L6 story first, then write VideoZip as the natural follow-up — but I want to check before I commit a few weeks to it.

The multi-turn caching experiment from the previous update is still planned; it's the cleanest demonstration that L6's question-invariance translates to a real serving-time win.

Best,
Armaan

### Detailed Version

#### Recap of where we left off (now superseded by 10× matrix — see top banner)

Previous update established that Thinker Layer 6 of Qwen2.5-Omni encodes a question-invariant cross-modal saliency signal that aligns with OmniZip's audio keep mask (AUC `0.6528 ± 0.1105` vs L14's `0.5253 ± 0.0662`; L6 wins on `58/59` WorldSense questions) — that part still stands. The 118-question accuracy snapshot (`Baseline 29.7% / OmniZip 30.5% / OmniZip+L6 28.8%`) is now superseded by the 10× repeat matrix in Update 3 above. **Reframed claim:** L6's value is "a question-invariant, video-level surrogate for OmniZip's audio mask that enables offline precomputation and cross-query amortization." Paired prediction agreement (`114/118 = 96.6%`) and identical efficiency profile remain valid.

#### What was new in this update

##### Mechanistic framing for the L6 result

The L6 result was previously stated as an empirical curiosity. I now have a layer-level interpretability writeup that gives it a mechanistic interpretation:

- **Architectural anatomy** for the Thinker–Talker stack of Qwen2.5-Omni (28-layer dense Thinker + 4-layer Talker decoder, TMRoPE temporal positions) and Qwen3-Omni (MoE Thinker–Talker with multi-codebook causal-ConvNet).
- **What to probe at which depth.** Early Thinker layers (L0–L6) — modality-input geometry, cross-modal token alignability. Mid (L10–L14) — fusion / event binding. Late (L20–L27) — task semantics / answer formation. The L6 result is consistent with this picture: the layer that aligns with OmniZip's pruning is the layer where modality-input geometry is most legible, *before* the fusion stack mixes modalities into reasoning structures that no longer respect the pruning mask.
- **A protocol for the next probes.** Linear probes on residual streams per layer, activation patching across modalities to locate fusion circuits, causal tracing along TMRoPE time slots for streaming. Token-compression itself is repurposed as a mechanistic probe: which layer's saliency, when used to prune, preserves task accuracy? That is exactly what the L6 sweep was, but now it has a place in a larger map.

The methodological contribution beneath the empirical one: **Gini concentration was ambiguous between L6 and L14; AUC against the actual downstream keep mask was decisive.** Concentration without alignment would have shipped the wrong layer. Saving this so future layer-probing work (VideoZip's video-saliency layer pick, Qwen3-Omni MoE expert routing, hybrid-score layer choice) builds on the right metric hierarchy.

##### VideoZip plan

OmniZip uses audio to guide video pruning. For video-primary tasks (action QA, long-form video understanding), the information density is reversed: video should guide audio. OmniSIFT (arXiv 2602.04804, Feb 2026) does exactly that direction but trains 4.85 M parameters. VideoZip closes the training-free gap with one extra twist: **bidirectional cross-modal anchoring** — video saliency drives the audio compression ratios, and audio embeddings anchor which video tokens to keep (β-weighted into the dpcknn anchor scoring inside ISTM).

Implementation specified at function level: `omnizip_video_saliency` (frame-aggregated attention importance → per-group retention scores), `omnizip_audio_compress` (per-group audio compression with video-derived ratios), `omnizip_istm_audio_anchored` (audio embeddings guide dpcknn anchor selection, β=0.3 default), `omnizip_videozip` (entry point). All four are designed to live alongside the existing `omnizip_units.py` so a `guide_mode: audio | video | adaptive` switch in `omnizip_config` selects between OmniZip, VideoZip, and an attention-entropy-driven adaptive router.

Four ablations are written: direction of guidance, β sweep, per-group vs global, adaptive routing.

##### Repository reorganization

All `.md` notes consolidated under `docs/`. New `docs/PROJECT_BRIEFING.md` is a single-page entry: research goal, headline result table, method stack, full repo map (including `mechinterp/`, `vizzing/`, vendored `*-main/` reference trees), verified citations with arXiv IDs, three live threads, and pointers to HPC sync and bench output layout. `mechinterp/outputs/` now holds the canonical depth-curve PNGs (cross-modal curves, self-vs-cross, Gini-by-depth, cross-modal heatmap, last-token attention by depth), with historical copies preserved at `vizzing/depth_curve/` for audit.

---

## Update 1 — Initial L6-cache layer sweep + replacement benchmark

### Short Version

Hi Professors,

Quick update on the OmniZip / early-layer saliency project. I ran a layer sweep over Qwen2.5-Omni Thinker layers `[0, 1, 3, 6, 10, 14, 20, 27]` to see whether an internal saliency signal could replace OmniZip's audio saliency score. I started with Spearman, Gini, and score spread because I first wanted to know two basic things: whether the signal changes across questions, and whether it becomes concentrated enough to support pruning. The first finding was that the ranking is almost completely question-invariant across prompts, which suggested caching might be possible. The second finding was that the first strong concentration jump happens at Layer 6, with another peak at Layer 14. That left L6 and L14 as the main candidates, but those descriptive metrics were not enough to tell which one actually matches OmniZip.

Because of that ambiguity, I added stronger pruning-oriented metrics: ROC AUC against OmniZip's actual audio keep mask, top-k Jaccard overlap, separation at the pruning threshold, and temporal autocorrelation. The reason for this second pass was that I needed to test not just whether the signal looks structured, but whether it actually reproduces OmniZip's keep/drop behavior. That was the decisive step: Layer 6 gets `0.653 +/- 0.111` AUC, while Layer 14 is `0.525 +/- 0.066` (near random), and L6 beats L14 on `58/59` questions. So L6 is the only layer that looks both cacheable and aligned with OmniZip.

I then used precomputed Layer-6 saliency to replace OmniZip's native audio saliency in the real pipeline. On the full 118-question benchmark, OmniZip+L6-cache gets `34/118 = 28.8%` versus `36/118 = 30.5%` for stock OmniZip, while preserving essentially the same efficiency: `1470 ms` prefill versus `1455 ms`, with the same `18.5 GB` peak allocated VRAM. In the paired comparison, the cached-L6 system gives the exact same prediction as OmniZip on `114/118` questions, so it is acting as a high-fidelity replacement signal rather than a loose approximation.

> *Note 2026-05-07:* the 118-Q gap is now known to be within run-to-run noise on the 10× matrix (Update 3); OmniZip ≈ baseline at T=0.1. The paired prediction agreement and the L6 mechanistic findings still stand.

I think this is a good basis to build on because it suggests an early Thinker layer already contains a question-invariant, cacheable saliency signal that overlaps with OmniZip's pruning decisions. The next missing experiment is the multi-turn setting, where the same video is queried multiple times and the cache can actually be reused.

Best,
Armaan

### Detailed Version

Hi Professors,

I wanted to send a more complete update on the Layer-6 cache idea for OmniZip.

The main question I tested was whether an early Thinker-layer signal inside Qwen2.5-Omni could replace OmniZip's handoff audio saliency signal. OmniZip normally uses saliency derived from the audio encoder's own self-attention. My hypothesis was that an early Thinker layer might already encode a ranking of which audio tokens matter most, and that this ranking might be stable enough to precompute once per video and reuse across questions.

To test that, I ran a layer sweep over `[0, 1, 3, 6, 10, 14, 20, 27]` and looked at several metrics. The first pass used cross-question Spearman correlation, Gini concentration, and score spread. Those gave two important early findings:

- The saliency ranking is almost completely question-invariant, which means it looks more like a video-level semantic saliency signal than a question-specific relevance map.
- The first strong concentration jump happens at Layer 6, with another strong peak at Layer 14.

That left an ambiguity: L6 and L14 both looked promising if I only looked at concentration. So I added stronger metrics that are closer to the actual pruning problem:

- ROC AUC against OmniZip's real audio keep mask
- Top-k Jaccard overlap between question-specific selections
- Separation between kept and dropped tokens at the pruning threshold
- Temporal autocorrelation to see whether the signal forms coherent blocks instead of noise

The AUC test turned out to be the decisive one. Layer 6 gets `0.6528 +/- 0.1105` versus OmniZip's actual audio keep mask, while Layer 14 gets only `0.5253 +/- 0.0662`. Layer 6 beats Layer 14 on `58/59` questions. So although L14 looks concentrated by Gini, it is not actually aligned with OmniZip's real pruning behavior. That is why I ended up focusing on L6.

After that, I ran the actual replacement experiment. I precomputed Layer-6 saliency offline once per video using the stock Qwen2.5-Omni model, then patched OmniZip so that its `attn_logits` audio saliency input comes from the cached Layer-6 scores instead of the audio encoder's native score. Everything else in OmniZip stayed the same: same pruning ratios, same video compression, same token merging, same Thinker and decoder.

On the full 118-question benchmark *(superseded by 10× matrix; see Update 3 — gaps below are within run noise)*:

- Baseline: `35/118 = 29.7%`
- OmniZip: `36/118 = 30.5%`
- OmniZip + cached L6: `34/118 = 28.8%`

Efficiency is essentially unchanged relative to OmniZip:

- OmniZip prefill: `1455 ms`
- OmniZip + cached L6 prefill: `1470 ms`
- Both use `18.5 GB` peak allocated VRAM

The paired comparison is the most encouraging part. Cached L6 and OmniZip give the exact same prediction on `114/118` questions. Only 2 questions flip from correct under OmniZip to incorrect under the cached version, and there are no broad failure patterns. So the right interpretation is not that cached L6 is a weaker system overall, but that it is a high-fidelity surrogate for OmniZip's audio saliency.

Why I think this is promising:

1. It reveals a mechanistic result: an early Thinker layer already contains a usable cross-modal saliency signal.
2. That signal is effectively question-invariant, which makes caching realistic.
3. It is strong enough to replace OmniZip's native audio saliency with only a very small paired degradation *(later shown to be within 10×-matrix noise)*.
4. It opens the door to a real systems advantage in multi-turn settings, where the same video is queried multiple times.

So my current view is that this is already a strong insight plus proof-of-concept. The missing experiment that would really elevate it is to measure the multi-turn scenario directly: for repeated queries on the same video, the Layer-6 cache is reused once, while OmniZip still has to recompute its native saliency every time.

Best,
Armaan
