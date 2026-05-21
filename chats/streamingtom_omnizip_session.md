# Session: StreamingTOM × OmniZip Composition for Streaming Qwen2.5-Omni

## Turn 1 — User
> StreamingTOM: Streaming Token Compression for Efficient Video Understandin /notebooklm explain the 2 stages for me with enough to connect to the others like @omnizip.pdf @omnisift.pdf , use these as sources in the notebook, you can use websearch /ultrathink

## Turn 1 — Assistant
Invoked the notebooklm skill to set up sourcing.

## Turn 2 — Assistant
Tried `notebooklm status` via Bash — `command not found`. Asked user about install / login.

## Turn 2 — User
> might need toa ctivate venv

## Turn 3 — Assistant
Found `venv\Scripts\Activate.ps1`. Activated venv → `notebooklm status` revealed an existing notebook context: **"Keda Tao - Token Compression Papers"** (`736d820f-5af8-46d8-b7f6-877e7694d6e4`).

## Turn 3 — User
> ok authenticated

## Turn 4 — Assistant
Listed sources — found StreamingTOM (arxiv abs), OmniZip (arxiv abs), DyCoke, HoliTom, Survey, Visual Coding already loaded. **OmniSift was missing.** Added `omnisift.pdf` (`3e4a1de4`), waited for indexing (Unicode crash on success message but indexing succeeded).

Ran a scoped `notebooklm ask --save-as-note` against StreamingTOM + OmniZip + OmniSift. Generated note **"StreamingTOM 2-Stage Explanation + OmniZip/OmniSift Comparison"** (`656d82a7…`) covering CTR, OQM, and a comparison table across saliency signal / locus / modality / streaming / training axes.

Key cross-paper finding surfaced: **OmniSift requires fine-tuning** (STE on cross-attention), while StreamingTOM and OmniZip are training-free.

## Turn 5 — User
> just explain the streamingTOM process , is it trainign free? how does it differ from omnizip /ultrathink

## Turn 5 — Assistant
Pulled deeper algorithmic specifics from StreamingTOM source via NotebookLM (revealed source was abstract-only at this point), then synthesized:

**StreamingTOM pipeline** (per incoming frame):
- **CTR (Stage 1, pre-LLM)**: Score visual tokens by adjacent-frame change + intrinsic saliency; keep top-K under fixed per-frame budget; drop the rest. Strictly causal.
- **OQM (Stage 2, post-LLM)**: KV entries quantized to 4-bit + grouped; on-demand group retrieval + dequant during attention. Active KV bounded.

Reported numbers: **15.7× KV compression, ~2× TTFT, 1.2× lower peak memory** vs LiveVLM.

**Training-free?** Yes — fully. No fine-tuning, no learned scorer, no STE.

**Differences from OmniZip** (table covering: locus = hybrid pre+post vs pre-only; saliency = adj-frame Δ vs audio-attention; modality = video vs omni; setting = streaming vs offline; KV = bounded vs untouched).

Closed with: "They could in principle compose — OmniZip's audio-guided selector could replace CTR's visual scorer if you extended StreamingTOM to omnimodal streaming."

## Turn 6 — User
> They could in principle compose — OmniZip's audio-guided selector could replace CTR's visual scorer if you extended StreamingTOM to omnimodal streaming. explore this in a subagent, use notebooklm for sourcing and stuff /notebooklm /ultrathink /ultraplan

## Turn 6 — Assistant
Spawned a general-purpose subagent with detailed brief covering 7 critical questions (causality of OmniZip selector, OQM modality assumptions, sync tension, audio token handling, quantitative plausibility, failure modes, MVE).

**Subagent output**: `docs/streamingtom_omnizip_composition.md`

**Verdict**: risky but feasible, ~12-GPU-hr MVE.

**Key findings reported back**:
1. Mechanical composition works — OQM's `mean(K)` + cosine top-K is modality-agnostic by accident; OmniZip's `S_cross` is already per-window; TMRoPE 2s chunks form a natural near-causal boundary.
2. Non-causality bites in TWO places: unmasked audio attention `A` (fixable) AND **global budget normalization** of `ρ'_v(i)` (mathematically needs whole clip; t-digest streaming quantile proposed).
3. Compounding: `0.45 × 0.25 × top-K ≈ 5–10%` of FP16 baseline — but compounds **best in low-saliency windows, worst in dense ones** (the inverse of what you'd want).
4. **Biggest silent risk**: OmniZip's selector elevates tokens *because* of cross-modal saliency; OQM stores them 4-bit and retrieves by mean-key cosine. **Selection score and retrieval score are uncorrelated** → most informative tokens most vulnerable to retrieval miss.

Side artifacts: added OmniZip full PDF (`6d47c43b`) and StreamingTOM full PDF (`41d40947`) to the notebook (so future queries weren't abstract-only). Saved note "StreamingTOM x OmniZip composition - blockers" (`35b298d7`).

## Turn 7 — User
> ok lets create a folder called "stomxomnizip" and move the above created doc there first

## Turn 7 — Assistant
`mkdir stomxomnizip && mv docs/streamingtom_omnizip_composition.md stomxomnizip/`

## Turn 8 — User
(Read tool invoked on the moved doc — full 168-line file shown.)
> ok what are the blockers? /notebooklm @stomxomnizip/streamingtom_omnizip_composition.md

## Turn 8 — Assistant
Extracted the blockers from the doc:

**Hard blockers (must solve)**:
1. OmniZip's selector non-causal in TWO places: unmasked audio self-attention + global top-ρ_a anchor selection + global budget normalization. The third is the mathematically hard fix.
2. OQM has no slot for audio (defines `M_t = {FP16 sys} ∪ {Q_4(visual)}`). Must extend to tri-structure or fuse audio into visual groups.
3. Dynamic-ρ vs fixed-G mismatch — OmniZip wants per-window adaptive budget; OQM wants `G=50` (where the 15.7× headline comes from).

**Soft blocker**:
4. 2s forced lookahead from TMRoPE — pure per-frame causality impossible; "near-causal, chunk-bounded" is honest framing.
5. Audio handling decision is forced — option (a) "skip audio pruning" breaks `S_cross` (the very mechanism that makes the composition novel). Committed to running OmniZip's full audio path.

**Silent quality risks**:
6. Selector score ⊥ retrieval score (FM1) — *the experiment that decides the paper*.
7. Variable-ρ × fixed-G silently drops the exact tokens OmniZip kept.
8. Bursty audio events break per-frame budgets.
9. Streaming quantile drifts at scene cuts.

## Turn 9 — User
> ok how can we handle this? i want audio-visual input so we need a tri modal pruning. use web search /ultrathink /ultraplan

## Turn 9 — Assistant
Spawned a subagent to design tri-modal pruning that resolves the 9 blockers, with web search for recent related work + NotebookLM grounding.

**Subagent output**: `stomxomnizip/trimodal_pruning_design.md`

**Headline decisions**:
- **Tri-modal = (A) audio × (V_s) video-spatial × (V_t) video-temporal frame selection.** Three other interpretations rejected on-page (text-question; KV-tier; joint-pool / EchoingPixels — requires training).
- **Single most novel piece**: Quad-structure OQM = `{sys-FP16, audio-q4, video-spatial-q4, video-temporal-summary-FP16}`, each modality with its own group budget G_m and own retrieval bank. NotebookLM probe of StreamingTOM §3.4 confirmed int4 packing math is agnostic to G value. **No prior streaming KV work decomposes by modality this way.**

**Blocker scoreboard**:
- **6/9 cleanly resolved**: Q1 sliding-window audio attn, Q1b t-digest streaming quantile, Q2 quad-OQM, Q3 TMRoPE-forced lookahead, Q4 audio-anchor as own group, FM2 explicit-fixed-G + overflow logging.
- **3/9 mitigated-with-instrumentation only**: FM1 quant-distortion of audio-elevated tokens (needs ablation), FM3 bursty audio spillover, FM4 quantile drift at scene cuts.

**Most surprising web find — and a real threat**:
- **EchoingPixels (arxiv 2512.10324, Dec 2025)** — co-pools audio+video into single multimodal token pool with trained Cross-Modal Semantic Sieve + Sync-RoPE. **5–20% retention, 2–3× speedup.** Direct competitor that didn't exist when prior doc was written. *Trained* → training-free niche preserved but quality bar rose.
- **Second surprise**: OmniSift's selector has **4.85M trained parameters** and explicitly defends *asymmetric* video→audio guidance. Kills any "tri-modal = bidirectional cross-modal" interpretation under training-free constraint.

**New NotebookLM sources added**: EchoingPixels (`46bc33d8`), Video-SALMONN-S (`03785de2`), STC (`d1145bb4`).
**Notes saved**: EchoingPixels architecture (`2ac856e6`), Video-SALMONN-S TTT (`9f113bdd`), STC hierarchy (`d57400e9`).

## Turn 10 — User
> ok what to do now?

## Turn 10 — Assistant
Offered four options via AskUserQuestion: FP16-anchor ablation (the MVE), read EchoingPixels deeply, build streaming-AVUT benchmark, implement OmniZip-causal selector.

User initially picked FP16-anchor ablation but interrupted and reframed.

## Turn 11 — User
> whats the next best move? what if we can do all at once ? like an ablation? /ultrathink

## Turn 11 — Assistant
Reframed: **the right first move is a unified evaluation harness, not any single experiment**. The four candidate experiments have hidden dependencies; the harness collapses them.

**Proposed harness config axes**: pre-LLM selector × post-LLM memory × audio handling × lookahead L. Naïve full factorial = 192 cells = ~25 days on one GPU.

**Three-phase sweep**:
1. Screening (Plackett-Burman, ~12 cells, ~36 GPU-hr) — main effects.
2. Focused factorial on axes that mattered (~16 cells, ~48 GPU-hr).
3. FM1 decider (~4 cells, ~12 GPU-hr) — definitive head-to-head.

**Total: ~100 GPU-hr, ~4 days on one GPU.** Entire paper's experimental backbone in one harness.

Offered to spawn a subagent to spec the harness (`stomxomnizip/ablation_harness_spec.md` + stub skeleton).

## Turn 12 — User
> make a folder aclled "chats" and put this session's chat in a mkardown inside that folder

## Turn 12 — Assistant
Created `chats/streamingtom_omnizip_session.md` (this file).

---

## Artifacts produced this session

**Files (in `C:\Users\Armaan\Desktop\PURS\`)**
- `stomxomnizip/streamingtom_omnizip_composition.md` — feasibility analysis, 9 blockers, MVE
- `stomxomnizip/trimodal_pruning_design.md` — tri-modal design resolving 6/9 blockers + quad-OQM
- `chats/streamingtom_omnizip_session.md` — this transcript

**NotebookLM notebook `736d820f` ("Keda Tao - Token Compression Papers"):**

Sources added during session:
- `omnisift.pdf` — `3e4a1de4`
- `omnizip.pdf` — `6d47c43b`
- StreamingTOM full PDF — `41d40947`
- EchoingPixels (2512.10324) — `46bc33d8`
- Video-SALMONN-S (2510.11129) — `03785de2`
- STC (2512.00891) — `d1145bb4`

Notes saved:
- "StreamingTOM 2-Stage Explanation + OmniZip/OmniSift Comparison" — `656d82a7`
- "StreamingTOM x OmniZip composition - blockers" — `35b298d7`
- "EchoingPixels CS2 architecture and training/streaming status" — `2ac856e6`
- "Video-SALMONN-S TTT memory vs OQM compatibility" — `9f113bdd`
- "STC hierarchical ViT-cache + LLM-pruner architecture" — `d57400e9`

## Open thread

Awaiting user decision on whether to spawn the harness-spec subagent or skip the spec and code directly.
