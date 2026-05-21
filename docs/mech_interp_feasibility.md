# Mech Interp on Qwen2.5/3-Omni — Feasibility Assessment & Phased Plan

Verdict: **Yes, very doable** — substantially more so than the source doc implies. You already
own the hardest piece (a forked, instrumented `Qwen2_5OmniThinker` with full access to attention
logits, residual streams, and modality/time token indices). The "port TransformerLens to Qwen-Omni"
framing in the doc overstates the missing tooling. Below is what is feasible *now*, what isn't,
and a 3-tier roadmap.

---

## 1. What you already have (vs. what the doc assumes)

| Doc assumption | Reality in this repo |
|---|---|
| Need to wrap Thinker as `HookedTransformer` (§9.1) | `OmniZip-main/omnizip/modeling_qwen2_5_omni.py` is a full HF Qwen2.5-Omni fork (4797 lines) already intercepting attention logits + hidden states inside the Thinker decoder loop. `output_attentions=True`/`output_hidden_states=True` + `register_forward_hook` on `model.thinker.model.layers[i]` covers everything TransformerLens would expose. |
| Need to track modality per token (§2.1) | OmniZip already computes `audio_indices`, `video_indices`, `text_indices` from `input_ids` — the modality mask is sitting in `videozip_units.py`. |
| Need to map tokens to time via TMRoPE (§4.3) | TMRoPE position IDs are computed inside the modeling file; per-token time is a free byproduct. |
| Need new benchmark infra | `OmniZip-main/lmms-eval` already runs Video-MME, MMAU, OmniBench, MVBench end-to-end. Reuse same harness, just swap the model wrapper for an instrumented one. |
| TransformerLens supports Qwen-Omni | It doesn't (only Qwen3-0.6B text). Skip TransformerLens. Use HF hooks + `nnsight`/`nnterp` if you want a unified interface (nnterp wraps HF directly — doc-supported for 50+ archs). |

**Implication:** the "port the toolbox" effort the doc imagines is mostly already done. The
remaining work is *experimental design + dataset construction*, not framework wrangling.

---

## 2. Per-section feasibility (doc § → effort)

| Doc § | Method | Effort | Blocker |
|---|---|---|---|
| §3 | Layer-wise linear probes (audio/video/cross-modal labels) | **Low** (3–5 days) | Need labeled clips; AudioSet/VGGSound/AVQA cover most |
| §4.1 | Single-modality activation patching | **Low** (already wire compatible) | None |
| §4.2 | Cross-modal patching for fusion circuits | **Medium** | Building audio↔video conflict pairs (the only real bottleneck) |
| §4.3 | TMRoPE-window temporal patching | **Medium**, novel angle | None — you have the time IDs |
| §5.1 | Thinker layer ablation curves | **Trivial** (1–2 days) | None |
| §5.2 | Talker layer ablation w/ WER+NMOS | **Medium-High** | Need SEED-TTS-style speech eval pipeline (not yet in repo) |
| §6 | Per-head modality attention | **Trivial** | OmniZip already computes this aggregated; just unpool it |
| §7 | OmniZip-as-mechanistic-probe | **Medium**, **high novelty** | Cross-validate "what OmniZip prunes" vs "what patching shows is causally needed" — likely paper-worthy |
| §8 | Cross-attention saliency / TDA / SAEs | **Medium** | SAEs need training compute; rest are cheap |
| §9.1 | TransformerLens wrapper | **Skip** | Not worth it — HF hooks suffice |
| §10 (Qwen3-Omni MoE router probe) | Expert routing analysis | **High** (hardware) | Qwen3-Omni 30B-A3B → multi-GPU; doable on a cluster, not a 4090 |

---

## 3. Hard constraints / honest risks

- **Hardware:** Qwen2.5-Omni-7B Thinker fits comfortably on a single 24 GB GPU with attention
  logging on short (≤30 s) clips. Long-context patching (>2 min, full 32 k tokens) will OOM —
  use chunked clips. Qwen3-Omni-30B work is **not** local; budget cluster time.
- **Speech eval (Talker work):** No WER/NMOS pipeline currently in repo. Adding it is a week
  of plumbing (`whisper-large-v3` for WER, `UTMOS`/`NISQA` for MOS). Defer to Tier 3.
- **Dataset bottleneck:** All causal-tracing experiments need *minimal-pair* inputs (audio
  contradicting vision). Existing benchmarks aren't designed for this — expect to construct
  ~500–2000 synthetic pairs (e.g., dub mismatched audio onto AVQA clips, swap object counts).
  This is the single biggest time sink; everything else is mechanical.
- **Streaming/Talker internals:** Talker code path is more entangled than Thinker; the doc's
  §5.2 "ablate Talker-only layers" is real but needs ~1 week of code spelunking before the
  first ablation runs.

---

## 4. Phased plan

### Tier 1 — "Quick wins" (1–2 weeks, single GPU)
Goal: confirm the core mech-interp picture on Qwen2.5-Omni-7B; produce 3–4 publishable
diagnostic figures with existing infra.

1. **Per-layer modality attention map** — fraction of attention mass on audio vs video vs text
   per (head, layer) over MMAU + Video-MME. (Reuses OmniZip's attention logging.)
2. **Layer ablation curves** — zero each Thinker layer's output, plot ΔAccuracy on text-only
   (MMLU), audio-only (LibriSpeech-test), video-QA (Video-MME), and audio-video (OmniBench).
   Expect early layers universal, late layers task-specific.
3. **Linear probes by layer** — train probes on frozen residuals for: speaker ID (audio
   tokens), object presence (video tokens), audio-vision-consistent vs conflicting (cross-modal).
   Reproduces the suppression-onset-layer finding from arXiv 2604.02605 on a different AVLLM.
4. **OmniZip-importance vs probe-importance overlay** — for each clip, plot OmniZip's
   per-token retention score against the gradient-based saliency on the Thinker output. If
   they correlate strongly, OmniZip is implicitly tracking a causal circuit; if not, you
   have a new compression signal.

**Deliverable:** `experiments/tier1_diagnostic/` with 4 plotting scripts + one writeup figure each.

### Tier 2 — "The meat" (3–5 weeks)
Goal: causal claims about audio-vision fusion in Qwen2.5-Omni.

5. **AV-conflict minimal pairs (dataset)** — 1000 clips: (a) original AVQA, (b) audio dubbed
   from a different clip with contradicting answer, (c) video swapped to a contradicting one.
   Two answer options per item.
6. **Cross-modal activation patching** — patch audio-token residuals from "clean" into
   "corrupted" at each layer; same for video tokens. Identify the **fusion-onset layer**
   (where audio patching stops changing the answer) and the **vision-suppression layer**
   (where video patching dominates). Direct replication of arXiv 2604.02605 methodology
   adapted to Qwen-Omni's TMRoPE.
7. **TMRoPE temporal patching** — patch per-2s-block residuals; trace which input window
   drives each output token. Cross-check against OmniZip's window-importance scores.
8. **Attention knockout** — zero attention from text-output tokens to video tokens at deep
   layers; measure restoration of audio-correct answers. This is the strongest causal test
   in the AVLLM paper; it works as-is on Qwen.

**Deliverable:** A short workshop paper (NeurIPS Mech Interp / ICLR BlogPost / a clean
arXiv preprint) — "How Qwen2.5-Omni Fuses Audio and Vision: A Mechanistic Account."

### Tier 3 — "Research scope" (6+ weeks, cluster needed)
9. **Talker dissection** — speech-codec eval pipeline + per-layer Talker ablation;
   empirically validate the Thinker-semantics / Talker-acoustics separation claim from the
   tech report.
10. **Qwen3-Omni MoE router probes** — log expert assignment per (modality, layer); test
    whether MoE makes audio suppression worse or better than dense Qwen2.5-Omni.
11. **OmniZip ↔ mech interp coupling (back to your VideoZip work)** — use the patching
    importance signal *as the pruning signal* for a new variant: "MechZip." Plausibly
    SOTA on the speed/quality frontier and gives the compression work scientific backing.

---

## 5. Concrete next step (if you want to start today)

Build Tier 1 step #1 (per-layer modality attention) — it requires zero new dataset and
reuses OmniZip's attention logging. ~200 lines of script. It will tell you within a day
whether the audio-suppression-by-depth pattern from arXiv 2604.02605 holds for Qwen2.5-Omni
specifically; that result alone determines whether Tier 2 is worth pursuing.

---

## 6. What to drop from the source doc

- §9.1 "Wrap Thinker as HookedTransformer" — skip; HF hooks are sufficient and your fork
  already exposes more than TransformerLens would.
- §8 "Text explanations of internal embeddings via another LLM" — fashionable but low signal
  for this project; defer indefinitely.
- §10 "Coupling with efficiency techniques" — keep, but reframe as Tier 3 step #11 above
  (MechZip), which is concretely actionable rather than aspirational.

---

## References used for this assessment

- arXiv 2604.02605 — *Do Audio-Visual Large Language Models Really See and Hear?* (the methodology you'd replicate)
- arXiv 2511.14465 — *nnterp: A Standardized Interface for Mech Interp of Transformers* (HF wrapper, supports Qwen family)
- arXiv 2509.17765 — Qwen3-Omni Technical Report (MoE Thinker-Talker, 234 ms first-packet)
- arXiv 2503.20215 — Qwen2.5-Omni Technical Report (TMRoPE, Thinker-Talker)
- arXiv 2502.17516 — Survey on Mech Interp for Multi-Modal Foundation Models
- Local: `OmniZip-main/omnizip/modeling_qwen2_5_omni.py`, `videozip_units.py`, `lmms-eval/`
