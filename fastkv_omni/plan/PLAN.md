# FastKV-Omni: Reproducing FastKV Analysis on Qwen2.5-Omni

**Source paper:** FastKV — Decoupling of Context Reduction and KV Cache Compression
for Prefill-Decoding Acceleration (arXiv 2502.01068, ACL Findings 2026).
**Target model:** Qwen2.5-Omni-7B (Thinker–Talker omnimodal architecture).
**Mode:** *Analysis-only reproduction* — recreate the diagnostic experiments
that motivate FastKV's design (not the full benchmark suite).
**Skill:** `.agents/skills/paper2code` — stages 1–5.

---

## 1. What FastKV actually does (verified from `FastKV-main/baselines/fastkv/`)

Two decoupled mechanisms inside a Llama/Mistral decoder:

1. **Per-layer KV cache compression** (SnapKV-style, runs at *every* layer)
   - Take the last `window_size=8` queries, score keys via attention.
   - Smooth scores with `avg_pool1d(kernel_size=7)`, sum across GQA group.
   - Keep top-`(max_capacity_prompt − window_size)` keys + the trailing window.
2. **Token-Selective Propagation (TSP)** at one chosen layer `tsp_idx`
   - At that layer only, gather `tsp_length` important token indices.
   - Hidden states + position_ids are sliced for *all subsequent layers*.
   - Result: layers ≤ `tsp_idx` see full context, layers > `tsp_idx` see a
     pruned sequence → big prefill speedup, accuracy preserved.

Defaults observed in code: `window_size=8`, `kernel_size=7`,
`max_capacity_prompt=512`, `tsp_length=2048`, `tsp_rate=0.25`,
`pooling="avgpool"`. `tsp_idx` is the central knob (paper sweeps it).

## 2. Which "analysis" experiments to reproduce

From the FastKV paper, the analysis figures that justify TSP are:

| # | Analysis | What it shows | Output artifact |
|---|----------|---------------|-----------------|
| A1 | Layer-wise attention entropy / sparsity | Attention concentrates as depth grows → safe to prune late | `analysis/A1_layerwise_entropy.png` |
| A2 | Top-K token-set similarity across layers (Jaccard / cosine) | Late layers agree on which tokens matter → one selection suffices | `analysis/A2_topk_jaccard.png` |
| A3 | TSP-layer sensitivity sweep (accuracy vs. `tsp_idx`) | Identifies the "elbow" layer | `analysis/A3_tsp_sweep.csv/.png` |
| A4 | Compression-ratio vs. accuracy trade-off | Pareto curve | `analysis/A4_pareto.png` |
| A5 | Attention sink / window contribution | Validates window+top-K split | `analysis/A5_window_contrib.png` |

*Out of scope* (user said analysis-only): full LongBench/RULER/NIAH runs,
end-to-end latency benchmarking. Keep the hooks but don't execute the suites.

## 3. New analysis specific to omnimodal (the actual research contribution)

Qwen2.5-Omni's Thinker LLM ingests interleaved **text + image + audio + video**
tokens. So beyond reproducing A1–A5, add:

| # | Omni-only analysis | Question answered |
|---|--------------------|-------------------|
| O1 | Per-modality attention mass across layers | Do text layers attend differently to audio vs video tokens with depth? |
| O2 | Per-modality top-K survival under FastKV | Does TSP unfairly prune one modality? |
| O3 | Per-modality optimal `tsp_idx` | Should TSP layer differ when audio dominates vs vision dominates? |
| O4 | Modality-aware KV budget (`max_capacity_prompt` split per modality) | Does balancing budget by modality beat global top-K? |
| O5 | Talker-conditioned analysis | Does Talker-side decoding preserve quality if Thinker prunes? |

These directly address whether FastKV ports cleanly to omnimodal. O1–O3 are
diagnostic; O4–O5 are exploratory and depend on O1–O3 findings.

## 4. Architectural surgery required for Qwen2.5-Omni

FastKV ships `llama_model.py` and `mistral_model.py` — direct monkey-patches
of `LlamaAttention` / `LlamaDecoderLayer`. For Omni:

- Target: `Qwen2_5OmniThinkerForConditionalGeneration` → its `model.layers[i]`
  are `Qwen2_5OmniDecoderLayer` with `Qwen2_5OmniAttention` (GQA, RoPE,
  Hugging Face transformers ≥ 4.52).
- Mirror FastKV's pattern: write `qwen25omni_model.py` that subclasses /
  monkey-patches `Qwen2_5OmniAttention.forward` to expose attention weights
  and call a `kv_cluster.update_kv(...)` (port `baselines/fastkv/utils.py`
  unchanged — math is GQA-agnostic, just thread `num_key_value_groups`).
- TSP slicing in `Qwen2_5OmniDecoderLayer.forward` after `self_attn`,
  identical to lines 252–257 of `llama_model.py`.
- **Extra:** carry a `modality_ids` tensor (0=text, 1=image, 2=video,
  3=audio) alongside `position_ids` so analyses O1–O4 can group scores by
  modality. Build it from the Thinker processor's output token ranges.
- Talker is unaffected — it consumes Thinker's hidden states; analysis stops
  at Thinker.

## 4b. Repo hygiene — vendored, never in-place

`FastKV-main/`, `Qwen2.5-Omni/`, and any other `*-main/` folders the user
drops in are **read-only references**. They are never modified. To extend or
patch them, copy the file(s) we need into `fastkv_omni/vendored/<name>/` and
edit the copy. The patch lives in `fastkv_omni/src/`, the upstream snapshot
stays pristine.

Concrete copies for this project (done once, then untouched):
- `FastKV-main/baselines/fastkv/utils.py`        → `vendored/fastkv/utils.py`
- `FastKV-main/baselines/fastkv/llama_model.py`  → `vendored/fastkv/llama_model.py` (reference for the patch pattern only)
- `Qwen2.5-Omni/cookbooks/*.ipynb` prompt cells  → `vendored/qwen25omni_helpers/prompts.py` (extracted, not whole notebooks)

## 5. paper2code stage mapping

| Stage | File | Action |
|-------|------|--------|
| 1 Acquisition | `.paper2code_work/2502.01068/` | Skip arXiv fetch — local PDF + `FastKV-main/` is ground truth. Run `extract_structure.py` only to stay on-protocol. |
| 2 Contribution | `contribution.md` | Statement: "Decouple per-layer KV compression from one-shot context-reduction at an intermediate TSP layer." |
| 3 Ambiguity audit | `ambiguity_audit.md` | Flag: tsp_idx selection criterion; pooling kernel choice; modality-tagging convention (NEW, not in paper). |
| 4 Code generation | `fastkv_omni/src/` | See §6. Cite paper sections in every file header. |
| 5 Walkthrough | `notebooks/walkthrough.ipynb` | One notebook per analysis A1–A5, O1–O5. |

## 6. File layout (paper2code scaffold conformant)

```
fastkv_omni/
├── PLAN.md                          # this file
├── REPRODUCTION_NOTES.md            # honest log of unspecified choices
├── src/
│   ├── qwen25omni_attention.py      # patched Qwen2_5OmniAttention.forward
│   ├── qwen25omni_layer.py          # patched decoder layer w/ TSP slicing
│   ├── kv_cluster.py                # ported from baselines/fastkv/utils.py
│   ├── modality_tagger.py           # builds modality_ids from processor
│   ├── hooks.py                     # attention-weight capture for analysis
│   └── patch.py                     # entrypoint: load Omni + apply patches
├── analysis/
│   ├── a1_layerwise_entropy.py
│   ├── a2_topk_jaccard.py
│   ├── a3_tsp_sweep.py
│   ├── a4_pareto.py
│   ├── a5_window_contrib.py
│   ├── o1_modality_attention.py
│   ├── o2_modality_survival.py
│   ├── o3_modality_tsp_idx.py
│   ├── o4_budget_split.py
│   └── o5_talker_quality.py
├── configs/
│   ├── omni_baseline.yaml           # full KV
│   ├── omni_fastkv_default.yaml     # FastKV @ tsp_idx=15, cap=512
│   └── sweep_tsp_idx.yaml
├── data/
│   └── samples/                     # ~30 short audio+video+text prompts
├── vendored/                        # READ-ONLY snapshot copies of upstream code
│   ├── fastkv/                      # copy of FastKV-main/baselines/fastkv/{utils,llama_model}.py
│   └── qwen25omni_helpers/          # copies of cookbook prompt-loading helpers only
└── notebooks/
    └── walkthrough.ipynb
```

## 7. Execution order (when user gives the go-ahead)

1. Pull a small held-out set of multimodal prompts (Qwen cookbook
   `video_information_extracting.ipynb` + `universal_audio_understanding.ipynb`
   provide ready-made examples — reuse, don't redownload datasets).
2. Implement `src/patch.py` + `src/kv_cluster.py` first; confirm baseline
   logits match an unpatched run within 1e-4 when no compression is applied.
3. Run A1–A2 (cheap, no sweep) → produces the layer-depth narrative.
4. Run A3 sweep `tsp_idx ∈ {0, 3, 5, 7, 10, 12, 15, 18, 20, 21, 25, 27}`
   — 12 layers across the 28-layer Thinker (Qwen2.5-7B backbone). Endpoints
   intentionally included as sanity bracketing:
   - `tsp_idx=0` → TSP at layer 0 = maximum compression (all 27 downstream
     layers see only the pruned `tsp_length` token set). Expected to degrade
     hardest; calibrates the worst case.
   - `tsp_idx=27` → TSP at the final layer = effectively no propagation
     pruning (only the LM head sees the slice). Should match baseline KL ≈ 0;
     calibrates the no-op case.
   Metric: token-level KL-divergence vs. baseline output distribution (proxy
   for accuracy; no benchmark scores required in analysis-only mode).
5. Run O1–O3 on prompts grouped by dominant modality (audio-heavy vs
   video-heavy vs text-heavy).
6. Compile findings in `notebooks/walkthrough.ipynb`.

## 8. Honest unknowns (must be flagged in REPRODUCTION_NOTES.md)

- The paper does **not** specify `tsp_idx` per model size in detail; pick by
  sweep and document.
- Qwen2.5-Omni Thinker uses **TMRoPE** (Time-aligned Multimodal RoPE — the
  Omni-specific extension of Qwen2-VL's MRoPE that aligns audio with video
  on the temporal axis). `position_ids` is shape `(3, B, T)` carrying
  `(temporal, height, width)` per token. FastKV's reference call
  `torch.gather(position_ids, dim=1, index=tsp_idx)` (in
  `vendored/fastkv/llama_model.py:255`) is wrong on three counts when ported:
    1. Wrong rank: it would gather on the batch axis. Correct call is
       `torch.gather(position_ids, dim=2, index=tsp_idx.unsqueeze(0).expand(3, -1, -1))`.
    2. Per-axis coupling: the same `T`-index must be applied to all three
       rope-axis rows so each kept token's `(t, h, w)` triple stays
       internally consistent — the applied rotary phase is
       `RoPE(t) ⊕ RoPE(h) ⊕ RoPE(w)` and decoupling axes corrupts it.
    3. Audio↔video time alignment is a semantics risk, not a math bug:
       TSP may keep an audio token at `t=37` while dropping the video token
       at the same `t`. Positions remain valid, but the cross-modal bridge
       the model was trained on is severed. **This is exactly what analysis
       O2 should measure** — per-modality survival rate at each TSP layer.
  **Reference:** `scripts/viz_tmrope.py` documents the position-id
  construction and is the canonical source for the (3, B, T) layout.
  **Mitigation:** logits-equivalence smoke test in §7 step 2 — when the
  patch is loaded but compression is disabled (cap = full length), output
  logits must match the unpatched baseline within 1e-4. Any larger gap
  means the gather is wrong.
- Modality boundaries depend on processor version; pin
  `transformers >= 4.52` and snapshot the token-id ranges per sample.
- The talker's autoregressive audio generation is *not* analyzed — only the
  Thinker's KV cache. State this clearly.

## 9. References to consult before coding

- FastKV paper: arXiv 2502.01068 (use local PDF if present).
- `FastKV-main/baselines/fastkv/{llama_model.py, utils.py}` — ground truth
  for the algorithm.
- `Qwen2.5-Omni/cookbooks/` — input formatting per modality.
- HF transformers `Qwen2_5OmniThinkerForConditionalGeneration` source — via
  `context7` MCP query when implementing patches.
