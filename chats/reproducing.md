# Reproducing FastKV on Qwen2.5-Omni — Planning & First Implementation

Session transcript covering the planning phase and the initial TSP-only patch.
Project root: `C:\Users\Armaan\Desktop\PURS`. Target output dir: `fastkv_omni/`.

---

## Turn 1 — Initial request

**User:**
> use `.agents/skills/paper2code` to reproduce "FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration" (`FastKV-main/`). I want to similarly reproduce the experiments done in this paper but on the omni modal architecture (`Qwen2.5-Omni/`). I just want to infer information of analysis like this. /ultrathink /ultraplan

**Assistant — produced `fastkv_omni/plan/PLAN.md`** with:
- Analysis-only scope (skip LongBench/RULER/NIAH benchmark scoring)
- Five FastKV-paper analyses to reproduce: A1 layer-wise entropy, A2 top-K cross-layer agreement, A3 TSP-idx sweep, A4 compression Pareto, A5 window contribution
- Five new omni-only analyses: O1 per-modality attention mass, O2 modality survival under FastKV, O3 per-modality optimal tsp_idx, O4 modality-aware budget, O5 Talker-conditioned quality
- paper2code stage mapping (1–5)
- File scaffold under `fastkv_omni/`
- Honest unknowns flagged: tsp_idx selection, MRoPE position-id slicing, modality boundaries

---

## Turn 2 — Scope clarifications

**User:**
> do not edit any official codebases of papers i have pasted in, just copy them to the targeted script folder and use that.
> 1. whats analysis only?
> 2. lets add 0,3,7,12,18,21,27 as well /ultrathink

**Assistant:**
- Saved feedback memory `feedback_never_edit_paper_source_trees.md` — vendor copies into the project folder; treat `*-main/` dirs as read-only.
- Defined **analysis-only** = analysis figures + ablations that explain *why* the design works; skip benchmark scoring and latency tables. If the analysis assumption holds, then promote to a full benchmark run.
- Expanded TSP sweep to `{0, 3, 5, 7, 10, 12, 15, 18, 20, 21, 25, 27}` — 12 layers across the 28-layer Thinker. Endpoints kept on purpose: `tsp_idx=0` = max-compression worst case, `tsp_idx=27` = no-op calibrator (KL ≈ 0).
- PLAN.md updated: new §4b "Repo hygiene", `vendored/` added to §6 file layout, §7 step 4 uses expanded sweep with reasoning for endpoints.

---

## Turn 3 — Cross-checking with existing work

**User:**
> is there anything similar we did in `scripts/`? /ultrathink

**Assistant — overlap audit:** ~50–60% of the plan was already implemented in `scripts/`.

| Plan item | Existing script | Overlap | Action |
|---|---|---|---|
| A1 layer-wise entropy | `viz_attention_depth_curve.py`, `viz_attention_encoders.py` | ~85% | Reuse (Gini ≈ entropy proxy) |
| A2 top-K cross-layer agreement | `viz_crossmodal_spearman.py` (already has L6 vs L14 Jaccard) | ~75% | Reuse + extend |
| A3 TSP-idx sweep | `viz_layer_depth_experiment.py` (8 depth checkpoints, multi-GPU runner) | ~50% | Reuse runner; swap metric to KL-vs-baseline |
| A4 Pareto | `plot_prefill_score_memory_time.py` + `run_qwen_omni_benchmark_matrix.py` | ~60% | Reuse runner; add FastKV as a method |
| A5 window contribution | none | 0% | Net-new |
| O1 modality attention mass | `viz_attention_depth_curve.py` (modality share heatmap) | ~95% | Done |
| O2 modality survival | `viz_attention_omnizip.py` mask-capture pattern | ~70% | Reuse; swap mask source |
| O3 per-modality tsp_idx | partial | ~30% | Add modality grouping on A3 |
| O4 modality budget | none | 0% | Net-new |
| O5 Talker quality | none | 0% | Net-new |

**Reusable infrastructure noted:** Thinker attention hooking pattern, modality token tagging, mask-capture monkey-patch (OmniZip), multi-GPU metadata-driven runner, benchmark/Pareto runner with prefill_ms/VRAM tracking, TMRoPE viz reference.

---

## Turn 4 — Terminology correction (TMRoPE, not MRoPE)

**User:**
> why mrope correct slicing? doesnt `Qwen2.5-Omni/` use TMROPE?

**Assistant — clarified:**
- **RoPE** — 1-D position. `(B, T)`. Llama/Mistral.
- **MRoPE** (Qwen2-VL) — 3-axis `(temporal, height, width)`. `(3, B, T)`.
- **TMRoPE** (Qwen2.5-Omni) — MRoPE extended so audio shares the temporal axis with video at matching wall-clock time. Still `(3, B, T)`.

**Why "correct slicing" still matters under TMRoPE:**
1. **Wrong tensor rank.** FastKV's `gather(dim=1, …)` would gather on the batch axis. Correct call: `gather(dim=2, index=tsp_idx.unsqueeze(0).expand(3, -1, -1))`.
2. **Per-axis coupling.** Same `T`-index applied to all three rope-axis rows so each kept token's `(t, h, w)` triple stays jointly meaningful — applied phase is `RoPE(t) ⊕ RoPE(h) ⊕ RoPE(w)`.
3. **Audio↔video time alignment is a semantics risk.** TSP can keep an audio token at `t=37` while dropping the video token at the same `t`. Math stays valid; the cross-modal bridge the model was trained on is severed. Exactly what analysis O2 measures.

**Plan edit:** PLAN.md §8 now spells out the three concrete failure modes, the correct gather call, and the logits-equivalence smoke test that catches it. Cross-references `scripts/viz_tmrope.py`.

---

## Turn 5 — What's left to explore

**User:**
> is there anything similar we did in `scripts/`? /ultrathink what to do now that has not been explored?

**Assistant — unexplored surface (everything in `scripts/` is observational; nothing modifies the forward pass):**

1. **Decoupling claim itself, on Omni** — nobody runs "drop tokens at L=15, see if generation matches baseline."
2. **Output-level fidelity** — every existing metric is internal; none measure `KL(patched, baseline)` or generated-answer match.
3. **Decode-time behavior** — existing scripts run a single prefill; none capture how compression affects the *N*-th generated token vs the *5*-th.
4. **Talker-side audit (uniquely Omni)** — does Thinker compression degrade Talker audio (WER/MOS)?
5. **Time-slice pruning under TMRoPE** — prune `(temporal-position, modality)` not individual tokens.
6. **Cumulative / multi-layer TSP** — your L6 ≈ L14 finding suggests a second TSP at L14 may be redundant or beneficial.
7. **Modality-aware pooling** — FastKV's `avg_pool1d(kernel_size=7)` crosses modality boundaries.
8. **Modality-balanced KV budget** — global top-K starves text/audio when video dominates.
9. **GQA group interaction** — existing scripts uniformly average heads; FastKV sums within KV groups.
10. **Window position under question-at-end prompts** — FastKV's anchor window is the question; behavior on prepended-question prompts is untested.

**Recommended first move:** Build the **TMRoPE-correct Thinker patch + KL-vs-baseline harness** first. Reasons: (a) only enabling piece — items 1–8 all require an actual patched forward pass; (b) minimum artifact to say "FastKV works/doesn't on Omni"; (c) the KL harness is reusable across all later experiments.

Smallest meaningful sequence:
```
src/qwen25omni_fastkv.py    # vendored fastkv/utils.py + TMRoPE-correct gather
analysis/kl_vs_baseline.py  # measure logit divergence on N prompts
analysis/a3_tsp_sweep.py    # the 12-layer sweep — first scientific output
```

---

## Turn 6 — Implementation

**User:**
> ok lets do this then first. dont do any other items, just this. will all the files created and edited be in `fastkv_omni/`? /ultrathink

**Assistant — confirmed yes, all in `fastkv_omni/`. Implementation decisions:**

- **TSP-only first.** Per-layer SnapKV-style cache compression deliberately omitted (cleaner first experiment).
- **No attention monkey-patch.** Score TSP indices inside the *decoder layer* forward by re-projecting hidden_states to Q/K (small cost, simple code).
- **Patch `model.thinker.model.forward`** so post-TSP `position_ids`, `position_embeddings`, and `attention_mask` propagate correctly — this is where TMRoPE-correct slicing lives.
- **Outputs land in `fastkv_omni/outputs/`** (not `vizzing/`) to keep everything inside the project folder.

### Files created

```
fastkv_omni/
├── plan/PLAN.md                          (updated)
├── src/qwen25omni_fastkv.py              528 lines  — patch + embedded smoke test
├── analysis/kl_vs_baseline.py            244 lines  — KL harness, also importable
├── analysis/a3_tsp_sweep.py              136 lines  — drives the 12-layer sweep
└── vendored/fastkv/
    ├── utils.py                          137 lines  — exact upstream copy (read-only ref)
    └── llama_model.py                    400 lines  — exact upstream copy (read-only ref)
```

### Patch architecture (`src/qwen25omni_fastkv.py`)

Three runtime monkey-patches installed by `apply_fastkv()`:

1. **Per-layer `Qwen2_5OmniDecoderLayer.forward`** — runs the original forward, then if this is the TSP layer and `q_len > tsp_length`, re-projects pre-attention hidden states to Q/K, applies rotary, computes scores via FastKV's recipe (last-`window_size` queries → softmax → sum → `avg_pool1d(kernel_size=7)` → top-K). Stashes `self.new_tsp_indices` for the model loop to consume.

2. **Thinker LM forward** (`_patched_thinker_lm_forward`) — rewritten layer loop. After each layer, checks `decoder_layer.new_tsp_indices`; if set, gathers hidden_states (`dim=1`), position_ids (TMRoPE-aware: `dim=2` with broadcast across the 3 rope-axis rows), recomputes `position_embeddings` via `self.rotary_emb`, and slices the 4D causal mask along both query and key axes.

3. **Per-layer `kv_cluster`** config object holding FastKV hyperparameters.

**TMRoPE-correct slicing:**
```python
def _slice_position_ids(position_ids, tsp_idx):
    if position_ids.dim() == 3:                           # TMRoPE: (3, B, T)
        idx = tsp_idx.unsqueeze(0).expand(3, -1, -1)
        return torch.gather(position_ids, dim=2, index=idx)
    elif position_ids.dim() == 2:                         # plain RoPE: (B, T)
        return torch.gather(position_ids, dim=1, index=tsp_idx)
```

**Two no-op gates** for the smoke test:
- `tsp_length` larger than the prompt → `select_tsp_indices` returns `None`.
- `tsp_idx >= num_layers` → no layer fires TSP.

**Fully reversible** via `disable_fastkv()` — restores all original `forward` methods.

### KL harness (`analysis/kl_vs_baseline.py`)

- `run_one_config(model, processor, prompts, tsp_idx, tsp_length, …, baseline_cache)` — runs baseline + patched forwards; per-prompt baseline logits cached on CPU so a sweep doesn't re-run baseline 12 times.
- When patched run produces a shorter sequence (post-TSP), KL is computed only on the trailing-window region (the only tokens TSP guarantees are retained).
- Three default text prompts (short QA, medium summary, long passage). `--prompts_json` flag for custom.
- CLI: `--tsp_idx`, `--tsp_length`, `--window_size`, `--kernel_size`, `--pooling`, `--prompts_json`, `--out_jsonl`.

### Sweep (`analysis/a3_tsp_sweep.py`)

- Default sweep: `{0, 3, 5, 7, 10, 12, 15, 18, 20, 21, 25, 27}`. Override with `--tsp_idxs`.
- Loads model once; reuses baseline cache across all sweep points.
- Outputs:
  - `outputs/a3_tsp_sweep/sweep.jsonl` — one record per (tsp_idx, prompt)
  - `outputs/a3_tsp_sweep/sweep.csv`
  - `outputs/a3_tsp_sweep/sweep.png` — KL vs tsp_idx, one line per prompt, log-y

### Run order on the server

```bash
# 1. Smoke test — patch must be a no-op when TSP is disabled
python fastkv_omni/src/qwen25omni_fastkv.py --smoke \
       --model_path /data/armaan/models/Qwen2.5-Omni-7B
# expected: "PASSED — patch is logits-equivalent in no-op mode"

# 2. Single-config KL (cheap sanity check)
python fastkv_omni/analysis/kl_vs_baseline.py --tsp_idx 27 --tsp_length 2048
# expected: KL ≈ 0 at tsp_idx=27

# 3. Full A3 sweep
python fastkv_omni/analysis/a3_tsp_sweep.py --tsp_length 2048
```

### Honest caveats logged

1. Cannot run locally on Windows (no Python in PATH). Smoke test is the gate before trusting any sweep numbers.
2. `model.thinker.lm_head` path assumption — falls back to `F.linear` against tied `embed_tokens.weight` if not present.
3. Trailing-window-only KL when sequences differ — reasonable proxy, understates KL on the body.
4. Inner-LM forward rewrite assumes the standard HF text-decoder layout.

---

## Turn 7 — Sync to server

**User:**
> give command to copy to server armaan@10.244.120.178

**Assistant — three options:**

**Option 1 — Use the existing wrapper (full project sync):**
```powershell
cd C:\Users\Armaan\Desktop\PURS
.\scripts\sync-to-data-armaan.ps1
```

**Option 2a — `fastkv_omni/` only via Git Bash / WSL (rsync, deltas):**
```bash
rsync -avz --delete \
  /c/Users/Armaan/Desktop/PURS/fastkv_omni/ \
  armaan@10.244.120.178:/data/armaan/purs/fastkv_omni/
```

**Option 2b — `fastkv_omni/` only via PowerShell (scp):**
```powershell
scp -r C:\Users\Armaan\Desktop\PURS\fastkv_omni armaan@10.244.120.178:/data/armaan/purs/
```

After it lands, on the server:
```bash
ssh armaan@10.244.120.178
cd /data/armaan/purs
python fastkv_omni/src/qwen25omni_fastkv.py --smoke
```

---

## Memory updates this session

- `feedback_never_edit_paper_source_trees.md` — vendor copies into project folder; treat `*-main/` dirs as read-only.

## Key references

- FastKV paper: arXiv 2502.01068, ACL Findings 2026
- Vendored algorithm source: `fastkv_omni/vendored/fastkv/utils.py`
- Vendored patch pattern reference: `fastkv_omni/vendored/fastkv/llama_model.py`
- TMRoPE construction reference: `scripts/viz_tmrope.py`
- Existing Thinker attention hooks: `scripts/viz_attention_depth_curve.py`, `scripts/viz_layer_depth_experiment.py`, `scripts/viz_early_layer_relevance.py`
- Mask-capture monkey-patch template: `scripts/viz_attention_omnizip.py`
- Benchmark/Pareto runner (for A4 later): `scripts/run_qwen_omni_benchmark_matrix.py`
