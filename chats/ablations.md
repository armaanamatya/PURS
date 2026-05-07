# Session: Ablations — Token Compression for Qwen2.5-Omni

**Date:** 2026-05-03 / 2026-05-04  
**Branch:** `mds`

---

## Part 1: Ablation Space (ultrathink)

**User prompt:** What ablations can we do with these papers/code — OmniZip, OmniSift, ReDiPrune, DivPrune, MixKV, etc?

### Methods in scope

| Method | What it does | Key params |
|---|---|---|
| **OmniZip** | Audio-guided visual token compression for Qwen2.5-Omni | rho_audio=0.3, rho_video=0.6, g=3, contextual_ratio=0.05 |
| **ReDiPrune** | Text-conditioned diversity+relevance visual token pruning | subset_ratio, alpha, tau, prune_mode |
| **DivPrune** | Pure diversity-based frame/token selection (CVPR 2025) | subset_ratio |
| **MixKV** | KV-cache compression: attention + key diversity + value norm | budget, select_method |
| **Baseline** | Full-KV uncompressed model | — |
| **GPTQ/AWQ** | Weight quantization (Int4) | — |

### Ablation taxonomy produced

#### 1. OmniZip internal sweeps
- `rho_video` sweep: 0.3, 0.45, 0.6, 0.75, 0.9
- `rho_audio` sweep: 0.1, 0.2, 0.3, 0.5, 0.7
- **Audio guidance OFF** (replace audio-guided video merging with uniform ratios) — most critical ablation
- `contextual_ratio` sweep: 0.0, 0.025, 0.05, 0.1, 0.2
- `g` sweep: 1, 2, 3, 5, 8
- `rho_video=1.0` sanity check

#### 2. ReDiPrune internal sweeps
- `alpha` sweep: 0.0 → 1.0 (α=0 = pure DivPrune; α=1 = max text relevance)
- `subset_ratio` sweep: 0.25, 0.33, 0.5, 0.67
- Frame-level vs token-level pruning at matched ratio
- Random text query (replace question with noise) — tests if task-relevance is real
- `tau` sweep: 0.05, 0.1, 0.2, 0.5

#### 3. MixKV internal sweeps
- `select_method`: snapkv → vnorm → headwisemixkv
- `budget` sweep: 64, 128, 256, 512

#### 4. Cross-method comparisons at matched compression
- OmniZip vs DivPrune → audio guidance vs pure diversity
- **OmniZip vs ReDiPrune** → audio query vs text query (sharpest cross-paper ablation)
- ReDiPrune (α=0) vs DivPrune → sanity check
- Input token pruning vs KV-cache compression at matched VRAM
- Quantization vs token compression at matched throughput

#### 5. Stacking
- OmniZip + MixKV (orthogonal axes)
- ReDiPrune + MixKV
- GPTQ + OmniZip
- GPTQ + OmniZip + MixKV (maximum compression floor)

#### 6. Modality ablations
- `--no_audio` across all methods
- Audio present as input but NOT guiding selection (OmniZip-specific)

#### 7. Free stratified analysis (from existing results.jsonl)
- VideoMME short / medium / long
- With subtitles vs without
- Question category stratification (audio-heavy vs visual-OCR vs counting)
- Temperature sensitivity (0.1 vs 0.9, 10 repeats each)

### Priority ranking

**Run first:**
1. Audio guidance OFF in OmniZip
2. Alpha sweep in ReDiPrune
3. OmniZip vs ReDiPrune at matched compression
4. OmniZip + MixKV stacked

**Run second:**
5. rho_video sweep in OmniZip
6. --no_audio across all methods
7. Frame vs token level in ReDiPrune
8. Input token pruning vs KV compression at matched VRAM

**Free (analyze existing results):**
9. Short/medium/long stratification
10. With/without subtitles
11. Temperature variance per method

---

## Part 2: ablations.md created

**File:** `C:\Users\Armaan\Desktop\PURS\ablations.md`

Full ablation table written from the above taxonomy, organized into 7 categories with priority ranking.

---

## Part 3: Ablation Scripts — ultrathink + ultraplan

**User prompt:** Produce separate scripts for each ablation. Same config settings. Do not edit any main official repo code. Make folder `ablationscripts/`. Use parallel sub-agents. Ground in papers' math. Make explanation doc.

### Key source math discovered

#### OmniZip (`OmniZip-main/omnizip/omnizip_units.py`)

**Audio compression:**
```
dominant_num = round((1 - rho_audio) × N_audio)
importance_i = sum(mean_heads(attn_logits))  → top-dominant_num kept
contextual_num = round(contextual_ratio × N_audio) uniformly sampled as anchors
Merge remaining → anchor via softmax(max_k cos_sim(a_j, v_k)) weighted avg
```

**Audio → video guidance (the key mechanism):**
```
audio_group_retention_i = fraction of audio tokens kept in group i
video_merging_ratio_i = 0.75 + (0.35 - 0.75) × audio_group_retention_i
                      = 0.75 - 0.40 × audio_group_retention_i

High audio retention → merging_ratio = 0.35 → keep 65% of video (important moment)
Low audio retention  → merging_ratio = 0.75 → keep 25% of video (unimportant moment)

Normalized so Σ video_merging_ratio_i = rho_video × n_groups
```

**Video selection per group (`omnizip_istm`):**
- Even frames: `dpcknn` — density-peak k-NN diversity
- Odd frames: keep tokens LEAST similar to previous frame (novelty)

**Audio guidance OFF implementation:**
Import `omnizip_audio_attn` and `omnizip_istm` from original module, monkey-patch `omnizip_units.omnizip` to replace the `audio_group_retention → adjusted_vs` step with `[rho_video] * n_groups`.

#### ReDiPrune
```
score_i = min_{j∈S} (1 - cos_sim(v_i, v_j)) + α × cos_sim(v_i, q_text)

α=0: pure DivPrune (diversity only)
α=1: equal weight diversity + relevance
tau: relevance pre-filter gate (tokens with rel < tau excluded from candidates)
text query: mean_pool(embed_tokens(question_text))
```

#### DivPrune
```
Greedy farthest-point sampling on cosine distance:
sel[0] = most isolated token
sel[t] = argmax_i(min_{j∈sel} d(v_i, v_j))  where d = 1 - cos_sim
```

#### MixKV
```
s_attn = mean_pool_5(mean_W(softmax(Q_w @ K^T / √d)))   [SnapKV]
s_vnorm = normalize(||V_i||_2)
s_sim = normalize(-cos_sim(K_i, mean(K)))               [diversity]

snapkv:        score = s_attn
vnorm:         score = s_attn + s_vnorm × (mean(s_attn)/mean(s_vnorm))
headwisemixkv: score_h = hs_h × s_sim + (1-hs_h) × (s_attn + s_vnorm)

capacity = budget - window_size
Final KV = [top-capacity, window_32]
```

### Standard config (all scripts)
```python
DEFAULT_FPS = 2.0
DEFAULT_MAX_PIXELS = 100352
DEFAULT_MAX_FRAMES_VIDEOMME = 768
DEFAULT_MAX_FRAMES_OTHER = 128
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MODEL = "/data/armaan/models/Qwen2.5-Omni-7B"
```

### Parallel agents launched (5 simultaneous)

| Agent | Scripts produced |
|---|---|
| OmniZip sweeps | `omnizip_rho_video.py`, `omnizip_rho_audio.py`, `omnizip_audio_off.py`, `omnizip_contextual.py`, `omnizip_g_sweep.py`, `omnizip_sanity.py` |
| ReDiPrune sweeps | `rediprune_alpha.py`, `rediprune_ratio.py`, `rediprune_mode.py`, `rediprune_random_query.py`, `rediprune_tau.py` |
| MixKV sweeps | `mixkv_method.py`, `mixkv_budget.py` |
| Stacking/cross-method | `stack_omnizip_mixkv.py`, `stack_rediprune_mixkv.py`, `stack_gptq_omnizip.py`, `noaudio_all_methods.py` |
| Docs & runner | `README.md`, `run_ablations.sh` |

### Final file manifest — `ablationscripts/` (19 files)

| Script | Ablation | Values / Conditions |
|---|---|---|
| `omnizip_rho_video.py` | Video compression ratio | 0.3, 0.45, 0.6, 0.75, 0.9 |
| `omnizip_rho_audio.py` | Audio compression ratio | 0.1, 0.2, 0.3, 0.5, 0.7 |
| `omnizip_audio_off.py` | Audio guidance disabled | uniform, noaudio |
| `omnizip_contextual.py` | Safety-net coverage | 0.0, 0.025, 0.05, 0.1, 0.2 |
| `omnizip_g_sweep.py` | Merge group size | 1, 2, 3, 5, 8 |
| `omnizip_sanity.py` | Near-full vs default | near_full, default |
| `rediprune_alpha.py` | Text relevance weight | 0.0, 0.25, 0.5, 0.75, 1.0 |
| `rediprune_ratio.py` | Keep ratio / Pareto curve | 0.25, 0.33, 0.5, 0.67, 0.75 |
| `rediprune_mode.py` | Frame vs token level | frame, token |
| `rediprune_random_query.py` | Real vs random text query | real_query, random_query, zero_alpha |
| `rediprune_tau.py` | Relevance pre-filter threshold | 0.0, 0.05, 0.1, 0.2, 0.5 |
| `mixkv_method.py` | Scoring signal | snapkv, vnorm, headwisemixkv |
| `mixkv_budget.py` | KV budget | 64, 128, 256, 512 |
| `stack_omnizip_mixkv.py` | OmniZip + MixKV stacked | 4 conditions |
| `stack_rediprune_mixkv.py` | ReDiPrune + MixKV stacked | 3 conditions |
| `stack_gptq_omnizip.py` | Weight quant + token prune | 4 conditions |
| `noaudio_all_methods.py` | All methods, no audio input | 5 methods |
| `README.md` | Full math + run instructions | — |
| `run_ablations.sh` | Grouped runner (HIGH/MED/LOW) | — |

### Script design conventions
- **Self-contained**: no shared base module, all utilities inlined
- **No upstream edits**: `OmniZip-main/`, `ReDiPrune-main/`, `divprune-main/`, `MixKV-main/` untouched
- **`--value` / `--condition`** flag for single-point runs (parallel GPU dispatch)
- **Output per value**: `results.jsonl`, `vram_log.jsonl`, `console.log`, `stderr.log`, `run_summary.json`
- **Sweep summary**: `sweep_summary.json` at ablation root
- **Timing**: prefill via `thinker_max_new_tokens=1` + CUDA events; e2e via full generate
- **VRAM**: delta per question logged to `vram_log.jsonl`

### Output directory structure
```
ablation_outputs/
  omnizip_rho_video/
    rho_0p30/results.jsonl, vram_log.jsonl, console.log, run_summary.json
    rho_0p45/...
    rho_0p60/...
    sweep_summary.json
  omnizip_audio_off/
    uniform/...
    noaudio/...
  rediprune_alpha/
    alpha_0p00/...   ← equivalent to DivPrune
    alpha_0p50/...
    sweep_summary.json
  ...
```

---

## Key insight from the session

The sharpest cross-paper ablation: **OmniZip uses audio as the guidance signal; ReDiPrune uses text (the question)**. Both implemented on the same model at the same compression ratio. Running `rediprune_alpha.py` (α=0 = DivPrune) vs `omnizip_audio_off.py` (uniform video, audio signal removed) vs standard OmniZip isolates whether audio or text or pure diversity is the most informative selection signal for video token compression.
