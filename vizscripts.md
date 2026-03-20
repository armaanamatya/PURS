# Visualization & Evaluation Scripts — Deep Systems Reference

> Complete technical documentation for all four custom scripts in `PURS/` that instrument
> the Qwen2.5-Omni + OmniZip pipeline. Covers architecture, data flow, algorithm internals,
> and every non-obvious design decision.

---

## Table of Contents

1. [End-to-End Pipeline Architecture](#1-end-to-end-pipeline-architecture)
2. [OmniZip Algorithm Internals](#2-omnizip-algorithm-internals)
3. [viz_attention_qwen.py — Baseline Attention Probe](#3-viz_attention_qwenpy)
4. [viz_attention_omnizip.py — Compressed Attention Probe](#4-viz_attention_omnizippy)
5. [eval_qwen_omni.py — Baseline Evaluation Harness](#5-eval_qwen_omnipy)
6. [eval_qwen_omni_zip.py — OmniZip Evaluation Harness](#6-eval_qwen_omni_zippy)
7. [Cross-Script Comparison Matrix](#7-cross-script-comparison-matrix)
8. [Shared Assumptions, Gotchas & Subtle Bugs](#8-shared-assumptions-gotchas--subtle-bugs)
9. [Data Flow Diagrams](#9-data-flow-diagrams)

---

## 1. End-to-End Pipeline Architecture

### 1.1 From .mp4 to Token Sequence

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          VIDEO FILE (.mp4)                                   │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  process_mm_info(conversation, use_audio_in_video=True)                      │
│  ├── process_audio_info()                                                    │
│  │   ├── av.open(path) → checks audio stream exists                         │
│  │   ├── librosa.load(path, sr=16000) → raw waveform (float32, 16kHz)       │
│  │   └── Returns: List[np.ndarray]  (one waveform per audio/video element)   │
│  └── process_vision_info()                                                   │
│      ├── smart_nframes(fps=X, total_frames, video_fps)                       │
│      │   └── nframes = round(total_frames / video_fps * fps)                 │
│      │       clamped to [min_frames=4, max_frames=768], aligned to FRAME_FACTOR=2│
│      ├── _read_video_decord(path, nframes) → uniform temporal sampling       │
│      │   └── idx = torch.linspace(0, total-1, nframes).round().long()        │
│      └── Returns: (images_list, videos_list)                                 │
│          videos[0].shape = (nframes, C, H, W)                                │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Qwen2_5OmniProcessor(text=..., audio=..., videos=...)                       │
│  ├── Tokenizer: text → input_ids with placeholder tokens                     │
│  │   audio_token_id inserted N times (N = audio encoder output length)       │
│  │   video_token_id inserted M times (M = nframes × tokens_per_frame)        │
│  ├── Audio encoder frontend: waveform → mel spectrogram → input_features     │
│  │   Shape: (1, n_mels, time_steps)                                          │
│  ├── Vision encoder frontend: frames → pixel_values_videos                   │
│  │   Shape: (nframes, C, H, W) with spatial patching                         │
│  └── Returns BatchEncoding with:                                             │
│      input_ids, attention_mask, input_features, feature_attention_mask,       │
│      pixel_values_videos, video_grid_thw, audio_feature_lengths,             │
│      video_second_per_grid                                                   │
└──────────────┬───────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Thinker Forward Pass (modeling_qwen2_5_omni.py, line 2460+)                 │
│                                                                              │
│  STEP 1: Embed text tokens                                                   │
│    inputs_embeds = self.get_input_embeddings()(input_ids)                     │
│    Shape: (B, seq_len, 3584)  [hidden_dim=3584 for 7B]                       │
│                                                                              │
│  STEP 2: Encode & scatter audio (line 2474-2488)                             │
│    audio_features, attn_logits, attn_key = self.get_audio_features(...)      │
│    ├── audio_tower forward → Whisper-style encoder                           │
│    │   Returns last_hidden_state + attention logits from final layer          │
│    ├── attn_logits: softmax(QK^T/√d) from audio encoder's last attention     │
│    │   Shape: (num_heads, audio_seq_len, audio_seq_len)                      │
│    │   THIS IS THE IMPORTANCE SIGNAL THAT DRIVES OMNIZIP                     │
│    └── masked_scatter: audio embeddings replace audio placeholder positions   │
│                                                                              │
│  STEP 3: Encode & scatter images (line 2490-2499)                            │
│    image_embeds = self.get_image_features(pixel_values, image_grid_thw)       │
│    └── masked_scatter: image embeddings replace image placeholder positions   │
│                                                                              │
│  STEP 4: Encode & scatter video (line 2501-2511)                             │
│    video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)│
│    └── masked_scatter: video embeddings replace video placeholder positions   │
│                                                                              │
│  STEP 5: Compute RoPE position_ids (line 2521-2545)                          │
│    get_rope_index() → TMRoPE (Time-aligned Multimodal RoPE)                  │
│    position_ids shape: (3, B, seq_len) — 3 axes for temporal/spatial/text     │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════       │
│  STEP 6: OmniZip compression (line 2547-2577) — ONLY IF:                     │
│    pixel_values_videos is not None AND                                        │
│    audio_features is not None AND                                             │
│    attn_logits is not None AND                                                │
│    self.omnizip_config is not None                                            │
│                                                                              │
│    inputs_embeds, global_mask = omnizip(                                      │
│        inputs_embeds, attn_logits, input_ids, ...)                            │
│                                                                              │
│    inputs_embeds = inputs_embeds[:, global_mask, :]    # (B, compressed, D)   │
│    attention_mask = attention_mask[:, global_mask]      # (B, compressed)      │
│    position_ids   = position_ids[:, :, global_mask]    # (3, B, compressed)   │
│  ═══════════════════════════════════════════════════════════════════════       │
│                                                                              │
│  STEP 7: LLM backbone (line 2579-2589)                                       │
│    outputs = self.model(inputs_embeds, attention_mask, position_ids, ...)     │
│    28 decoder layers, each with multi-head self-attention + MLP               │
│                                                                              │
│  STEP 8: LM head                                                             │
│    logits = self.lm_head(hidden_states)  → vocab distribution                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Token Layout in the Sequence

After Step 4, `inputs_embeds` contains a heterogeneous sequence:

```
Position:  0 ─────────────── sys_len ──── sys_len+N_audio ──── +N_video ──── seq_end
Content:   [SYSTEM TEXT]     [AUDIO×N]    [VIDEO×M]            [USER TEXT + special tokens]
Token IDs: regular IDs       audio_tok_id  video_tok_id        regular IDs
Embeds:    text embeddings   audio encoder video encoder        text embeddings
                             output        output
```

- `N_audio` depends on audio duration: ~50 tokens/second after the Whisper-style encoder
- `N_video = nframes × tokens_per_frame` where `tokens_per_frame` depends on spatial resolution
  (typically 196 = 14×14 patches for standard resolution)
- For a 24-second video at `fps=1.0`: ~24 frames × 196 tokens = ~4,704 video tokens + ~1,200 audio tokens

### 1.3 Key Model Config Fields

| Field | Source | Role |
|-------|--------|------|
| `model.thinker.config.audio_token_id` | `config.json` | Sentinel ID in `input_ids` marking audio placeholder positions |
| `model.thinker.config.video_token_id` | `config.json` | Sentinel ID marking video placeholder positions |
| `model.thinker.config.image_token_id` | `config.json` | Sentinel ID marking image placeholder positions |
| `model.thinker.nframes` | Set by user code | Number of sampled video frames — required by OmniZip for temporal grouping |
| `model.thinker.omnizip_config` | Set by user code | Dict with `{rho_audio, rho_video, g, contextual_ratio}` — triggers compression |
| `model.thinker.rope_deltas` | Computed internally | Cached RoPE position offsets for multi-turn generation |

---

## 2. OmniZip Algorithm Internals

Source: `OmniZip-main/omnizip/omnizip_units.py` (416 lines, 3 functions)

### 2.1 High-Level Flow

```
omnizip()
  │
  ├── 1. Extract audio & video embeddings from flat sequence by token_id
  │
  ├── 2. omnizip_audio_attn() → audio keep_mask + merge_plan
  │      Uses attn_logits from audio encoder as importance signal
  │
  ├── 3. Execute merge_plan: weighted-average pruned audio tokens into anchors
  │      Weights derived from audio-video cross-similarity
  │
  ├── 4. Compute per-group audio retention ratios
  │
  ├── 5. Map audio retention → video merging ratios (INVERSE relationship)
  │      High audio retention → low video compression (and vice versa)
  │
  ├── 6. omnizip_istm() → video keep_mask per temporal group
  │      Even frames: spatial diversity (dpcknn)
  │      Odd frames: temporal redundancy (cosine similarity to previous frame)
  │
  └── 7. Assemble global_mask over entire sequence
         Text tokens: always kept (mask=True)
         Audio tokens: keep_mask from step 2
         Video tokens: mask from step 6
```

### 2.2 `omnizip_audio_attn()` — Audio Token Compression (lines 4–70)

**Inputs:**
- `audio_feature`: `(N_audio, D)` — audio token embeddings from the sequence
- `video_feature`: `(N_video, D)` or `None` — video token embeddings
- `attn_logits`: `(num_heads, audio_seq, audio_seq)` — softmax attention weights from the audio encoder's last layer
- `merging_ratio`: float — fraction of audio tokens to **drop** (confusingly named; `1 - merging_ratio` = keep rate)
- `contextual_ratio`: float — fraction of total audio tokens reserved as contextual anchors
- `g`: int — max number of tokens merged into each anchor

**Algorithm:**

```
Step A: Dominant token selection (lines 24-29)
  dominant_num = round((1 - merging_ratio) × N_audio)
  topk(attn_logits, dominant_num)  → keep these positions

  NOTE: attn_logits is used as a 1D importance vector here.
  The actual attn_logits shape is (H, T, T) but it gets consumed
  as a 1D vector in the topk call — so somewhere upstream the
  multi-head attention matrix has been reduced to a per-token score.
  In practice, the calling code passes attn_logits already shaped
  for this purpose.

Step B: Contextual anchor selection (lines 36-41)
  From the REMAINING (non-dominant) tokens:
  contextual_num = round(contextual_ratio × N_audio)
  Uniformly sample anchors with step = remaining_count / contextual_num
  These anchors preserve local temporal context that topk might miss.

Step C: Merge planning (lines 47-68)
  For tokens that are neither dominant nor anchor:
  1. Compute cosine similarity: pool_tokens @ anchor_tokens^T  → assignment
  2. Each pool token assigned to its most similar anchor
  3. Cross-modal scoring:
     - If video exists: score = max(audio_token @ video_tokens^T)
       i.e., how much does this audio token correlate with ANY video token
     - If no video: score = similarity to assigned anchor
  4. Per anchor: keep top-g scored tokens as merge sources
  5. merge_plan = {anchor_idx: [list of token indices to merge into it]}
```

**Output:** `(keep_mask, merge_plan)` where `keep_mask[i]=True` for dominant + anchor tokens.

### 2.3 Token Merging Execution (lines 170–190 of `omnizip()`)

After `omnizip_audio_attn()` returns the merge plan, the main `omnizip()` function executes it:

```python
for anchor_rel_idx, merge_rel_list in merge_plan.items():
    # Compute cross-modal similarity weights
    scores = (a_norm[merge_rel] @ v_norm.T).max(dim=1).values
    w = softmax(scores)

    # Weighted merge: new_anchor = (anchor + Σ(w_i × token_i)) / (1 + Σw_i)
    anchor_vec = audio_feature[anchor_rel_idx]
    merged_vec = (audio_feature[merge_rel] * w.unsqueeze(-1)).sum(dim=0)
    new_anchor = (anchor_vec + merged_vec) / (1.0 + w.sum())

    # Write back to the flat embedding tensor IN-PLACE
    flat_embeds[anchor_global_idx] = new_anchor
```

**Key insight:** The merge doesn't just drop tokens — it folds their information into the anchor
via a weighted average, where weights come from audio-video cross-attention similarity.
This preserves more information than pure pruning.

### 2.4 Audio-Guided Video Compression (lines 192–289)

The core innovation: **audio retention guides video compression ratios**.

**Two code paths** based on `num_input_frames % 4`:

| Path | Condition | Grouping Strategy |
|------|-----------|-------------------|
| **Aligned** (line 192) | `num_input_frames % 4 == 0` | `group_count = nframes / 4`, each group = 4 frames |
| **General** (line 292) | Otherwise | `VIDEO_GROUP_SIZE = tokens_per_frame × 4`, `AUDIO_GROUP_SIZE = 50` |

**In both paths, the same adaptive ratio logic applies (lines 197–247 / 322–372):**

```
For each temporal group i:
  1. audio_retention[i] = fraction of audio tokens kept in group i

  2. Inverse mapping:
     video_ratio[i] = max_ratio + (min_ratio - max_ratio) × audio_retention[i]

     Where min_ratio=0.35, max_ratio=0.75

     HIGH audio retention (important audio) → LOW video_ratio (keep more video)
     LOW  audio retention (silence/noise)   → HIGH video_ratio (aggressive video pruning)

  3. Constrained normalization:
     Scale middle groups so that Σ(video_ratio) = rho_video × n_groups
     Fix the min and max groups, adjust others proportionally
     Clamp all to [0.35, 0.75]
```

**Rationale:** When the audio is informative (speech, important sounds), the model needs
corresponding visual context, so video is preserved. When audio is uninformative, video
tokens from that segment are less critical and can be pruned more aggressively.

### 2.5 `omnizip_istm()` — Intra-Spatial-Temporal Merging (lines 72–122)

Compresses video tokens within each temporal group using two complementary strategies:

**Even frames → Spatial diversity via dpcknn (lines 102–108):**
```
dpcknn(tokens, keep_rate, k=5):
  1. Compute pairwise cosine similarity matrix
  2. For each token: mean similarity to its k-nearest neighbors
  3. KEEP tokens with LOWEST mean kNN similarity (most diverse/unique)
  4. This is density-peak-based: tokens in sparse regions survive
```

**Odd frames → Temporal redundancy removal (lines 110–120):**
```
  1. Compute per-token cosine similarity between frame t and frame t-1
  2. KEEP tokens with LOWEST similarity to previous frame (most novel)
  3. This removes temporally redundant patches (static background, etc.)
```

**Design choice:** Even/odd alternation ensures every pair of consecutive frames gets one
spatial-diversity pass and one temporal-redundancy pass. The `merging_ratio` list has two
elements: `[ratio_for_first_2_frames, ratio_for_remaining_frames]`.

### 2.6 `global_mask` Assembly (lines 276–282 / 402–409)

```python
global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
# Start with all True (keep everything)

global_mask[video_indices] = video_mask   # Apply video pruning decisions
global_mask[audio_indices] = audio_mask   # Apply audio pruning decisions
# Text tokens remain True — never pruned
```

The `global_mask` has shape `(original_seq_len,)` and is the single artifact that connects
original and compressed token spaces. It's what the viz script's monkey-patch captures.

### 2.7 `rho` Semantics — Definitive Resolution

Tracing through the code:

| Layer | Parameter Name | Actual Meaning |
|-------|---------------|----------------|
| `omnizip()` signature | `merging_ratio_audio` | Passed directly to `omnizip_audio_attn(merging_ratio=...)` |
| `omnizip_audio_attn()` | `merging_ratio` | `dominant_num = round((1 - merging_ratio) × N)` → **merging_ratio = fraction to DROP** |
| `eval_qwen_omni_zip.py` CLI | `--rho_audio` | Docstring says "fraction to KEEP" |
| `viz_attention_omnizip.py` comment | `rho_audio` | Comment says "fraction to DROP" |
| `demo.py` defaults | `rho_audio=0.4` | At 0.4: `dominant_num = 0.6 × N` → **60% kept**, 40% dropped |

**Verdict:** `rho_audio=0.4` means **drop 40%, keep 60%**. The parameter IS the drop ratio.
The eval script docstring saying "fraction to KEEP" is **incorrect**. The viz script comment
saying "fraction to DROP" is **correct**.

For `rho_video`, the same parameter flows to `omnizip_istm(merging_ratio=...)` where
`keep_ratio = 1.0 - merging_ratio[ratio_id]` (line 96) — confirming: **rho = drop fraction**.

---

## 3. viz_attention_qwen.py

**File:** `PURS/viz_attention_qwen.py` (289 lines)
**Purpose:** Probe where Qwen2.5-Omni's decoder attention goes *without any compression* —
a clean baseline showing raw cross-modality attention distribution.

### 3.1 Inputs

| Source | Detail |
|--------|--------|
| `VIDEO_PATH` | Hardcoded: `videos/worldsense/attribute_reasoning/video.mp4` (24s video) |
| `MODEL_PATH` | `/workspace/model` (Qwen2.5-Omni-7B weights on remote GPU server) |
| `QUESTION` | `"What is the profession of the man with a beard wearing a suit in the video?"` |
| `VIZ_LAYERS` | `[0, 13, 27]` — first, middle, last of 28 decoder layers |

### 3.2 Execution Flow — Step by Step

**Step 1: CVE Bypass (line 29)**
```python
_qwen_mod.check_torch_load_is_safe = lambda: None
```
Silences CVE-2025-32434 safety check that prevents `torch.load()` on PyTorch < 2.6.
The patch targets the *module-level* reference in `modeling_qwen2_5_omni`, not the
`transformers.utils` original, because the modeling module does `from ... import check_torch_load_is_safe`
which creates its own binding.

**Step 2: Model Load (lines 47–54)**
```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0",
    attn_implementation="eager",  # MANDATORY for attention extraction
)
```
`eager` attention computes `softmax(QK^T/√d) @ V` in Python (not fused CUDA kernels),
so the `(heads, seq, seq)` attention weight matrix is available in memory. `flash_attention_2`
and `sdpa` fuse the operation and never materialize the full weight matrix.

**Cost:** ~3–5× slower and ~2× more VRAM than sdpa for the same sequence length.

**Step 3: Input Preparation (lines 57–91)**

The conversation is formatted as a chat template with system prompt + user message containing
video + text. `process_mm_info()` extracts:
- `audios`: list of 1 waveform array (from video's audio track, 16kHz)
- `videos`: list of 1 tensor `(nframes, C, H, W)` — frame count depends on video duration

`model.thinker.nframes = num_input_frames` (line 79) — harmless on baseline but needed for
consistency if the model code checks it.

**Step 4: Thinker Forward (lines 120–132)**
```python
output = model.thinker(
    **thinker_inputs,
    output_attentions=True,   # return per-layer attention weights
    use_audio_in_video=True,  # process audio from video
    use_cache=False,          # no KV cache (single forward, not autoregressive)
    return_dict=True,
)
```

`THINKER_KEYS` filter (lines 106–116) restricts the input dict to only keys the thinker's
`forward()` accepts, avoiding `TypeError` from unexpected kwargs like `return_audio`.

`output.attentions`: tuple of 28 tensors, each `(1, num_heads, seq_len, seq_len)` in fp16.
At seq_len=6000, each tensor is `1 × 28 × 6000 × 6000 × 2 bytes ≈ 1.9 GB`. Total for
28 layers: ~53 GB — which is why they're immediately moved to CPU as fp32 (line 131).

**Step 5: Modality Mask Construction (lines 135–143)**
```python
ids = input_ids[0].cpu()
audio_pos = (ids == audio_token_id).nonzero(as_tuple=True)[0]   # absolute position indices
video_pos = (ids == video_token_id).nonzero(as_tuple=True)[0]
text_mask = (ids != audio_token_id) & (ids != video_token_id)
text_pos  = text_mask.nonzero(as_tuple=True)[0]
```

These position tensors are used to slice into attention rows/columns by modality.

**Step 6: `modality_fractions()` (lines 145–152)**
```python
def modality_fractions(attn_layer, query_idx):
    row = attn_layer[0, :, query_idx, :].mean(0)   # average over all heads → (seq_len,)
    a_sum = row[audio_pos].sum().item()
    v_sum = row[video_pos].sum().item()
    t_sum = row[text_pos].sum().item()
    total = a_sum + v_sum + t_sum + 1e-9
    return a_sum/total, v_sum/total, t_sum/total
```

`query_idx = last_q = seq_len - 1` — the last token of the prompt. This is the position
where the model must integrate all prior context to produce the first generation token.
Head-averaging smooths out head specialization (some heads attend to audio, others to video).

### 3.3 Output Plots

All saved to `attention_viz_qwen/` at 150 DPI:

**Plot 1: `1_modality_fractions.png`** (lines 156–184)
- 3-panel bar chart (one per VIZ_LAYER: 0, 13, 27)
- Bars: Audio (red), Video (blue), Text (green) — percentage of total attention
- Reveals how modality importance shifts across network depth
- Early layers often attend broadly; later layers typically concentrate on task-relevant modality

**Plot 2: `2_audio_token_attention.png`** (lines 189–214)
- Area plot + line of attention weight per audio token at layer 27
- X-axis ≈ time (audio tokens are sequential temporal embeddings)
- Top-5 peaks annotated with token indices
- Reveals which moments in the audio track the model considers important for answering
- Dynamic figure width: `min(14, max(8, n_audio_tokens // 15))` — scales with audio length

**Plot 3: `3_video_frame_attention.png`** (lines 218–248)
- Per-frame bar chart at layer 27
- `frame_attn[f] = Σ attention_weight[video_tokens_in_frame_f]`
- Tokens per frame computed as `len(video_pos) // num_input_frames` (integer division)
- Most-attended frame highlighted with red border (2px)
- Color intensity scaled by attention magnitude (Blues colormap)

**Plot 4: `4_modality_across_layers.png`** (lines 252–282)
- Line plot: mean per-token attention weight for each modality across all 28 layers
- Audio (red circles), Video (blue squares), Text (green triangles)
- Y-axis: mean attention per token (not fraction) — comparable across modalities of different sizes
- Shows the "attention lifecycle": when each modality gets attended to during processing

---

## 4. viz_attention_omnizip.py

**File:** `PURS/viz_attention_omnizip.py` (403 lines)
**Purpose:** Same modality-attention analysis as Section 3, but after OmniZip compression.
The central technical challenge: attention tensors live in *compressed* index space while
token identities (audio/video/text) are known in *original* index space.

### 4.1 Additional Inputs

| Parameter | Default | Meaning (verified) |
|-----------|---------|---------|
| `rho_audio` | 0.4 | Drop 40% of audio tokens |
| `rho_video` | 0.7 | Drop 70% of video tokens |
| `g` | 3 | Max tokens merged per contextual anchor |
| `contextual_ratio` | 0.05 | 5% of audio tokens reserved as uniformly-spaced anchors |

### 4.2 The Monkey-Patch Interception (lines 57–66)

```python
import omnizip.omnizip_units as omnizip_units_module
_original_omnizip = omnizip_units_module.omnizip
_captured_mask    = [None]   # mutable container for closure capture

def _patched_omnizip(*args, **kwargs):
    result = _original_omnizip(*args, **kwargs)
    _captured_mask[0] = result[1].cpu()   # result = (embeds, global_mask)
    return result

omnizip_units_module.omnizip = _patched_omnizip
```

**Why this works:** The thinker's `forward()` imports `omnizip` at call time (line 2549:
`from omnizip.omnizip_units import omnizip`). However, Python caches modules in `sys.modules`,
so the `from ... import omnizip` on subsequent calls resolves to the already-imported module.
But crucially, the *first* import in the thinker does `from omnizip.omnizip_units import omnizip`,
which binds the function name at import time. The patch works because it replaces the function
on the *module object* before the thinker's first forward call, and the thinker's import
creates a new binding that picks up the patched version.

**Why `[None]` instead of a plain variable:** Python closures over mutable containers work
by reference; reassigning a bare `_captured_mask = result[1]` inside the closure would create
a local variable instead of modifying the outer scope.

**Cleanup (line 151):** `omnizip_units_module.omnizip = _original_omnizip` — restores the
original function after the forward pass to avoid side effects on subsequent calls.

### 4.3 Position Remapping: Original ↔ Compressed Space (lines 162–189)

This is the most intricate part of the script. The problem:

```
Original space (input_ids):    [T T T A A A A A V V V V V V T T T]
                                         ↓ OmniZip ↓
Compressed space (attention):  [T T T A A A V V V T T T]
                                       ↑ some audio/video tokens removed
```

Attention tensors have shape `(1, heads, compressed_len, compressed_len)` but we need to
know which compressed position corresponds to which original modality token.

**Solution: `orig_to_comp` mapping (lines 163–168)**
```python
orig_to_comp = torch.full((orig_seq_len,), -1, dtype=torch.long)
comp_idx = 0
for orig_idx in range(orig_seq_len):
    if global_mask[orig_idx].item():
        orig_to_comp[orig_idx] = comp_idx
        comp_idx += 1
```

This builds a dense map: `orig_to_comp[i] = j` means original position `i` maps to
compressed position `j`. Dropped tokens get `-1`.

**Derived index sets (lines 170–188):**
```
audio_orig  = original positions of ALL audio tokens       (from input_ids)
audio_kept  = original positions of SURVIVING audio tokens (filtered by global_mask)
audio_comp  = compressed positions of surviving audio      (via orig_to_comp lookup)
audio_drop  = original positions of PRUNED audio tokens    (for visualization)

Same for video_kept, video_comp, video_drop, text_comp.
```

`audio_comp` is what you index into `attentions[layer][0, :, last_q, :]` to get
attention weights for surviving audio tokens.

### 4.4 Output Plots

All saved to `attention_viz_omnizip/` at 150 DPI:

**Plot 1: `1_modality_fractions.png`** (lines 204–234)
Same as baseline Plot 1 but computed over compressed sequence. Title includes compression
stats: `orig=XXXX → compressed=YYYY tok`. Uses `audio_comp`, `video_comp`, `text_comp`
to index the compressed attention row.

**Plot 2: `2_audio_token_attention.png`** (lines 238–282)
Unlike baseline, this plot shows ALL original audio token positions on the x-axis:
- **Kept tokens** (red fill + dots): attention values looked up via `audio_comp`
- **Dropped tokens** (vertical grey lines): mark where OmniZip pruned

The construction is non-trivial (lines 243–248):
```python
audio_full_attn = np.full(n_audio, np.nan)  # NaN = dropped
for rank, (orig_idx, comp_idx) in enumerate(zip(audio_kept, audio_comp)):
    orig_audio_rank = (audio_orig == orig_idx).nonzero(...)[0].item()
    audio_full_attn[orig_audio_rank] = row[comp_idx].item()
```
This translates from global original index → rank within audio tokens → position in the plot.

**Plot 3: `3_video_frame_attention.png`** (lines 286–338)
**Two-panel figure** (3:1 height ratio):
- **Top panel:** Per-frame attention sum (kept tokens only), same colormap as baseline
- **Bottom panel:** Red bars showing number of dropped tokens per frame
- Frame-level token assignment: `frame_id = rank_in_video // tokens_per_frame`

This dual view reveals OmniZip's decisions: frames with high drop counts but high attention
suggest the compression successfully identified and kept the most informative tokens.

**Plot 4: `4_modality_across_layers.png`** (lines 342–373)
Same cross-layer analysis as baseline, but labels say "Audio (kept)", "Video (kept)" to
clarify these are surviving tokens only. Mean attention per kept token.

**Plot 5: `5_token_retention.png`** (lines 377–396)
**NEW — not in baseline.** Side-by-side bar chart:
- Audio original vs Audio kept (reds)
- Video original vs Video kept (blues)
- Title includes actual ρ values used
- Absolute token counts with numeric labels

### 4.5 Why `last_q = compressed_len - 1`

In the baseline, `last_q = seq_len - 1` because the last token is always a text token
(never pruned). After OmniZip, the sequence is shorter but text tokens are still preserved
— so `compressed_len - 1` still corresponds to the last text token of the prompt.

---

## 5. eval_qwen_omni.py

**File:** `PURS/eval_qwen_omni.py` (227 lines)
**Purpose:** Automated benchmark harness for baseline Qwen2.5-Omni-7B on a local video MCQ
dataset described by `metadata.json`. No compression.

### 5.1 CLI Interface

```bash
python eval_qwen_omni.py \
    --metadata  metadata.json           # JSON array of video+Q&A entries
    --videos    /workspace/videos       # root dir for video files
    --output    /workspace/results.jsonl # per-question predictions
    --log       /workspace/eval.log     # appended run log
    --category  lecture                 # optional: filter by dataset or task_type
```

### 5.2 Expected Data Format (`metadata.json`)

```json
[
  {
    "file": "videos/worldsense/attribute_reasoning/video.mp4",
    "dataset": "worldsense",
    "task_type": "attribute_reasoning",
    "duration_s": 24,
    "questions": [
      {
        "question": "What is the profession of the man with a beard?",
        "choices": ["A. Doctor", "B. Lawyer", "C. Teacher", "D. Engineer"],
        "answer": "B",
        "task_type": "attribute_reasoning"
      }
    ]
  }
]
```

- Entries without `"questions"` key are silently skipped with a count
- `"file"` paths can be absolute, relative, or Windows-style with backslashes
- `"answer"` must be a single letter A–D
- `"choices"` can be pre-prefixed (`"A. Doctor"`) or raw (`"Doctor"`) — the script detects
  and adds prefixes if missing (line 67: `choices[0].startswith("A")`)

### 5.3 Video Resolution Strategy

```python
{"type": "video", "video": video_path, "fps": 1.0, "max_pixels": 360*420}
```

- **`fps=1.0`:** Sample 1 frame per second. For a 24s video → 24 frames → ~4,704 video tokens.
  Conservative to keep VRAM manageable without compression.
- **`max_pixels=151,200`:** Each frame resized so total pixels ≤ 151,200 (roughly 360×420).
  This caps spatial resolution to reduce per-frame token count.
- Together: ~24 × ~100-196 tokens per frame ≈ 2,400–4,700 video tokens per video

### 5.4 `resolve_video_path()` — Cross-Platform Path Resolution (lines 102–124)

Three-stage fallback:
1. Try the stored path verbatim (works if metadata was generated on the same machine)
2. Extract filename, try flat lookup in `videos_dir`
3. Recursive glob by stem + common extensions: `mp4`, `mkv`, `webm`, `avi`

Normalizes Windows backslashes to forward slashes before extraction.

### 5.5 Inference Pipeline (`run_inference`, lines 66–98)

**Prompt construction:**
```
{question}

A. choice1
B. choice2
C. choice3
D. choice4

Think step by step, then end your response with 'Answer: X' where X is A, B, C, or D.
```

**Model call:**
```python
output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, max_new_tokens=512)
```
- `return_audio=False` — skip the Talker (speech synthesis) head entirely
- `max_new_tokens=512` — generous but bounded; MCQ answers rarely exceed 200 tokens
- Uses `sdpa` attention (set at model load) for speed

**Output parsing (lines 91–98):**
1. Strip echoed prompt: find last occurrence of `"assistant"` in decoded string
2. Primary: regex `Answer:\s*([A-D])` (case-insensitive)
3. Fallback: last character in `ABCD` found anywhere in the response

### 5.6 `Tee` Logger (lines 23–47)

```python
class Tee:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")  # append mode
```

Replaces `sys.stdout` to dual-write to terminal + persistent file. Writes a timestamped
separator (`======...`) at the start of each run for easy log parsing. Implements `isatty()`
for compatibility with libraries that check terminal capabilities.

### 5.7 Output Format

**`results.jsonl`** — one JSON object per question:
```json
{
  "dataset": "worldsense",
  "task_type": "attribute_reasoning",
  "duration_s": 24,
  "question": "What is the profession of...",
  "choices": ["A. Doctor", "B. Lawyer", ...],
  "answer": "B",
  "prediction": "B",
  "correct": true,
  "reasoning": "The man appears to be in a courtroom setting..."
}
```

**Console/log output:**
```
  [✓] worldsense/attribute_reasoning [attribute_reasoning] pred=B ans=B
```

**Final summary:** Per-dataset accuracy breakdown with aligned formatting.

---

## 6. eval_qwen_omni_zip.py

**File:** `PURS/eval_qwen_omni_zip.py` (283 lines)
**Purpose:** Same benchmark as Section 5, but with OmniZip compression enabled.
Adds VRAM profiling, error logging, and the `"method": "omnizip"` tag for result merging.

### 6.1 CLI Interface (extends baseline)

```bash
python eval_qwen_omni_zip.py \
    --metadata         /workspace/metadata.json \
    --videos           /workspace/videos \
    --output           /workspace/results_zip.jsonl \
    --log              /workspace/eval_zip.log \
    --vram_log         /workspace/vram_log.jsonl \
    --rho_audio        0.4     # drop 40% of audio tokens
    --rho_video        0.7     # drop 70% of video tokens
    --g                3       # max merges per anchor
    --contextual_ratio 0.05    # 5% contextual anchors
```

### 6.2 OmniZip Activation — The Two-Line Integration

**Line 82–88: Config injection**
```python
omnizip_config = {
    "rho_audio": rho_audio, "rho_video": rho_video,
    "g": g, "contextual_ratio": contextual_ratio,
}
model.thinker.omnizip_config = omnizip_config
```

This is a *runtime attribute injection* — `omnizip_config` is not defined in the model's
`__init__`. The thinker's `forward()` checks `if self.omnizip_config is not None` (line 2550
in modeling), and if the attribute doesn't exist, Python raises `AttributeError`, caught by
a `hasattr()` check or the `if ... is not None` pattern (with a default of `None` set elsewhere).

**Line 118: Frame count injection (per-video)**
```python
model.thinker.nframes = videos[0].shape[0] if videos else 1
```

This MUST be set before every `model.generate()` call because different videos have different
frame counts, and OmniZip uses `nframes` to compute `video_token_per_frame` and temporal
grouping boundaries.

### 6.3 Model Class Difference

```python
# Baseline (eval_qwen_omni.py):
from transformers import Qwen2_5OmniForConditionalGeneration

# OmniZip (eval_qwen_omni_zip.py):
from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
```

The OmniZip version is a modified copy of the transformers class with:
1. Audio encoder modified to return `attn_logits` (attention weights from last layer)
2. OmniZip compression block inserted between embedding scatter and LLM backbone
3. Same weights — no fine-tuning or additional parameters

### 6.4 VRAM Profiling (lines 222–232)

```python
torch.cuda.reset_peak_memory_stats()          # Reset BEFORE inference
pred, reasoning = run_inference(...)
peak_vram = torch.cuda.max_memory_allocated() / 1024**3   # Peak during inference
curr_vram = torch.cuda.memory_allocated() / 1024**3       # After inference
```

Written to `vram_log.jsonl`:
```json
{"entry": "worldsense/attribute_reasoning", "task_type": "...",
 "duration_s": 24, "peak_vram_gb": 12.42, "after_vram_gb": 9.13}
```

**Analysis use cases:**
- Correlate `peak_vram_gb` with `duration_s` to quantify memory scaling
- Compare with baseline results to validate the claimed 1.4× memory reduction
- Detect VRAM spikes from unusually long/high-resolution videos

### 6.5 Enhanced Error Handling (lines 234–240)

```python
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    with open("/workspace/errors.log", "a") as ef:
        ef.write(f"\n--- {entry_label} ---\n{tb}\n")
    pred, reasoning = "ERROR", str(e)
```

Unlike baseline (which just prints the error), the OmniZip script:
1. Captures full traceback (not just the exception message)
2. Writes to a dedicated `errors.log` file (separate from the main log)
3. Continues execution — errored entries get `prediction="ERROR"` in results

### 6.6 Output JSONL Adds `"method": "omnizip"` (line 258)

```json
{"dataset": "worldsense", ..., "prediction": "B", "correct": true, "method": "omnizip"}
```

This allows `cat results.jsonl results_zip.jsonl | python analyze.py` for side-by-side
comparison without ambiguity about which model produced each result.

### 6.7 `resolve_video_path()` — Slightly Different Logic (lines 147–166)

Unlike baseline, this version tries to strip a `"videos/"` prefix from the stored path
before joining with `videos_dir`:
```python
for prefix in ("videos/", "videos\\"):
    if normalized.startswith(prefix):
        rel = normalized[len(prefix):]
```

This handles metadata generated with relative paths like `"videos/worldsense/video.mp4"`
when `--videos=/workspace/videos` is passed — avoiding double-nesting.

---

## 7. Cross-Script Comparison Matrix

| Dimension | viz_attention_qwen | viz_attention_omnizip | eval_qwen_omni | eval_qwen_omni_zip |
|-----------|--------------------|-----------------------|----------------|--------------------|
| **Model class** | `transformers` | `omnizip` | `transformers` | `omnizip` |
| **attn_impl** | `eager` | `eager` | `sdpa` | `sdpa` |
| **dtype** | fp16 | fp16 | bfloat16 | bfloat16 |
| **device_map** | `cuda:0` (fixed) | `cuda:0` (fixed) | `auto` | `auto` |
| **OmniZip active** | No | Yes | No | Yes |
| **output_attentions** | Yes | Yes | No | No |
| **use_cache** | False (explicit) | False (explicit) | True (default) | True (default) |
| **return_audio** | N/A (thinker only) | N/A (thinker only) | False | False |
| **Video fps** | raw (from process_mm_info) | raw | 1.0 | 1.0 |
| **max_pixels** | not set | not set | 360×420 | 360×420 |
| **VRAM tracking** | No | No | No | Yes |
| **Error handling** | None (crash) | None (crash) | Print + continue | Traceback log + continue |
| **Output format** | 4 PNG plots | 5 PNG plots | JSONL + log | JSONL + VRAM JSONL + error log |
| **Batch size** | 1 video | 1 video | N videos (sequential) | N videos (sequential) |
| **CVE bypass** | Yes | Yes | No | No |
| **sys.path setup** | `OmniZip-main/` + `qwen-omni-utils/src/` | Same | No modification | `os.path.dirname(__file__)` |

---

## 8. Shared Assumptions, Gotchas & Subtle Bugs

### 8.1 `use_audio_in_video=True` Is Load-Bearing

All four scripts pass this to both `process_mm_info()` AND `processor()` AND `model.generate()`/`model.thinker()`.

Under the hood:
- `process_audio_info()` checks `_check_if_video_has_audio(path)` using `av.open()` to probe for audio streams
- If True: extracts audio via `librosa.load(path, sr=16000)` through FFmpeg backend
- The flag must be consistent at every stage — if `process_mm_info()` extracts audio but
  the model call doesn't set the flag, the audio features will be misaligned with positions

### 8.2 `eager` vs `sdpa` vs `flash_attention_2`

| Implementation | Returns weights? | Relative speed | Memory |
|----------------|-----------------|----------------|--------|
| `eager` | Yes — full `(B, H, S, S)` | 1× (baseline) | High (materializes full matrix) |
| `sdpa` | No (fused kernel) | ~2–3× faster | Lower |
| `flash_attention_2` | No (IO-aware tiling) | ~3–5× faster | Lowest |

The viz scripts MUST use `eager`. The eval scripts use `sdpa` because they don't need weights.
Both the Flash and SDPA attention classes in the OmniZip modeling file contain the `return_logits`
code path (lines 691–706 and 752–767) that manually recomputes `softmax(QK^T/√d)` for the audio
encoder — this is separate from the decoder attention and always works regardless of `attn_implementation`.

### 8.3 `thinker` vs `model.generate()` — Architectural Note

```
Qwen2_5OmniForConditionalGeneration
  ├── thinker (Qwen2_5OmniThinkerForConditionalGeneration)
  │   ├── model (Qwen2_5OmniModel — the 28-layer transformer)
  │   ├── audio_tower (Qwen2_5OmniAudioEncoder — Whisper-style)
  │   ├── visual (Qwen2VisionTransformerPretrainedModel — ViT)
  │   └── lm_head (Linear → vocab logits)
  └── talker (speech synthesis head — skipped with return_audio=False)
```

- Viz scripts call `model.thinker(...)` directly because `model.generate()` doesn't support
  `output_attentions=True` in a clean way for this architecture
- Eval scripts call `model.generate(...)` which internally routes through `thinker.forward()`
  for prefill, then autoregressive decoding via `thinker.prepare_inputs_for_generation()`

### 8.4 OmniZip sys.path Discrepancy

| Script | sys.path Strategy |
|--------|-------------------|
| `viz_attention_qwen.py` | `sys.path.insert(0, "OmniZip-main/")` + `sys.path.insert(0, "OmniZip-main/qwen-omni-utils/src/")` |
| `viz_attention_omnizip.py` | Same as above |
| `eval_qwen_omni.py` | No path modification (uses transformers stock class) |
| `eval_qwen_omni_zip.py` | `sys.path.insert(0, os.path.dirname(__file__))` — assumes `omnizip/` is sibling |

The eval_qwen_omni_zip.py path strategy is more fragile. If the script is moved or symlinked,
`os.path.dirname(__file__)` might not contain the `omnizip/` package.

### 8.5 Answer Parsing — False Positive Risk

```python
# Primary: structured pattern
m = re.search(r"Answer:\s*([A-D])", decoded, re.IGNORECASE)

# Fallback: last ABCD character in response
letter = next((c for c in reversed(decoded.strip()) if c in "ABCD"), decoded.strip())
```

The fallback scans the entire response backwards. If the model writes something like
`"The answer is not B, it is actually C"`, the regex correctly extracts C, but if the model
writes `"Let me think about options A and D... Answer: B"` the regex correctly gets B.
The fallback only fires when the regex fails — risky because any trailing A/B/C/D in
reasoning text could be picked up.

### 8.6 `rho` Naming — Definitive Truth (Resolved in Section 2.7)

| Where | Says | Actually |
|-------|------|----------|
| `viz_attention_omnizip.py` line 47 comment | "fraction of audio tokens to DROP" | **CORRECT** |
| `eval_qwen_omni_zip.py` docstring line 14 | "fraction of audio tokens to KEEP" | **INCORRECT** |
| `omnizip_audio_attn()` line 25 | `dominant_num = round((1 - merging_ratio) × N)` | Confirms: merging_ratio = **drop fraction** |
| `omnizip_istm()` line 96 | `keep_ratio = 1.0 - merging_ratio[ratio_id]` | Confirms: merging_ratio = **drop fraction** |

`rho_audio=0.4` → keep 60%, drop 40%. `rho_video=0.7` → keep 30%, drop 70%.

### 8.7 Video Token Count — Integer Division Risk

```python
tpf = len(video_pos) // num_input_frames   # tokens per frame
```

If `len(video_pos)` is not perfectly divisible by `num_input_frames`, the last
`len(video_pos) % num_input_frames` tokens are silently excluded from frame-level analysis.
This matters when the processor applies different spatial resolutions to different frames
or when there's padding. The robust approach would use `video_grid_thw` to derive exact
frame boundaries:
```python
# video_grid_thw: (n_frame_groups, 3) where columns are (temporal, height, width)
# tokens_in_group = t × h × w for each row
```

### 8.8 The `attn_logits` Bridge Between Audio Encoder and OmniZip

The audio encoder's attention layers (both Flash and SDPA variants) have been modified to
optionally return attention logits when `return_logits=True`:

```python
# modeling_qwen2_5_omni.py, lines 691-706 (Flash variant)
if return_logits:
    with torch.no_grad():
        q = query_states.permute(1, 0, 2)   # [H, T, D]
        k = key_states.permute(1, 0, 2)
        attn_logits = torch.matmul(q, k.transpose(-1, -2))
        attn_logits = attn_logits / (q.shape[-1] ** 0.5)
        attn_logits = F.softmax(attn_logits, dim=-1)
        return_k = k
```

**This is a separate computation from the fused attention kernel** — even when using
Flash/SDPA for the actual attention output, the logits are recomputed manually in a
`no_grad()` block specifically for OmniZip's importance scoring. The `torch.cuda.empty_cache()`
on line 705 is there to free the temporary tensors from this recomputation.

The logits flow: `audio_tower.forward()` → `get_audio_features()` → stored as `attn_logits` →
passed to `omnizip()` → used in `omnizip_audio_attn()` for `torch.topk()` importance selection.

---

## 9. Data Flow Diagrams

### 9.1 Visualization Scripts — Data Flow

```
                    viz_attention_qwen.py              viz_attention_omnizip.py
                    ═══════════════════                ════════════════════════

Video file ──────►  process_mm_info()                  process_mm_info()
                    │                                   │
                    ▼                                   ▼
                    Processor(text, audio, video)       Processor(text, audio, video)
                    │                                   │
                    ▼                                   ▼
                    thinker.forward(                    thinker.forward(
                      output_attentions=True)             output_attentions=True)
                    │                                   │
                    │                                   ├── omnizip() called internally
                    │                                   │   └── monkey-patch captures global_mask
                    │                                   │
                    ▼                                   ▼
                    attentions[28] × (1,H,S,S)         attentions[28] × (1,H,S',S')
                    S = original seq_len                S' = compressed seq_len
                    │                                   │
                    ▼                                   ▼
                    Modality masks from input_ids       orig_to_comp remapping
                    audio_pos, video_pos, text_pos      audio_comp, video_comp, text_comp
                    │                                   │
                    ▼                                   ▼
                    4 PNG plots                         5 PNG plots
                    └── attention_viz_qwen/             └── attention_viz_omnizip/
```

### 9.2 Evaluation Scripts — Data Flow

```
metadata.json ─────► Load & filter entries
                     │
                     ▼
                     For each entry:
                     │
                     ├── resolve_video_path()
                     │
                     ├── For each question:
                     │   │
                     │   ▼
                     │   Format MCQ prompt
                     │   │
                     │   ▼
                     │   process_mm_info(conversation)
                     │   │
                     │   ▼
                     │   Processor(text, audio, video)
                     │   │
                     │   ▼
                     │   ┌─── eval_qwen_omni ──────┐   ┌── eval_qwen_omni_zip ──────────┐
                     │   │ model.generate(          │   │ model.thinker.nframes = N       │
                     │   │   return_audio=False,    │   │ torch.cuda.reset_peak_memory()  │
                     │   │   max_new_tokens=512)    │   │ model.generate(                 │
                     │   │                          │   │   return_audio=False,            │
                     │   │ (stock transformers)     │   │   max_new_tokens=512)            │
                     │   └──────────┬───────────────┘   │ (omnizip model class)            │
                     │              │                    │ peak_vram = max_memory_alloc()   │
                     │              │                    └──────────┬──────────────────────┘
                     │              ▼                               ▼
                     │   Parse answer (regex → fallback)   Parse answer + log VRAM
                     │   │                                 │
                     │   ▼                                 ▼
                     │   results.jsonl                     results_zip.jsonl + vram_log.jsonl
                     │
                     ▼
                     Print accuracy summary (overall + per-dataset)
```

### 9.3 OmniZip Internal Token Flow

```
inputs_embeds (B, seq_len, D)
  │
  ├── Extract by token_id ──► audio_feature (N_aud, D)    video_feature (N_vid, D)
  │                                   │                           │
  │                                   ▼                           │
  │                           omnizip_audio_attn()                │
  │                           ├── topk(attn_logits)               │
  │                           │   → dominant tokens (kept)        │
  │                           ├── uniform sampling                │
  │                           │   → contextual anchors (kept)     │
  │                           └── similarity assignment           │
  │                               → merge_plan                   │
  │                                   │                           │
  │                                   ▼                           │
  │                           Execute merge_plan:                 │
  │                           anchor = (anchor + Σw·tok)/(1+Σw)  │
  │                           Write merged vectors back to        │
  │                           flat_embeds IN-PLACE                │
  │                                   │                           │
  │                                   ▼                           │
  │                           audio_mask (N_aud,) bool            │
  │                           + audio group retention ratios      │
  │                                   │                           │
  │                                   ▼                           ▼
  │                           Adaptive video ratios ────► omnizip_istm()
  │                           (inverse of audio retention)   ├── Even frames: dpcknn
  │                                                          │   (spatial diversity)
  │                                                          └── Odd frames: cosine sim
  │                                                              (temporal redundancy)
  │                                                                    │
  │                                                                    ▼
  │                                                          video_mask (N_vid,) bool
  │                                                                    │
  └── global_mask = ones(seq_len)                                      │
      global_mask[audio_indices] = audio_mask  ◄───────────────────────┘
      global_mask[video_indices] = video_mask

      Return: (inputs_embeds_with_merges, global_mask)

      Caller applies:
        inputs_embeds = inputs_embeds[:, global_mask, :]     # prune
        attention_mask = attention_mask[:, global_mask]        # sync
        position_ids   = position_ids[:, :, global_mask]      # sync (3-axis TMRoPE)
```
