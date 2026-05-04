# Visualization Scripts Guide
## Qwen2.5-Omni Attention Analysis

---

## Architecture Primer (Read First)

```
Video/Audio file
    │
    ├─ Audio Encoder (32 Whisper-style layers)
    │       input:  raw mel-spectrogram features
    │       output: (T_audio, D_audio)  — e.g. ~1500 tokens × 1280 dims
    │       ↓
    │   audio_projection (linear) → (T_audio, D_thinker=3584)
    │   ← attn_logits captured here (last encoder layer self-attn)
    │
    └─ Vision Encoder (32 ViT-style blocks)
            input:  sampled video frames as patches
            output: (T_video_patches, D_vision)
            ↓
        visual MLP projector → (T_video, D_thinker=3584)

Both get written into input_ids / inputs_embeds at their respective
audio_token_id and video_token_id positions.

━━━━━━━━━━━ OmniZip compression block (optional) ━━━━━━━━━━━
    operates on inputs_embeds (B, L, 3584) — LLM embed space
    outputs: inputs_embeds[:, global_mask, :]  — compressed sequence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thinker LLM (28 decoder layers, D=3584)
    input_ids: [<sys_text> | <audio_tokens × T_audio> | <video_tokens × T_video> | <question_text>]
    all tokens share the same D=3584 space ← THIS is what all viz scripts see
```

**Key facts:**
- **No pruning happens in the encoders.** Both encoders run in full. The audio encoder's last-layer attention is only *read* as an importance signal — it is never modified.
- **Everything OmniZip does happens in LLM embed space** (post-projection, pre-Thinker forward pass).
- Audio and video tokens have **identical dimensions** (D=3584) by the time any viz script touches them.

---

## Does Any Pruning Happen in the Encoder?

**No.** The sequence of events is:

1. Audio Encoder runs all 32 layers on all T_audio tokens → produces `audio_features (T_audio, 3584)` + `attn_logits [H, T_audio, T_audio]` from the last layer
2. Vision Encoder runs all 32 blocks on all patches → produces `video_features (T_video, 3584)`
3. Both are inserted into `inputs_embeds` at their respective token positions
4. **Only then** does `omnizip()` drop tokens from `inputs_embeds`

The encoders are black boxes from OmniZip's perspective — they produce full outputs and OmniZip selects from those outputs.

---

## Can We Rank Text↔Audio/Video Relevance?

Yes — two natural places, both unexplored by current scripts:

### A. In embed space (pre-Thinker, zero cost)
```python
# After inputs_embeds is assembled but before Thinker forward pass
text_pos   = (input_ids[0] != audio_id) & (input_ids[0] != video_id)
audio_pos  = input_ids[0] == audio_id
video_pos  = input_ids[0] == video_id

text_e  = inputs_embeds[0, text_pos]    # (T_text,  3584)
audio_e = inputs_embeds[0, audio_pos]   # (T_audio, 3584)
video_e = inputs_embeds[0, video_pos]   # (T_video, 3584)

# Per-audio-token: max cosine similarity to any text token
a = F.normalize(audio_e, dim=-1)
t = F.normalize(text_e,  dim=-1)
text_audio_relevance = (a @ t.T).max(dim=1).values   # (T_audio,)

# Per-video-token: same
v = F.normalize(video_e, dim=-1)
text_video_relevance = (v @ t.T).max(dim=1).values   # (T_video,)
```
This is pure cosine similarity in the shared embedding space — no forward pass needed. It answers "which audio/video tokens are geometrically close to the question tokens before the LLM even runs."

### B. Early Thinker layers (layers 0–4, cheap forward hook)
```python
# Hook layer 0 self_attn, extract: how much do text queries attend to audio/video keys?
def hook(module, args, kwargs, output):
    attn = output[1]  # (1, heads, seq, seq) — eager attn only
    # text query rows, audio/video key columns
    text_to_audio = attn[0, :, text_pos, :][:, :, audio_pos].mean()
    text_to_video = attn[0, :, text_pos, :][:, :, video_pos].mean()
```
This captures what the LLM "looks at" when processing text tokens in early layers — before any deep reasoning — giving a relevance signal grounded in the model's actual attention geometry.

These signals could be used to **replace or augment** the current audio importance score (which only uses intra-audio attention) with a direct question-relevance score.

---

## Script Reference

---

### `viz_attention_qwen.py`
**What:** Baseline Qwen modality fraction analysis — no compression, shows raw attention distribution.

**Taps into:** `model.thinker` forward pass with `output_attentions=True`. Reads `output.attentions[li]` at layers 0, 13, 27 — shape `(1, heads, seq, seq)`.

**Information source:** Last prompt token's attention row (`seq_len - 1`), averaged over heads. Attention mass is bucketed by token type using `audio_token_id` / `video_token_id` positions in `input_ids`.

**Outputs → `attention_viz_qwen/`:**
| File | What it shows |
|------|---------------|
| `1_modality_fractions.png` | Bar chart: % attention to audio vs video vs text at layers 0, 13, 27 |
| `2_audio_token_attention.png` | Line plot: attention weight per audio token at layer 27 (temporal profile) |
| `3_video_frame_attention.png` | Bar chart: summed attention per video frame at layer 27, red border = most attended |
| `4_modality_across_layers.png` | Line plot: mean per-token attention to each modality across all 28 layers |

**Example finding:** If layer 0 shows ~70% attention to audio but layer 27 shows ~45%, the model redistributes attention deeper — audio dominates early (raw signal), text/video gain later (reasoning).

**Run:**
```bash
python viz_attention_qwen.py
# Hardcoded: videos/worldsense/attribute_reasoning/video.mp4
# Output: attention_viz_qwen/
```

---

### `viz_attention_omnizip.py`
**What:** Same plots as above but on the OmniZip-compressed sequence. Shows how attention redistributes after 40% audio + 70% video tokens are dropped.

**Taps into:** `omnizip.modeling_qwen2_5_omni` (OmniZip's custom model class). Monkey-patches `omnizip_units_module.omnizip` to intercept `global_mask` — a `(orig_seq_len,)` bool tensor.

**Key mechanism:** After capturing `global_mask`, builds `orig_to_comp[i]` = compressed index for original position `i` (or -1 if dropped). All attention lookups use compressed indices.

**Outputs → `attention_viz_omnizip/`:**
| File | What it shows |
|------|---------------|
| `1_modality_fractions.png` | Same bar chart but on compressed sequence |
| `2_audio_token_attention.png` | Audio attention with grey vertical lines marking dropped tokens |
| `3_video_frame_attention.png` | Frame attention (top) + dropped token count per frame (bottom panel) |
| `4_modality_across_layers.png` | Layer-by-layer modality attention (kept tokens only) |
| `5_token_retention.png` | Bar chart: original vs kept count for audio and video |

**Example finding:** With `rho_audio=0.4, rho_video=0.7`: if original sequence was 2800 tokens → compressed to ~900. The `2_audio_token_attention.png` shows which temporal regions of audio were pruned (grey lines clustered = silence or low-relevance audio stretches).

**Config:**
```python
OMNIZIP_CONFIG = {
    "rho_audio": 0.4,    # fraction of audio tokens DROPPED
    "rho_video": 0.7,    # fraction of video tokens DROPPED
    "g": 3,              # merge group size
    "contextual_ratio": 0.05,  # fraction of dropped kept as context
}
```

**Run:**
```bash
python viz_attention_omnizip.py
# Output: attention_viz_omnizip/
```

---

### `viz_attention_heatmap.py`
**What:** Full T×T attention heatmap at 2 decoder layers (default: layers 4 and 20). Reproduces paper Figure 2 style. **Never run yet** — no output directory exists.

**Taps into:** Full `(seq, seq)` attention matrix from `output.attentions[li]`, averaged over heads. Shows the entire causal attention pattern log-scaled.

**Outputs → `attention_viz_heatmap/`:**
| File | What it shows |
|------|---------------|
| `fig2_attention_heatmap.png` | T×T heatmap per layer with token-type color strip (red=audio, blue=video, green=text) and zoomed inset on a random audio block |

**What to look for:** The "audio dominance" pattern from the paper — strong vertical bands at audio token columns, meaning most query positions attend heavily to audio keys. The inset zooms into a contiguous audio block.

**Run:**
```bash
python viz_attention_heatmap.py
# Output: attention_viz_heatmap/
```

---

### `viz_attention_heatmap_qwen.py`
**What:** Upgraded heatmap with a time-window-aware inset. Instead of a random audio block, the inset picks the temporal window with the **strongest mutual audio↔video attention**. **Never run yet.**

**Taps into:** `output.attentions[li]` + `model.thinker.get_rope_index(...)` to get RoPE temporal position IDs per token. Discretizes into time chunks:
```
window_id = (temporal_pos - ref) // (position_id_per_seconds × seconds_per_chunk)
```

**Inset selection logic:** For each time window containing both audio and video tokens, score = `0.5 × (mean(audio→video attention) + mean(video→audio attention))`. Highest-scoring window becomes the inset.

**Outputs → `attention_viz_qwen_heatmap/`:**
| File | What it shows |
|------|---------------|
| `attention_heatmap_qwen_layers.png` | Head-averaged heatmap per layer, red dashed box = strongest AV cross-attention window, inset = zoom of that window |

**Run:**
```bash
python viz_attention_heatmap_qwen.py \
    --layers 4,20 \
    --video_path videos/worldsense/attribute_reasoning/video.mp4 \
    --out_dir attention_viz_qwen_heatmap
```

---

### `viz_attention_heatmap_omnizip.py`
**What:** OmniZip version of the above. Same time-window-aware heatmap but on the **compressed sequence**. Token positions must be remapped through `global_mask` before any indexing. **Never run yet.**

**Taps into:** Everything from `heatmap_qwen.py` plus the monkey-patch for `global_mask`. Critical difference: the attention matrix is now `(compressed_len, compressed_len)` — positions like `audio_pos[5]` in the original sequence may not exist after pruning, so `orig_to_comp` remapping is required throughout.

**RoPE in compressed space:** `temporal_pos_comp = temporal_pos_orig[global_mask.numpy()]` — time window IDs are computed on the surviving tokens only.

**Outputs → `attention_viz_omnizip_heatmap/`:**
| File | What it shows |
|------|---------------|
| `attention_heatmap_omnizip_layers.png` | Same as qwen heatmap but on compressed T×T — visual comparison with baseline shows how audio bands thin out after pruning |

**Run:**
```bash
python viz_attention_heatmap_omnizip.py \
    --layers 4,20 \
    --rho_audio 0.4 --rho_video 0.7 \
    --out_dir attention_viz_omnizip_heatmap
```

---

## Comparison: Baseline vs OmniZip Scripts

| Dimension | Baseline (`_qwen`) | OmniZip (`_omnizip`) |
|-----------|-------------------|----------------------|
| Sequence seen by LLM | full `orig_seq_len` | `compressed_len` (~30–60% of original) |
| Audio token positions | direct from `input_ids` | remapped through `orig_to_comp` |
| Video token positions | direct from `input_ids` | remapped through `orig_to_comp` |
| Attention matrix shape | `(orig, orig)` | `(compressed, compressed)` |
| Extra data captured | — | `global_mask (orig_seq_len,)` bool |
| Model class | `transformers.Qwen2_5OmniForConditionalGeneration` | `omnizip.modeling_qwen2_5_omni.Qwen2_5OmniForConditionalGeneration` |

The most informative comparison is `2_audio_token_attention.png` side-by-side: baseline shows a smooth curve; OmniZip shows the same curve with vertical grey lines marking what was dropped — revealing which temporal regions OmniZip deems low-relevance.

---

## `viz_attention_depth_curve.py` and `viz_attention_encoders.py`

These two scripts produce the `vizzing/` benchmark sweep outputs (per question, per dataset). See `vizzing/` directory for output. Both support `--video_path`, `--question`, `--out_dir` CLI flags and are called by `scripts/run_all_attention_viz.py` for batch benchmark runs.

`viz_attention_depth_curve.py` uses **forward hooks** (not `output_attentions=True`) to stream compact statistics across all 28 layers without holding the full `(28, 1, heads, seq, seq)` tensor in memory — critical for long videos.

`viz_attention_encoders.py` hooks the **Q and K projections** inside the audio and vision encoders to manually reconstruct attention weights — the encoders don't expose weights through their API.
