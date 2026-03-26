# OmniZip — Complete Systems-Level Technical Breakdown

**Paper**: [arXiv:2511.14582](https://arxiv.org/abs/2511.14582) (Tao et al., Nov 2025)
**Repo**: [github.com/KD-TAO/OmniZip](https://github.com/KD-TAO/OmniZip)

---

## 1. High-Level Architecture

### 1.1 The Problem — Quadratic Bottleneck in Omnimodal LLMs

Qwen2.5-Omni processes audio, video, and text in a unified pipeline. The token counts explode:

| Modality | Typical count | Source |
|---|---|---|
| Video | 150,528 tokens | 768 frames × 196 patches/frame |
| Audio | ~500-2000 tokens | Whisper-style encoder, depends on duration |
| Text | ~50-100 tokens | User prompt |

Self-attention in the LLM decoder is O(n^2) in sequence length. With 150k+ tokens, this means:
- **KV-cache memory**: ~40GB+ for a 7B model at fp16
- **Prefill latency**: tens of seconds even on A100
- **Per-step decode cost**: proportional to total cached KV entries

**Prior art compresses only one modality**: FastV prunes vision tokens, AudioPruner prunes audio. None jointly compress audio+video while preserving cross-modal semantic alignment.

### 1.2 OmniZip's Solution

A **training-free**, **inference-time** token compression framework that:
1. Uses the audio encoder's own attention weights as a **saliency signal** (zero extra cost)
2. **Jointly compresses** both audio and video tokens
3. Uses audio information density to **dynamically allocate** per-window video compression budgets
4. Compresses video via an **interleaved spatio-temporal scheme** (alternating spatial diversity + temporal novelty)

**Result**: 3.42x inference speedup, 1.4x memory reduction, near-zero accuracy degradation.

### 1.3 Where It Sits in the Pipeline

```
                     ┌─────────────────────────────────────────────────────┐
                     │              Qwen2.5-Omni Architecture              │
                     └─────────────────────────────────────────────────────┘

Raw Video (.mp4)     Raw Audio (from video)         Text Prompt
      │                       │                          │
      ▼                       ▼                          ▼
┌──────────────┐   ┌────────────────────┐      ┌──────────────────┐
│ Vision Tower │   │ Audio Tower        │      │ Text Embedder    │
│ (ViT-based)  │   │ (Whisper-style)    │      │ (token → embed)  │
│              │   │                    │      │                  │
│ pixel_values │   │ mel features       │      │ input_ids        │
│ → patches    │   │ → Conv1d → Conv2d  │      │                  │
│ → ViT layers │   │ → Positional Emb   │      │                  │
│ → spatial    │   │ → N encoder layers │      │                  │
│   merge      │   │ → AvgPool → Proj   │      │                  │
│              │   │                    │      │                  │
│ Output:      │   │ Output:            │      │ Output:          │
│ video_embeds │   │ audio_features     │      │ text_embeds      │
│ [N_v, D]     │   │ [N_a, D]           │      │ [N_t, D]         │
│              │   │ + attn_logits ←NEW │      │                  │
│              │   │ + attn_key   ←NEW  │      │                  │
└──────┬───────┘   └────────┬───────────┘      └────────┬─────────┘
       │                    │                           │
       ▼                    ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Thinker Forward (Prefill Stage)                     │
│                                                                     │
│  1. inputs_embeds = text_embeds                                     │
│  2. masked_scatter(audio_token positions, audio_features)           │
│  3. masked_scatter(video_token positions, video_embeds)             │
│  4. Compute position_ids via get_rope_index() (M-RoPE)             │
│                                                                     │
│  ══════════════════════════════════════════════════════════════════  │
│  ║  5. *** OmniZip Injection *** (lines 2547-2584)               ║  │
│  ║     inputs_embeds, global_mask = omnizip(...)                  ║  │
│  ║     inputs_embeds = inputs_embeds[:, global_mask, :]           ║  │
│  ║     attention_mask = attention_mask[:, global_mask]             ║  │
│  ║     position_ids = position_ids[:, :, global_mask]             ║  │
│  ══════════════════════════════════════════════════════════════════  │
│                                                                     │
│  6. self.model(inputs_embeds, attention_mask, position_ids, ...)    │
│     → Qwen2 LLM decoder (autoregressive generation)                │
│                                                                     │
│  7. logits = self.lm_head(hidden_states)                            │
└─────────────────────────────────────────────────────────────────────┘
       │
       ▼
  Text output tokens  (talker/TTS disabled during eval)
```

**Critical detail**: OmniZip runs at **prefill time only**. Once the compressed sequence enters the LLM decoder, autoregressive decoding proceeds normally with a smaller KV cache. The compression is a one-shot operation — no per-layer or per-step overhead.

---

## 2. Module-by-Module Analysis

### 2.1 `omnizip/omnizip_units.py` — The Compression Engine (423 lines)

This single file contains ALL the compression logic. Three functions:

| Function | Lines | Purpose |
|---|---|---|
| `omnizip_audio_attn()` | 4-70 | Audio token selection + cross-modal merging |
| `omnizip_istm()` | 72-122 | Interleaved Spatio-Temporal video compression |
| `omnizip()` | 124-416 | Orchestrator with two code paths |

**Code path split in `omnizip()`**: There are TWO implementations inside this function:
- **Path A (lines 192-289)**: Triggered when `num_input_frames % 4 == 0`. Uses frame-count-derived groups.
- **Path B (lines 291-416)**: General fallback. Uses fixed `VIDEO_GROUP_SIZE = video_token_per_frame * 4` and `AUDIO_GROUP_SIZE = 50`.

Both paths share the same logic; Path A is slightly optimized for the common case where frame count divides evenly by 4.

---

### 2.2 `omnizip/modeling_qwen2_5_omni.py` — Patched Qwen2.5-Omni (~4600 lines)

This is a **full copy** of HuggingFace's `transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py` with precisely 5 categories of modifications:

#### Modification 1: Audio Attention Return Signature
**Files touched**: `Qwen2_5OmniAudioAttention`, `Qwen2_5OmniAudioFlashAttention2`, `Qwen2_5OmniAudioSdpaAttention`

All three attention implementations are modified to accept `return_logits: bool = False` and optionally compute + return attention logits:

```python
# ORIGINAL (vanilla transformers):
def forward(self, hidden_states, cu_seqlens) -> torch.Tensor:
    ...
    return attn_output

# PATCHED (OmniZip):
def forward(self, hidden_states, cu_seqlens, return_logits=False) -> Tuple:
    ...
    attn_output = <normal_attention>  # unchanged

    if return_logits:
        with torch.no_grad():                             # no gradient needed
            q = query_states.permute(1, 0, 2)             # [H, T, D]
            k = key_states.permute(1, 0, 2)               # [H, T, D]
            attn_logits = torch.matmul(q, k.transpose(-1, -2))  # [H, T, T]
            attn_logits = attn_logits / (q.shape[-1] ** 0.5)    # scale
            attn_logits = F.softmax(attn_logits, dim=-1)         # normalize
            return_k = k
    else:
        attn_logits = None
        return_k = None
    torch.cuda.empty_cache()
    return attn_output, attn_logits, return_k
```

**Why this design**: The actual attention computation (Flash Attention or SDPA) is left untouched. The logit computation is done *separately* in a `no_grad()` block. This means:
- Flash Attention still runs for speed/memory during actual forward pass
- The explicit QK^T matmul only happens on the LAST encoder layer (see below)
- `torch.cuda.empty_cache()` is called to reclaim the temporary logit memory

**Shape**: `attn_logits` is `[H, T, T]` where H = number of attention heads, T = audio sequence length within a chunk.

#### Modification 2: Encoder Layer Propagation
**Class**: `Qwen2_5OmniAudioEncoderLayer`

```python
def forward(self, hidden_states, cu_seqlens, return_logits=False):
    ...
    hidden_states, logits, k = self.self_attn(
        hidden_states=hidden_states,
        cu_seqlens=cu_seqlens,
        return_logits=return_logits,   # ← only True for last layer
    )
    ...
    return outputs, logits, k
```

#### Modification 3: Audio Encoder — Last-Layer-Only Extraction
**Class**: `Qwen2_5OmniAudioEncoder.forward()`

The encoder loop passes `return_logits=True` ONLY to the final layer:

```python
for idx, encoder_layer in enumerate(self.layers):
    if idx == len(self.layers) - 1:        # LAST layer only
        layer_outputs, logits, attn_key = encoder_layer(
            hidden_states, cu_seqlens, True   # return_logits=True
        )
    else:
        layer_outputs, _, _ = encoder_layer(
            hidden_states, cu_seqlens, False   # return_logits=False
        )
    hidden_states = layer_outputs[0]
```

**After the encoder loop — Dimensionality Reduction of `attn_logits`**:

This is a critical and subtle step. The raw attention logits from the last layer have shape `[H, T, T]` where T is the pre-pooling audio token count. But the audio features that reach the LLM have been downsampled (AvgPool1d with stride=2 + projection). So the logits must be downsampled to match:

```python
with torch.no_grad():
    if logits is not None:
        # --- ATTENTION SALIENCY SCORE ---
        # logits shape: [H, T, T]
        attn_mean = logits.mean(dim=0)    # average over heads → [T, T]
        attn_mean = attn_mean.sum(dim=0)  # column-sum = "how much is each token attended TO" → [T]

        # Downsample to match AvgPool1d(stride=2):
        T = attn_mean.shape[0]
        if T % 2 == 1:
            attn_mean = attn_mean[:-1]    # drop last if odd
        attn_mean = attn_mean.view(-1, 2).mean(dim=-1)  # pairwise average → [T/2]

        # --- KEY VECTORS (for future use, currently used as video proxy) ---
        H, Tk, D = attn_key.shape
        if Tk % 2 == 1:
            attn_key = attn_key[:, :-1, :]
        attn_key = attn_key.view(H, -1, 2, D).mean(dim=2)  # downsample keys → [H, Tk/2, D]
        attn_key = attn_key.mean(dim=0, keepdim=True)        # head-average → [1, Tk/2, D]

return BaseModelOutput(last_hidden_state=token_audio), attn_mean, attn_key
```

**The `attn_mean` vector** is the per-token saliency score passed to `omnizip_audio_attn()` as `attn_logits`. It measures **how much each audio token was collectively attended to** by all other tokens across all heads. This is the core signal that drives audio token selection.

**The `attn_key` tensor** contains the head-averaged, downsampled key vectors from the last encoder layer. In the current codebase it's returned but not directly used in `omnizip_units.py` — the `omnizip_audio_attn()` function uses the audio/video *features* instead. This appears to be reserved for potential future use or experimental branches.

#### Modification 4: `get_audio_features()` — Passthrough
```python
def get_audio_features(self, input_features, ...):
    ...
    audio_outputs, attn_logits, attn_key = self.audio_tower(...)
    audio_features = audio_outputs.last_hidden_state
    return audio_features, attn_logits, attn_key  # ← NEW: returns logits+key
```

#### Modification 5: Thinker `forward()` — OmniZip Injection + Post-compression
**The injection** (lines 2547-2584):

```python
# OmniZip Inference
if pixel_values_videos is not None and audio_features is not None and attn_logits is not None:
    from omnizip.omnizip_units import omnizip
    if self.omnizip_config is not None:
        omnizip_config = self.omnizip_config
    else:
        omnizip_config = {"rho_audio": 0.3, "rho_video": 0.6, "g": 3, "contextual_ratio": 0.05}

    inputs_embeds, global_mask = omnizip(
        inputs_embeds, attn_logits, input_ids,
        self.config.audio_token_id, self.config.video_token_id,
        num_input_frames=self.nframes,
        merging_ratio_audio=omnizip_config["rho_audio"],
        merging_ratio_v=omnizip_config["rho_video"],
        contextual_ratio=omnizip_config["contextual_ratio"],
        g=omnizip_config["g"],
    )

    # Apply the mask — this is where the actual token reduction happens:
    inputs_embeds  = inputs_embeds[:, global_mask, :]        # [B, L', D]  (L' < L)
    attention_mask = attention_mask[:, global_mask]           # [B, L']
    position_ids   = position_ids[:, :, global_mask]         # [3, B, L']
```

**Critical post-compression behavior**:
- `inputs_embeds` is sliced from `[B, L, D]` → `[B, L', D]` where L' < L
- `attention_mask` is correspondingly sliced so the LLM only attends to surviving tokens
- `position_ids` is sliced on the **last dimension** (shape `[3, B, L']` because Qwen2.5-Omni uses 3D M-RoPE — temporal, height, width position channels)
- **Position IDs are NOT re-indexed**. The surviving tokens keep their ORIGINAL position IDs. This means the LLM sees "gaps" in position space — position 0, 1, 5, 8, 12, ... — which is correct because RoPE is relative and handles non-contiguous positions naturally.
- `cache_position` is NOT adjusted — it's left as-is, meaning the KV cache for subsequent autoregressive steps starts from the compressed length.

**Also present**: A `random_pruning()` method (line 2358) — a baseline comparison that randomly selects video tokens. This confirms that the authors benchmarked OmniZip against random-drop baselines.

---

### 2.3 `lmms-eval/lmms_eval/models/simple/qwen2_5_omni.py` — Evaluation Wrapper

Registered model class `Qwen2_5_Omni` (decorator: `@register_model("qwen2_5_omni")`).

**Conditional import**: At module load time, checks `os.environ.get("WRAPPER")`:
```python
if method_wrapper == "OmniZip":
    from omnizip.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
else:
    from transformers import Qwen2_5OmniForConditionalGeneration
```

**Constructor** accepts OmniZip params as model_args:
```python
def __init__(self, ..., OMNIZIP_RHO_AUDIO=0.3, OMNIZIP_RHO_VIDEO=0.6,
             OMNIZIP_G=3, OMNIZIP_CONTEXTUAL_RATIO=0.05, ...):
    if os.environ.get("WRAPPER") == "OmniZip":
        self.omnizip_config = {
            "rho_audio": OMNIZIP_RHO_AUDIO,
            "rho_video": OMNIZIP_RHO_VIDEO,
            "g": OMNIZIP_G,
            "contextual_ratio": OMNIZIP_CONTEXTUAL_RATIO,
        }
    else:
        self.omnizip_config = None
```

**Per-sample injection** in `generate_until()`:
```python
self.model.thinker.omnizip_config = self.omnizip_config
self.model.thinker.nframes = videos[0].shape[0]  # actual frame count
```

**Other details**:
- Audio resampled to 16kHz via `librosa.resample()` (Qwen2.5-Omni requirement)
- Long audio split into 5-minute chunks: `split_audio(audio, 4800000)` (4.8M samples = 5 min @ 16kHz)
- Video detection: `VideoFileClip(path).audio is not None` determines `use_audio_in_video`
- Talker is disabled: `self._model.disable_talker()` (text-only output for benchmarks)

---

### 2.4 `eval/eval.py` — Standalone Evaluator

Used for benchmarks NOT in lmms-eval (AVUT, ShortVid-Bench, WorldSense).

**Architecture**:
- Direct import of patched model (no env var needed — uses `--WAPPER-METHOD omnizip` arg)
- Loads JSON benchmark data, iterates samples
- Each sample: load video → build conversation → tokenize → generate → extract answer → compare to ground truth
- Computes per-domain and per-problem-type accuracy breakdowns
- Logs to file + console

**Config paths**: Reads `DATA_PATH` and `VIDEO_DIR` from hardcoded defaults or args.

---

### 2.5 `demo.py` — Minimal Inference

```python
python demo.py --omnizip --rho_audio 0.4 --rho_video 0.7 --video assets/example.mp4
```

Simplest possible pipeline:
1. Load model with `--omnizip` flag → import patched class
2. Build conversation with video + text
3. Set `model.thinker.omnizip_config` and `model.thinker.nframes`
4. Generate with `TextStreamer` for real-time output

---

### 2.6 `eval.sh` — lmms-eval Launch
```bash
export WRAPPER=OmniZip
OMNIZIP_RHO_AUDIO=0.3
OMNIZIP_RHO_VIDEO=0.6
OMNIZIP_G=3
OMNIZIP_CONTEXTUAL_RATIO=0.05
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12347 \
    -m lmms_eval \
    --model qwen2_5_omni \
    --model_args "pretrained=Qwen/Qwen2.5-Omni-7B,attn_implementation=flash_attention_2,
                  max_num_frames=768,
                  OMNIZIP_RHO_AUDIO=${OMNIZIP_RHO_AUDIO},
                  OMNIZIP_RHO_VIDEO=${OMNIZIP_RHO_VIDEO},
                  OMNIZIP_G=${OMNIZIP_G},
                  OMNIZIP_CONTEXTUAL_RATIO=${OMNIZIP_CONTEXTUAL_RATIO}" \
    --tasks videomme --batch_size 1 --output_path ./logs/
```

Note: `--model_args` is a comma-separated key=value string that gets parsed and passed to the `Qwen2_5_Omni.__init__()` constructor.

---

## 3. Token Flow — Complete End-to-End Trace

### 3.1 Concrete Example
Video: 32 frames, 14×14 spatial patches, 2 FPS, 16 seconds long
Audio: 16 seconds of speech at 16kHz

### 3.2 Step-by-Step

#### Step 1: Processor Tokenization
```python
processor(text=text, audio=audios, images=images, videos=videos, ...)
```
Produces:
- `input_ids [1, L]` — contains special token IDs: `audio_token_id`, `video_token_id`, `image_token_id` as placeholders
- `pixel_values_videos [N_v_raw, C, H, W]` — raw video pixel tensors
- `input_features [1, n_mels, T_mel]` — mel spectrogram features
- `feature_attention_mask [1, T_mel]` — audio padding mask

#### Step 2: Audio Encoding
```
input_features [1, 128, T_mel]
    │
    ├── feature_attention_mask → audio_feature_lengths (actual mel length)
    │
    ├── Chunking: split mel into n_window*2-length chunks
    │   chunk_num = ceil(feature_lens / (n_window * 2))
    │   Each chunk padded to same length
    │
    ├── Conv1d(128, d_model, k=3, p=1) → GELU
    ├── Conv2d(d_model, d_model, k=3, s=2, p=1) → GELU  ← 2x temporal downsample
    │
    ├── + Sinusoidal positional embedding
    │
    ├── Pack into variable-length batch (cu_seqlens for flash attention)
    │
    ├── N encoder layers (pre-norm transformer):
    │   for idx, layer in enumerate(self.layers):
    │       if idx == LAST:
    │           output, logits, attn_key = layer(..., return_logits=True)
    │       else:
    │           output, _, _ = layer(..., return_logits=False)
    │
    ├── AvgPool1d(kernel=2, stride=2)  ← another 2x downsample
    ├── LayerNorm
    ├── Linear projection (d_model → output_dim)
    │
    ├── attn_mean reduction:
    │   logits [H, T, T] → mean(dim=0) → [T, T] → sum(dim=0) → [T]
    │   → pairwise-average downsample → [T/2]   (matches AvgPool1d)
    │
    └── Output: audio_features [N_a, D], attn_mean [N_a], attn_key [1, N_a, D]
```

**Effective audio downsampling**: Original mel → Conv2d (÷2) → AvgPool1d (÷2) = **4x temporal compression** before OmniZip even runs.

#### Step 3: Video Encoding
```
pixel_values_videos
    │
    ├── Vision model (Qwen2-VL ViT):
    │   - Patch embedding (14×14 patches)
    │   - ViT transformer layers
    │   - Spatial merge (merge_size × merge_size patches → 1 token)
    │
    └── Output: video_embeds [N_v, D]
         where N_v = num_frames × (H/14/merge_size) × (W/14/merge_size)
```

#### Step 4: Embedding Merge (Thinker Forward)
```python
# Start with text embeddings:
inputs_embeds = self.get_input_embeddings()(input_ids)  # [B, L, D]

# Scatter audio features into audio_token positions:
audio_mask = (input_ids == audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

# Scatter video features into video_token positions:
video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
```

At this point, `inputs_embeds` is a single flat tensor containing all modalities:
```
[BOS] [sys_text...] [audio_bos] [audio₁] [audio₂] ... [audioₙ] [audio_eos]
[video_bos] [video₁] [video₂] ... [videoₘ] [video_eos] [user_text...] [EOS]
```

#### Step 5: M-RoPE Position Computation
```python
position_ids, rope_deltas = self.get_rope_index(
    input_ids, image_grid_thw, video_grid_thw,
    attention_mask, use_audio_in_video, audio_feature_lengths, ...
)
# position_ids shape: [3, B, L]  — three channels for M-RoPE:
#   channel 0: temporal position
#   channel 1: height position
#   channel 2: width position
```

Qwen2.5-Omni uses **Multi-dimensional Rotary Position Embedding (M-RoPE)** which assigns 3D position coordinates to every token:
- Text tokens: all three channels are the same sequential index
- Video tokens: temporal = frame index, height/width = spatial patch position
- Audio tokens: temporal = audio time position, height/width = same as temporal

This is computed BEFORE OmniZip runs, on the FULL uncompressed sequence.

#### Step 6: OmniZip Compression (THE KEY STEP)

```python
inputs_embeds, global_mask = omnizip(
    inputs_embeds,               # [B, L, D]
    attn_logits,                 # [N_a] — per-audio-token saliency score
    input_ids,                   # [B, L] — to find audio/video positions
    audio_token_id,              # int
    video_token_id,              # int
    num_input_frames=self.nframes,  # 32
    merging_ratio_audio=0.3,     # drop 30% of audio tokens
    merging_ratio_v=0.6,         # drop 60% of video tokens on average
    contextual_ratio=0.05,       # 5% of audio tokens become merge anchors
    g=3,                         # up to 3 tokens merged per anchor
)
```

Internally (detailed in Section 4):
```
a) Locate audio_indices, video_indices from input_ids
b) Extract audio_feature [N_a, D], video_feature [N_v, D] from flat_embeds
c) Audio compression:
   - omnizip_audio_attn() → audio_mask [N_a], merge_plan {anchor: [merged...]}
   - Execute merge: modify flat_embeds in-place at anchor positions
d) Compute video_token_per_frame = N_v / num_frames
e) Group audio/video into temporal windows
f) Per-group: audio_retention → video_merging_ratio (inverse mapping)
g) Normalize ratios to hit target rho_video average
h) For each paired group: omnizip_istm() → video_mask
i) Combine: global_mask = ones(L); global_mask[video_indices] = video_mask; global_mask[audio_indices] = audio_mask
j) Return modified flat_embeds and global_mask
```

#### Step 7: Post-Compression Slicing
```python
inputs_embeds  = inputs_embeds[:, global_mask, :]   # [B, L', D]
attention_mask = attention_mask[:, global_mask]       # [B, L']
position_ids   = position_ids[:, :, global_mask]      # [3, B, L']
```

**Why position_ids work without re-indexing**: M-RoPE uses relative positions. Token at position 5 attending to token at position 2 uses rotation angle proportional to |5-2|=3. If we remove position 3 and 4, the surviving tokens at positions 2 and 5 still produce the same relative rotation. The "gap" is semantically meaningful — it tells the LLM "something was here but we compressed it away."

#### Step 8: LLM Forward
```python
outputs = self.model(
    attention_mask=attention_mask,     # compressed
    position_ids=position_ids,         # compressed (with gaps)
    inputs_embeds=inputs_embeds,       # compressed
    use_cache=True,                    # KV cache built from compressed seq
    ...
)
```

The Qwen2 decoder now operates on ~40-60% fewer tokens. Since self-attention is O(n^2), this yields the 3.42x speedup.

#### Concrete Numbers (32 frames, rho_audio=0.3, rho_video=0.6):
```
Component         Before    After     Reduction
─────────────────────────────────────────────
Audio tokens      300       ~210      30% dropped (some merged)
Video tokens      6,272     ~2,509    60% dropped (varies by group)
Text tokens       50        50        0% (never compressed)
System tokens     ~20       ~20       0% (never compressed)
─────────────────────────────────────────────
TOTAL             ~6,642    ~2,789    58% reduction
KV cache          ~53MB     ~22MB     58% reduction
Attention FLOPs   ~44M      ~7.8M    82% reduction (quadratic!)
```

---

## 4. Core Algorithms — Full Mathematical Detail

### 4.1 Algorithm 1: Audio Token Compression — `omnizip_audio_attn()`

**Purpose**: Select which audio tokens to keep, which to drop, and how to merge information from dropped tokens into surviving ones.

**Inputs**:
- `audio_feature` [N_a, D]: Audio token embeddings (from LLM embedding layer, not raw encoder output)
- `video_feature` [N_v, D]: Video token embeddings (same source)
- `attn_logits` [N_a]: Per-token attention saliency score (column-sum of last-layer attention matrix, averaged over heads, downsampled to match)
- `merging_ratio`: fraction to drop (default 0.5, typically 0.3)
- `contextual_ratio`: fraction of remaining tokens to use as merge anchors (default 0.03, typically 0.05)
- `g`: max tokens to merge per anchor (default 3)

**Feature normalization**: If features are 3D (batched), they're reduced to 2D via `mean(dim=0)`. This handles multi-head scenarios.

#### Phase 1: Dominant Token Selection

```
dominant_num = round((1 - merging_ratio) * N_a)
dominant_num = min(dominant_num, attn_logits.size(0))  # safety clamp
_, topk_indices = torch.topk(attn_logits, dominant_num)
keep_mask[topk_indices] = True
```

The `attn_logits` vector is the column-sum of the attention matrix: `attn_mean = A.mean(dim=0).sum(dim=0)` where A is `[H, T, T]`. The column-sum measures **how much each token is attended to by all other tokens** — this is a standard attention-based importance score (used in ViT pruning literature as "attention rollout").

With `merging_ratio=0.3`, this keeps the top 70% most-attended audio tokens.

#### Phase 2: Contextual Anchor Placement

From the REMAINING tokens (those NOT in the dominant set):
```
remaining = all_idx[~keep_mask]        # indices of non-dominant tokens
contextual_num = round(0.05 * N_a)     # e.g., 15 anchors from 300 tokens
step = remaining.numel() // contextual_num
anchors = remaining[::step][:contextual_num]   # uniform stride sampling
keep_mask[anchors] = True
```

These anchors are the "merge targets" — pruned tokens will be merged INTO these anchors rather than simply discarded. The uniform spacing ensures temporal coverage across the entire audio sequence.

#### Phase 3: Cross-Modal Assignment + Weighted Merging

For each pruned token (not dominant, not anchor), find which anchor to merge into:

```python
# L2-normalize features for cosine similarity
a_norm = audio_feature / (audio_feature.norm(dim=-1, keepdim=True) + 1e-6)
v_norm = video_feature / (video_feature.norm(dim=-1, keepdim=True) + 1e-6)

# 1. Assign each pruned token to nearest anchor (audio-audio similarity)
sim_aa = a_norm[pool_global] @ a_norm[anchors].T   # [|pool|, |anchors|]
assign = sim_aa.argmax(dim=1)                        # nearest anchor per pruned token

# 2. Score each pruned token by cross-modal relevance (audio-video similarity)
sim_av = a_norm[pool_global] @ v_norm.T              # [|pool|, N_v]
scores = sim_av.max(dim=1).values                    # max similarity to ANY video token
```

**Why max over video tokens?** Each pruned audio token might correspond to a specific visual event. Taking the max finds the single most-relevant video token and uses that similarity as the importance weight. A pruned audio token strongly related to some video content gets a high score.

```python
# 3. For each anchor, merge its top-G assigned tokens
for c in range(num_anchors):
    candidates = pool_global[assign == c]     # tokens assigned to this anchor
    scores_c = scores[assign == c]            # their cross-modal scores
    topg = min(g, candidates.numel())
    _, sel = torch.topk(scores_c, topg, largest=True)
    chosen = candidates[sel]                  # top-G most video-relevant
    merge_plan[anchor_c] = chosen.tolist()
```

The actual merging happens LATER in `omnizip()` (not in `omnizip_audio_attn()`):

```python
# Back in omnizip(), lines 170-190:
for anchor_rel_idx, merge_rel_list in merge_plan.items():
    merge_rel = torch.tensor(merge_rel_list, device=device)

    # Score merged tokens by their video relevance
    scores = (a_norm[merge_rel] @ v_norm.T).max(dim=1).values
    w = torch.softmax(scores, dim=0)     # normalize to weights

    # Weighted merge:
    anchor_vec = audio_feature[anchor_rel_idx]
    merged_vec = (audio_feature[merge_rel] * w.unsqueeze(-1)).sum(dim=0)
    new_anchor = (anchor_vec + merged_vec) / (1.0 + w.sum())

    # Write back to embedding tensor IN-PLACE
    anchor_global_idx = audio_indices[anchor_rel_idx]
    flat_embeds[anchor_global_idx] = new_anchor
```

**The merge formula**:
```
new_anchor = (anchor_vec + Σᵢ wᵢ · pruned_vec_i) / (1 + Σᵢ wᵢ)
```
Where `wᵢ = softmax(max_v cos(pruned_i, v_j))` across merged tokens. This is a normalized weighted average biased toward tokens with high cross-modal relevance.

---

### 4.2 Algorithm 2: Video Token Compression — `omnizip_istm()`

**ISTM = Interleaved Spatio-Temporal Merging**

**Input**: `video_feature [N_v, D]`, `num_tokens_per_frame` (e.g. 392 = 196×2 for paired groups), `merging_ratio [r₁, r₂]`

The function processes frames in pairs. Within each pair:
- **Frame at even index (t=0, 2, ...)**: Spatial compression
- **Frame at odd index (t=1, 3, ...)**: Temporal compression

#### Spatial Compression (even frames) — DPC-KNN

DPC-KNN = Density Peak Clustering with K-Nearest Neighbors

```python
def dpcknn(tokens, keep_rate=0.5, k=5):
    N = tokens.shape[0]
    num_keep = int(N * keep_rate)

    normed = F.normalize(tokens, dim=1)           # L2 normalize
    sim = torch.mm(normed, normed.T)              # [N, N] cosine similarity
    sim.fill_diagonal_(-inf)                       # exclude self-similarity

    knn_vals, _ = torch.topk(sim, k=5, dim=1)    # top-5 nearest neighbors
    knn_dist = knn_vals.mean(dim=1)               # avg similarity to 5-NN

    # SELECT tokens with LOWEST avg neighbor similarity (most isolated)
    selected = torch.topk(-knn_dist, num_keep).indices
    return selected
```

**Intuition**: In a frame, redundant patches (e.g., blue sky across 50 patches) are all very similar to each other — high knn_dist. Unique/informative patches (edges, objects, text) are more isolated — low knn_dist. DPC-KNN selects the most isolated tokens, ensuring spatial diversity.

**Why `topk(-knn_dist)` instead of `topk(knn_dist, largest=False)`?** Both are equivalent — selecting the tokens with LOWEST density (most isolated). The negation trick avoids the `largest=False` flag.

**k=5 is hardcoded**: This means each token looks at its 5 most similar neighbors. For 196 tokens per frame, k=5 is ~2.5% of tokens — a reasonable local neighborhood.

#### Temporal Compression (odd frames) — Novelty Detection

```python
# For odd frame t:
prev_tokens = video_feature[(t-1) * tpf : t * tpf]      # previous frame tokens
curr_tokens = video_feature[t * tpf : (t+1) * tpf]      # current frame tokens

prev_norm = F.normalize(prev_tokens, p=2, dim=1)
curr_norm = F.normalize(curr_tokens, p=2, dim=1)

# Per-position cosine similarity between current and previous frame
similarity = F.cosine_similarity(curr_norm, prev_norm, dim=1)  # [tpf]

# Keep tokens LEAST similar to previous frame (most novel)
keep_idx = similarity.topk(num_keep, largest=False).indices
```

**Intuition**: If a patch at position (row 3, col 5) looks identical across two consecutive frames, it's temporally redundant — drop it. If it changed (new object appeared, motion), it's novel — keep it.

**Key design choice**: The temporal comparison is **position-aligned** (patch i in frame t vs patch i in frame t-1). This assumes the visual content doesn't move significantly between frames — reasonable at 2 FPS.

#### Merging Ratio Assignment

The `merging_ratio` parameter is a 2-element list `[r₁, r₂]`:
- `r₁` applies to the FIRST pair of frames (frames 0,1): `ratio_id = 0 if t < 2 else 1`
- `r₂` applies to the SECOND pair (frames 2,3)

This allows differential compression within a group: the first pair might be compressed less (more important, often contains the scene-establishing content) while the second pair is compressed more.

#### Mask Output
Returns a boolean mask of shape `[N_v_group]` where True = keep, False = drop.

---

### 4.3 Algorithm 3: Audio-Guided Dynamic Video Ratio Allocation

This is the central innovation — using audio as a **temporal importance signal** for video.

#### Step 1: Temporal Grouping

**Path A** (nframes % 4 == 0):
```python
group_count = num_input_frames // 4
num_video_tokens_per_group = video_feature.shape[0] // group_count
num_audio_tokens_per_group = audio_feature.shape[0] // group_count
```

**Path B** (general):
```python
VIDEO_GROUP_SIZE = video_token_per_frame * 4    # 4 frames per group
AUDIO_GROUP_SIZE = 50                           # fixed 50 audio tokens per group

# Synchronous iteration:
while v_ptr + VIDEO_GROUP_SIZE <= N_v and a_ptr + AUDIO_GROUP_SIZE <= N_a:
    video_groups.append((v_ptr, v_ptr + VIDEO_GROUP_SIZE))
    audio_groups.append((a_ptr, a_ptr + AUDIO_GROUP_SIZE))
    v_ptr += VIDEO_GROUP_SIZE
    a_ptr += AUDIO_GROUP_SIZE
# Handle tail remainder
```

#### Step 2: Audio Retention Score Per Group

```python
for each group i:
    audio_retention[i] = audio_mask[start:end].float().mean()
    # Fraction of audio tokens KEPT in this temporal window
```

A group where 80% of audio tokens survived the dominant selection has `retention=0.8` — this is an information-dense window (lots of speech, sound effects). A group with only 30% retained has sparse audio — less important.

#### Step 3: Inverse Mapping to Video Compression

```python
min_ratio, max_ratio = 0.35, 0.75    # bounds on per-group video merging ratio

for retention in audio_group_retention:
    mapped = max_ratio + (min_ratio - max_ratio) * retention
    #       = 0.75   + (0.35 - 0.75) * retention
    #       = 0.75   - 0.4 * retention
    base_vs.append(clamp(mapped, 0.35, 0.75))
```

**The mapping is LINEAR AND INVERSE**:
| Audio Retention | Formula | Video Merging Ratio | Meaning |
|---|---|---|---|
| 0.0 (no audio) | 0.75 - 0.4×0 = 0.75 | 75% dropped | Aggressive compression |
| 0.5 (moderate) | 0.75 - 0.4×0.5 = 0.55 | 55% dropped | Moderate compression |
| 1.0 (all kept) | 0.75 - 0.4×1.0 = 0.35 | 35% dropped | Light compression |

#### Step 4: Global Budget Normalization

The per-group ratios must average to `rho_video`:

```python
target_total = rho_video * n_groups    # e.g., 0.6 * 8 = 4.8

# Fix the min and max groups (don't change the extremes)
# Scale the middle groups to hit the target
remain_target = target_total - min_val - max_val
scaling = remain_target / sum(middle_groups)

for middle groups:
    adjusted[i] = clamp(base[i] * scaling, 0.35, 0.75)

# If still off target, distribute remainder evenly
diff = target_total - sum(adjusted)
for middle groups:
    adjusted[i] += diff / num_middle_groups
```

This ensures the overall compression ratio matches the user-specified `rho_video` while allowing per-group variation.

#### Step 5: Apply ISTM Per Paired Group

Groups are processed in PAIRS (2 groups = 8 frames per ISTM call):

```python
for i in range(0, group_count, 2):
    group_feat = video_feature[v_start : v_end]    # 8 frames of tokens
    video_merging_ratio = [ratios[i], ratios[i+1]]  # 2-element list

    if group_len == expected:  # not a tail group
        group_mask = omnizip_istm(
            group_feat,
            num_tokens_per_frame = video_token_per_frame * 2,  # 2 groups
            merging_ratio = video_merging_ratio
        )
    else:  # tail/remainder group — keep all tokens (no compression)
        group_mask = torch.ones(group_len, ...)
```

**Why paired groups?** ISTM's temporal compression compares frame t with frame t-1. By processing 8 frames (2 groups of 4), it can detect temporal redundancy across the group boundary.

**Tail handling**: If the video doesn't divide evenly into groups, the remainder is kept uncompressed. This prevents edge artifacts from compressing partial groups.

---

## 5. Hyperparameters — Complete Reference

| Parameter | Symbol | Default (demo) | Default (eval) | Range | Effect |
|---|---|---|---|---|---|
| `rho_audio` | ρ_a | 0.4 | 0.3 | [0, 1] | Fraction of audio tokens to DROP. 0=keep all, 1=drop all |
| `rho_video` | ρ_v | 0.7 | 0.6 | [0, 1] | Average fraction of video tokens to DROP |
| `g` | G | 3 | 3 | [0, ∞) | Max tokens merged per contextual anchor. 0=pure pruning |
| `contextual_ratio` | c | 0.05 | 0.05 | [0, 1] | Fraction of non-dominant audio tokens used as merge anchors |
| `min_ratio` | - | 0.35 | 0.35 | hardcoded | Floor for per-group video merging ratio |
| `max_ratio` | - | 0.75 | 0.75 | hardcoded | Ceiling for per-group video merging ratio |
| `k` (DPC-KNN) | k | 5 | 5 | hardcoded | KNN neighborhood size for spatial diversity |
| `AUDIO_GROUP_SIZE` | - | 50 | 50 | hardcoded | Audio tokens per temporal group (Path B) |

### Sensitivity Analysis

**ρ_v (video)** is the most impactful parameter:
- `ρ_v = 0.4`: ~2x speedup, best accuracy retention
- `ρ_v = 0.6`: ~3.4x speedup, minimal accuracy loss (paper default)
- `ρ_v = 0.8`: ~5x speedup, noticeable degradation on fine-grained tasks

**ρ_a (audio)** is less sensitive because audio tokens are far fewer:
- `ρ_a = 0.1-0.3`: Typical range, merging recovers most lost information
- `ρ_a > 0.5`: Can lose important speech content

**G (merge count)** controls information recovery:
- `G = 0`: Pure pruning — dropped tokens are simply discarded
- `G = 3`: Moderate merging — each anchor absorbs up to 3 neighbors
- `G > 5`: Diminishing returns — merged representation becomes blurry

---

## 6. Utilities and Dependencies — Deep Dive

### 6.1 Audio Processing Chain
```
qwen-omni-utils:
  process_mm_info(conversation, use_audio_in_video=True)
    → Extracts audio from video file
    → Returns: audios, images, videos (as numpy/tensor)

librosa:
  librosa.resample(audio, orig_sr=X, target_sr=16000)
    → Qwen2.5-Omni requires exactly 16kHz input

moviepy:
  VideoFileClip(path).audio is not None
    → Determines use_audio_in_video flag
    → If no audio track: falls back to video-only mode (no audio compression)

split_audio(audio, 4800000):
  → Splits audio into 5-minute chunks (4.8M samples @ 16kHz)
  → Required because Whisper encoder has max context length
```

### 6.2 Video Processing
```
Qwen2.5-Omni processor:
  VIDEO_MAX_PIXELS = 128 * 28 * 28 = 100,352 (paper setting)
  FPS = 2.0
  FPS_MAX_FRAMES = 768
  → At 2 FPS: max video length = 768/2 = 384 seconds (6.4 minutes)

Vision encoder:
  ViT patches: 14×14 pixels
  Spatial merge: 2×2 (default) → 196 → 49 tokens per frame
  Or merge_size=1: 196 tokens per frame (used in paper: 128*28*28 / (14*14) = 128 patches)
```

### 6.3 Model Architecture Dependencies
```
flash-attn >= 2.1.0:
  - Required for flash_attn_varlen_func in audio encoder
  - Audio encoder uses variable-length packing (cu_seqlens) for efficiency
  - The "varlen" variant handles multiple audio chunks in a single batch

transformers (PINNED VERSION CRITICAL):
  - omnizip/modeling_qwen2_5_omni.py is a FULL COPY of HF's file
  - Any HF transformers update that changes the model code BREAKS OmniZip
  - Must match the exact version used to generate the copy

accelerate:
  - For multi-GPU inference
  - Used in eval.sh with --num_processes flag
```

### 6.4 File Dependency Graph
```
demo.py ─────────────────────────┐
eval/eval.py ───────────────────┐│
eval.sh ──→ lmms_eval ──→      ││
  lmms_eval/models/simple/      ││
    qwen2_5_omni.py ──────────┐ ││
                               ▼ ▼▼
                     omnizip/modeling_qwen2_5_omni.py
                               │
                               ▼
                     omnizip/omnizip_units.py
                        ├── omnizip_audio_attn()
                        ├── omnizip_istm()
                        └── omnizip()
```

---

## 7. Training / Inference Pipeline — Complete Reference

### 7.1 Training
**NONE**. OmniZip is entirely training-free. This is its key selling point over methods like Token Merging (ToMe) which require fine-tuning to recover from compression artifacts.

The compression decisions are made using:
- **Frozen audio encoder attention weights** — pre-existing in the pretrained model
- **Cosine similarity between embeddings** — computed at inference time
- **Heuristic threshold functions** — the linear mapping from audio retention to video ratio

### 7.2 Inference Modes

#### Mode 1: Quick Demo
```bash
python demo.py --omnizip \
    --rho_audio 0.4 --rho_video 0.7 --g 3 --contextual_ratio 0.05 \
    --video assets/example.mp4
```

#### Mode 2: lmms-eval Benchmarks
```bash
export WRAPPER=OmniZip
bash eval.sh
# Runs VideoMME by default; edit eval.sh for other lmms-eval tasks
```

#### Mode 3: Custom Benchmarks
```bash
python eval/eval.py --WAPPER-METHOD omnizip \
    --OMNIZIP_RHO_AUDIO 0.3 --OMNIZIP_RHO_VIDEO 0.6 \
    --OMNIZIP_G 3 --OMNIZIP_CONTEXTUAL_RATIO 0.05
```

### 7.3 Benchmarks and Datasets

| Benchmark | Type | Metrics | Loader | Dataset Source |
|---|---|---|---|---|
| VideoMME | Video QA (S/M/L) | Accuracy | lmms-eval | HuggingFace |
| AVUT | Audio-Visual Understanding | Accuracy | eval/eval.py | HuggingFace: tsinghua-ee/AVUTBenchmark |
| ShortVid-Bench | Short Video QA | Accuracy | eval/eval.py | HuggingFace: TencentARC/ShortVid-Bench |
| WorldSense | Spatial/Temporal Reasoning | Accuracy | eval/eval.py | Custom (jaaackhongggg.github.io) |

### 7.4 Hardware Requirements
- **Minimum**: 1x GPU with ~24GB VRAM (with VIDEO_MAX_PIXELS = 128*28*28)
- **Recommended**: 1x A100-80GB or H100 (for max_num_frames=768)
- **Flash Attention 2**: Required (CUDA ≥ 11.6, PyTorch ≥ 2.0)
- `attn_implementation="flash_attention_2"` must be explicitly specified

---

## 8. Architectural Insights and Design Decisions

### 8.1 Why Audio as a Guide?

Audio naturally encodes **temporal saliency** — speech segments, sound effects, and music all indicate "something is happening here." The audio encoder's self-attention already computes which audio tokens are most important (high inter-token attention = high information density). OmniZip repurposes this existing computation as a free saliency signal for video.

This is more principled than:
- **Uniform compression**: Compresses all windows equally, wastes budget on silence/static
- **Random pruning**: No semantic awareness
- **Attention-based video pruning (FastV)**: Only uses vision attention, ignores audio-visual alignment

### 8.2 Why the Column-Sum of Attention?

The attention matrix A[i,j] = "how much does token i attend to token j." The column sum Σᵢ A[i,j] measures "how much is token j attended to by ALL other tokens" — this is the **collective importance** of token j. A token that many others attend to is a hub of information.

### 8.3 Why Last Layer Only?

Attention patterns in early transformer layers tend to be diffuse (spread across many tokens). Later layers develop sharper, more semantically meaningful attention patterns. The last layer's attention is the most refined signal for importance.

### 8.4 Why Interleaved (Not Pure Spatial or Pure Temporal)?

Pure spatial compression loses temporal structure. Pure temporal compression loses spatial detail. By alternating:
- Even frames: spatial DPC-KNN preserves diverse spatial features
- Odd frames: temporal novelty detection preserves changes over time

This interleaving ensures both dimensions of information are retained.

### 8.5 Why In-Place Embedding Modification?

OmniZip modifies `flat_embeds` in-place during audio merging (line 190: `flat_embeds[anchor_global_idx] = new_anchor`). This is efficient — no tensor copy needed — and safe because:
1. The modified positions are only anchor positions (which survive the mask)
2. The merge combines information from pruned tokens INTO surviving ones
3. The pruned tokens' positions will be masked out by `global_mask` anyway

### 8.6 The Two Code Paths

`omnizip()` has two implementations (Path A: lines 192-289, Path B: lines 291-416) that achieve the same thing. Path A is triggered when `num_input_frames % 4 == 0` and uses simpler math (divides evenly). Path B handles arbitrary frame counts with explicit pointer-based grouping. This is likely an artifact of iterative development — Path B is the general case; Path A is an optimized special case.

### 8.7 Position ID Preservation (Not Re-indexing)

After compression, position IDs are sliced but NOT re-numbered. Token originally at position 500 keeps position 500 even if positions 490-499 were removed. This works because:
1. Qwen2.5-Omni uses M-RoPE (rotary positional embedding)
2. RoPE encodes RELATIVE distances between tokens
3. Gaps in position space are handled naturally — they simply increase the relative distance
4. The LLM interprets the gap as "compressed region" — semantically correct

### 8.8 Edge Cases and Safety

- **No audio track**: If `attn_logits is None` (no audio features), OmniZip is skipped entirely — falls through to uncompressed forward pass.
- **Tail groups**: Remainder frames/audio that don't fill a complete group are kept uncompressed (`torch.ones` mask).
- **Single frame**: If `num_tokens_per_frame` can't form even one group, ISTM returns all-True mask.
- **Empty merge plan**: If `g=0` or no tokens to merge, `merge_plan` is empty dict — no merging occurs, pure pruning.
- **Odd-length sequences**: Audio encoder handles odd T by dropping last element before pairwise averaging.

---

## 9. Comparison with Related Methods

| Method | Training-Free | Cross-Modal | Audio+Video | Speedup | Memory |
|---|---|---|---|---|---|
| FastV | Yes | No (vision only) | No | ~2x | ~1.2x |
| Token Merging (ToMe) | No (needs fine-tune) | No | No | ~2x | ~1.3x |
| LLaVA-PruMerge | Yes | No (vision only) | No | ~2x | ~1.2x |
| **OmniZip** | **Yes** | **Yes** | **Yes** | **3.42x** | **1.4x** |

OmniZip's advantage is primarily architectural: by compressing BOTH modalities and using cross-modal signals for budget allocation, it achieves higher compression ratios without proportional accuracy loss.
