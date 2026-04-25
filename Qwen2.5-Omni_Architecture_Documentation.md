# Qwen2.5-Omni Audio & Vision Encoder Architecture

## File: modeling_qwen2_5_omni_low_VRAM_mode.py

---

## AUDIO ENCODER (Lines 600-1017)

### Qwen2_5OmniAudioAttention (Lines 600-668)
- **embed_dim**: config.d_model
- **num_heads**: config.encoder_attention_heads
- **head_dim**: embed_dim / num_heads
- **Scaling**: head_dim^-0.5
- **Projections**: k_proj(no_bias), v_proj(bias), q_proj(bias), out_proj(bias)
- **Dropout**: config.attention_dropout

### Qwen2_5OmniAudioEncoderLayer (Lines 759-843)
**Pre-norm Architecture**:
1. self_attn_layer_norm(LayerNorm)
2. self_attn(attention)
3. Residual add
4. final_layer_norm(LayerNorm)
5. fc1(d_model → encoder_ffn_dim)
6. activation_fn(GELU or config)
7. fc2(encoder_ffn_dim → d_model)
8. Residual add

**Components**:
- dropout: config.dropout
- activation_dropout: config.activation_dropout

### Qwen2_5OmniAudioEncoder (Lines 844-1017)
**Pipeline**:
1. conv1: Conv1d(num_mel_bins → d_model, k=3, p=1)
2. GELU activation
3. conv2: Conv1d(d_model → d_model, k=3, s=2, p=1)
4. GELU activation
5. SinusoidsPositionEmbedding(max_source_positions, d_model)
6. Stacked AudioEncoderLayer x encoder_layers
7. ln_post: LayerNorm(d_model)
8. avg_pooler: AvgPool1d(k=2, s=2)
9. proj: Linear(d_model → output_dim)
10. audio_bos_eos_token: Embedding(2, output_dim)

**Windowed Attention (n_window)**:
- Chunks input into windows of size n_window * 2
- cu_seqlens: cumulative sequence lengths marking boundaries
- Attention mask: attention_mask[cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = 0 (enabled)
- Cross-window: torch.finfo().min (disabled)

**Sinusoidal Pos Embedding**:
- pos_emb[t, 2i] = sin(t / 10000^(2i/d_model))
- pos_emb[t, 2i+1] = cos(t / 10000^(2i/d_model))

---

## VISION ENCODER (Lines 1018-1400)

### Qwen2_5OmniVisionAttention (Lines 1018-1056)
- **dim**: config.hidden_size
- **num_heads**: 16 (default)
- **head_dim**: dim / num_heads
- **Projections**: q, k, v, proj (all Linear(dim, dim, bias=True))
- **Position Embedding**: apply_rotary_pos_emb_vision(q/k, rotary_pos_emb)
- **Attention Mask**: cu_seqlens boundaries (cross-image)

### Qwen2_5OmniVisionBlock (Lines 1144-1160)
**Pre-norm Block**:
1. norm1: Qwen2RMSNorm(hidden_size, eps=1e-6)
2. attn: VisionAttention with rotary embeddings
3. Residual add
4. norm2: Qwen2RMSNorm(hidden_size, eps=1e-6)
5. mlp: Qwen2_5OmniMLP (gated)
6. Residual add

**MLP (Gated SwiGLU)**:
- gate_proj: Linear(hidden_size → intermediate_size, bias=True)
- up_proj: Linear(hidden_size → intermediate_size, bias=True)
- down_proj: Linear(intermediate_size → hidden_size, bias=True)
- act_fn: GELU
- Forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))

### Qwen2_5_VisionPatchEmbed (Lines 1162-1185)
- **3D Conv**: Conv3d(in_channels, embed_dim, kernel=[temporal_patch_size, patch_size, patch_size], stride=kernel, bias=False)
- **Input view**: (B, C=3, temporal_patch_size, patch_size, patch_size)
- **Output**: (num_patches, embed_dim)

### Qwen2_5_VisionRotaryEmbedding (Lines 1187-1200)
- **theta**: 10000.0 (base)
- **inv_freq**: 1.0 / (theta ^ (torch.arange(0, dim, 2) / dim))
- **Forward**: seq = torch.arange(seqlen); freqs = torch.outer(seq, inv_freq)
- **Formula**: theta_i = base^(-2i/d_model); freq[pos, i] = pos * theta_i

### Qwen2_5OmniPatchMerger (Lines 1202-1216)
- **hidden_size**: context_dim * (spatial_merge_size ^ 2)
- **ln_q**: Qwen2RMSNorm(context_dim, eps=1e-6)
- **mlp**: Sequential[Linear(hidden_size → hidden_size), GELU, Linear(hidden_size → dim)]
- **Forward**: mlp(ln_q(x).view(-1, hidden_size))
- **Spatial Merge**: 2x2 reduction (default spatial_merge_size=2)

### Qwen2_5OmniVisionEncoder (Lines 1216-1400)
**Architecture**:
1. patch_embed: Qwen2_5_VisionPatchEmbed
2. rotary_pos_emb: Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
3. blocks: ModuleList[VisionBlock] x config.depth
4. merger: PatchMerger

**Attention Modes**:
- fullatt_block_indexes: List of layer indices using FULL attention
- Other blocks: Use WINDOWED attention (window_size, spatial_merge_size)

**Position Encoding**:
- Separates height, width, temporal indices
- rot_pos_emb: Computes rotary embeddings per dimension
- Accounts for spatial merging in position calculation

**Window Attention**:
- vit_merger_window_size = window_size / spatial_merge_size / patch_size
- get_window_index: Returns window_index and cu_window_seqlens

---

## DATA FLOW SUMMARY

### Audio
Input → Conv1d(GELU) → Conv1d(GELU) → Add Pos Emb 
→ [Attn + FFN] x N → LayerNorm → AvgPool → Linear → Output

### Vision
Frames → PatchEmbed(3D Conv) → [Attn(rotary) + MLP] x depth → PatchMerger → Output

---

## DIMENSION DETAILS

**Audio**:
- embed_dim: d_model (256-768)
- head_dim: d_model / encoder_attention_heads
- ffn_dim: encoder_ffn_dim (4x d_model typical)
- num_mel_bins: 80-128
- output_dim: 1024-2048

**Vision**:
- hidden_size: 1152-1536
- head_dim: hidden_size / 16 ≈ 72-96
- intermediate_size: 4x hidden_size
- out_hidden_size: 1024-2048
- patch_size: 14 (spatial)
- temporal_patch_size: 2
- spatial_merge_size: 2 (4x reduction per merge)

---

## NORMALIZATION & ACTIVATION

| Layer | Norm | Activation |
|-------|------|------------|
| AudioEncoderLayer | LayerNorm | GELU |
| VisionBlock norm1 | RMSNorm(1e-6) | - |
| VisionBlock norm2 | RMSNorm(1e-6) | - |
| VisionBlock MLP | - | GELU (gated) |
| PatchMerger | RMSNorm(1e-6) | GELU |

---

## KEY FEATURES

1. **Windowed Attention**: Audio uses cu_seqlens for chunked processing
2. **Rotary Embeddings**: Vision uses RoPE instead of learnable pos embs
3. **Gating**: Vision MLP uses gated activation
4. **RMSNorm**: Vision uses RMSNorm (eps=1e-6)
5. **3D Convolution**: Vision handles spatio-temporal patches
6. **Pre-norm**: Both use pre-norm architecture
7. **Sinusoidal Pos Emb**: Audio uses sinusoidal encoding
