# VideoZip: Video-Guided Audio Token Compression — Ultra-Plan

## Executive Summary

OmniZip uses **audio to guide video** pruning. This plan designs the exact inverse:
**video guides audio** compression, with audio providing cross-modal anchoring during
video token selection. This is training-free (unlike OmniSIFT, which trains 4.85M params).

---

## 1. Literature Gap

| System | Guide direction | Training-free | Cross-modal anchor |
|---|---|---|---|
| OmniZip (CVPR 2026) | Audio → Video | Yes | Video → Audio anchors |
| OmniSIFT (arXiv 2602.04804, Feb 2026) | Video → Audio | **No** (4.85M params, STE) | — |
| **VideoZip (proposed)** | **Video → Audio** | **Yes** | **Audio → Video anchors** |

VideoZip fills the exact gap: training-free, video-guided, with bidirectional cross-modal anchoring.

---

## 2. Current OmniZip Algorithm (Baseline)

```
input_embeds + attn_logits + input_ids
        │
        ├─ omnizip_audio_attn()
        │     ├─ importance = attn_logits.mean(0).sum(0)  → align to N_audio
        │     ├─ topk → keep_mask (dominant audio tokens)
        │     ├─ contextual anchors from remaining
        │     └─ sim_av = a_norm @ v_norm.T  [VIDEO guides AUDIO anchors]
        │           → merge_plan
        │
        ├─ merge pruned audio into anchors (weights from sim_av)
        │
        ├─ audio_group_retention[i] = mean(audio_mask[group_i])
        │
        ├─ audio_group_retention → video_merging_ratios
        │     mapped = max_ratio + (min_ratio - max_ratio) * retention
        │     [HIGH audio retention → LOW video pruning ratio]
        │
        └─ omnizip_istm() with video_merging_ratios
              ├─ even frames: dpcknn() spatial deduplication
              └─ odd frames: cosine sim to prev frame
```

---

## 3. VideoZip Algorithm (Full Role Swap)

```
input_embeds + attn_logits + input_ids
        │
        ├─ omnizip_video_saliency()          [NEW]
        │     ├─ importance = attn_logits.mean(0).sum(0)
        │     ├─ vid_importance = importance[video_indices]  [exact, not truncation]
        │     ├─ per_frame_imp = vid_importance.reshape(T, S).mean(1)   S=tokens/frame
        │     ├─ per_group_imp = per_frame_imp.reshape(G, F_per_G).mean(1)
        │     └─ video_group_retention[i] = normalize(per_group_imp[i])
        │
        ├─ video_group_retention → audio_merging_ratios   [SWAPPED]
        │     mapped = max_ratio + (min_ratio - max_ratio) * retention
        │     [HIGH video retention → LOW audio pruning ratio]
        │
        ├─ omnizip_audio_compress()          [NEW — per group]
        │     For each (audio_group, audio_merging_ratio):
        │       ├─ call omnizip_audio_attn(group_feat, video_feat, ratio)
        │       │     sim_av = a_norm @ v_norm.T  [video still guides audio anchors]
        │       └─ merge pruned audio into anchors (weights from sim_av)
        │
        └─ omnizip_istm_audio_anchored()    [MODIFIED]
              ├─ even frames: dpcknn_with_audio(tokens, audio_feat)
              │     score = diversity_score + beta * max_sim(v_token, audio)
              │     [AUDIO guides VIDEO anchor selection — the true swap]
              └─ odd frames: cosine sim to prev frame (unchanged)
```

### Key Swaps vs OmniZip

| Component | OmniZip | VideoZip |
|---|---|---|
| Saliency target | Audio tokens | Video tokens (per-frame aggregated) |
| Temporal guide | audio_retention → video_ratios | video_retention → audio_ratios |
| Anchor cross-modal | video → audio anchor scores | audio → video anchor scores |
| Compression order | audio first, then video | video saliency first, then audio |
| Fixed ratio modality | Audio (rho_audio) | Video (rho_video, or from vid saliency) |

---

## 3.6. L6-Cached Video Saliency Path (Optional, Headline Speedup)

**Empirical basis:** Thinker L6 already encodes a question-invariant audio saliency signal
(documented in `docs/method_l6_omnizip.md`, `docs/findings.md`). We hypothesize the same
holds for **video** tokens — early layers form a video-stable saliency map before any
question conditioning.

**Pipeline:**
```
First query on a video:
  full forward up to L6 → hidden_states[L6] → video_saliency_l6 → cache[video_id]
Subsequent queries:
  load cache[video_id] → skip the L6 forward entirely → use cached saliency
```

**Why this is novel:**
- OmniZip+L6 cache (existing) caches AUDIO saliency only. Video pruning still needs live attn.
- VideoZip+L6 cache flips this: VIDEO saliency is the cached signal (and now drives audio compression too).
- Net: one cached forward → both modalities pruned without per-query saliency compute.

**Substitution point:** Replaces only Step 1 of `omnizip_video_saliency()`. Steps 2–4
(spatial aggregation, frame grouping, normalization) are unchanged. See §4a-L6 below.

---

## 4. New Functions (Implementation Spec)

### 4a. `omnizip_video_saliency()` — `omnizip_units.py`

```python
def omnizip_video_saliency(
    video_feature: torch.Tensor,          # [N_v, D]
    video_indices: torch.Tensor,          # positions in full sequence
    attn_logits: torch.Tensor,            # [H, T, T] full attention matrix
    num_input_frames: int,
    video_token_per_frame: int,
    num_groups: int,
) -> List[float]:                         # video_group_retention[G]
    """
    Returns per-group video retention scores in [0,1].
    High score = this group is salient = audio in this group should be compressed LESS.
    """
    # Step 1: per-token importance at video positions
    if attn_logits.dim() == 3:
        importance = attn_logits.mean(dim=0).sum(dim=0)   # [T]
    else:
        importance = attn_logits.sum(dim=0)
    
    N_v = video_feature.shape[0]
    vid_importance = importance[video_indices[:N_v]]       # exact alignment, not truncation
    
    # Step 2: aggregate spatial tokens → per frame
    frames_x_spatial = vid_importance.reshape(num_input_frames, video_token_per_frame)
    frame_importance = frames_x_spatial.mean(dim=1)       # [T_frames]
    
    # Step 3: aggregate frames → per temporal group
    frames_per_group = max(1, num_input_frames // num_groups)
    group_importances = []
    for g in range(num_groups):
        s = g * frames_per_group
        e = s + frames_per_group if g < num_groups - 1 else num_input_frames
        group_importances.append(frame_importance[s:e].mean().item())
    
    # Step 4: normalize to [0, 1]
    mx = max(group_importances) + 1e-8
    return [v / mx for v in group_importances]
```

**Design rationale:**
- Uses `video_indices` for exact alignment (fixes OmniZip's truncation hack for video)
- Spatial averaging is necessary because video has H×W tokens per frame (not 1D like audio)
- Returns raw retention scores (normalized), not a mask — the mask is built per-group by `omnizip_audio_compress()`

---

### 4a-L6. `omnizip_video_saliency_l6()` — cached variant

```python
def omnizip_video_saliency_l6(
    video_feature: torch.Tensor,
    video_indices: torch.Tensor,
    l6_hidden_states: torch.Tensor,       # [T, D] — cached from prior forward
    num_input_frames: int,
    video_token_per_frame: int,
    num_groups: int,
) -> List[float]:
    """
    Drop-in replacement for omnizip_video_saliency() that reads from a cached L6
    activation snapshot instead of live attention logits.

    Importance scoring at L6: per-token L2 norm of hidden state (matches L6-cache
    audio path in docs/method_l6_omnizip.md).
    """
    N_v = video_feature.shape[0]
    vid_hidden = l6_hidden_states[video_indices[:N_v]]
    vid_importance = vid_hidden.norm(dim=-1)               # [N_v]

    # Steps 2-4 identical to omnizip_video_saliency()
    frames_x_spatial = vid_importance.reshape(num_input_frames, video_token_per_frame)
    frame_importance = frames_x_spatial.mean(dim=1)
    frames_per_group = max(1, num_input_frames // num_groups)
    group_importances = []
    for g in range(num_groups):
        s = g * frames_per_group
        e = s + frames_per_group if g < num_groups - 1 else num_input_frames
        group_importances.append(frame_importance[s:e].mean().item())
    mx = max(group_importances) + 1e-8
    return [v / mx for v in group_importances]
```

**Cache key:** video file hash (or stable video_id) — saliency is question-invariant per
the L6 finding (within-video std 0.0013).

---

### 4a-Sim. `omnizip_video_saliency_simonly()` — Aurelle's frozen-cross-modal baseline

```python
def omnizip_video_saliency_simonly(
    video_feature: torch.Tensor,           # [N_v, D]
    audio_feature: torch.Tensor,           # [N_a, D]
    num_input_frames: int,
    video_token_per_frame: int,
    num_groups: int,
) -> List[float]:
    """
    Saliency from frozen cross-modal embedding similarity only — no attention logits.
    Tests whether attn-logit importance is doing real work over raw embedding alignment.
    """
    v_norm = torch.nn.functional.normalize(video_feature, dim=-1)
    a_norm = torch.nn.functional.normalize(audio_feature, dim=-1)
    av_sim = v_norm @ a_norm.T                              # [N_v, N_a]
    vid_importance = av_sim.max(dim=1).values               # [N_v]

    frames_x_spatial = vid_importance.reshape(num_input_frames, video_token_per_frame)
    frame_importance = frames_x_spatial.mean(dim=1)
    frames_per_group = max(1, num_input_frames // num_groups)
    group_importances = []
    for g in range(num_groups):
        s = g * frames_per_group
        e = s + frames_per_group if g < num_groups - 1 else num_input_frames
        group_importances.append(frame_importance[s:e].mean().item())
    mx = max(group_importances) + 1e-8
    return [v / mx for v in group_importances]
```

---

### 4b. `omnizip_audio_compress()` — `omnizip_units.py`

```python
def omnizip_audio_compress(
    audio_feature: torch.Tensor,            # [N_a, D]
    video_feature: Optional[torch.Tensor],  # [N_v, D]
    attn_logits: torch.Tensor,
    audio_groups: List[Tuple[int, int]],    # [(a_start, a_end), ...]
    audio_merging_ratios: List[float],      # one ratio per group
    contextual_ratio: float = 0.05,
    g: int = 3,
) -> Tuple[torch.Tensor, Dict[int, List[int]]]:
    """Per-group audio compression with video-derived ratios."""
    N_a = audio_feature.shape[0]
    audio_mask = torch.zeros(N_a, dtype=torch.bool, device=audio_feature.device)
    merge_plan: Dict[int, List[int]] = {}
    
    for (a_start, a_end), ratio in zip(audio_groups, audio_merging_ratios):
        if a_start >= a_end:
            continue
        group_feat = audio_feature[a_start:a_end]
        # video_feature passed in full — omnizip_audio_attn handles cross-modal
        group_mask, group_plan = omnizip_audio_attn(
            audio_feature=group_feat,
            video_feature=video_feature,
            attn_logits=attn_logits,
            merging_ratio=ratio,
            contextual_ratio=contextual_ratio,
            g=g,
        )
        audio_mask[a_start:a_end] = group_mask
        for anchor, merges in group_plan.items():
            merge_plan[anchor + a_start] = [m + a_start for m in merges]
    
    return audio_mask, merge_plan
```

**Note:** Re-uses `omnizip_audio_attn()` per group — video still guides audio anchor selection
within each group (via `sim_av`). This is correct: even in VideoZip, video helps SELECT
which audio tokens to keep; but now VIDEO tells audio HOW MUCH to prune (the ratio).

---

### 4c. `omnizip_istm_audio_anchored()` — `omnizip_units.py`

```python
def omnizip_istm_audio_anchored(
    video_feature: torch.Tensor,
    audio_feature: Optional[torch.Tensor],  # [N_a, D] — guides video anchor selection
    num_tokens_per_frame: int = 196,
    merging_ratio: List[float] = [0.7, 0.7],
    audio_anchor_beta: float = 0.3,         # weight for audio-similarity term
):
    """ISTM with audio-guided anchor selection (the true directional swap)."""
    
    def dpcknn_audio_guided(tokens, audio, keep_rate=0.5, k=5, beta=0.3):
        N = tokens.shape[0]
        num_keep = int(N * keep_rate)
        if num_keep >= N:
            return torch.arange(N, device=tokens.device)
        
        with torch.no_grad():
            # Original diversity score
            normed = torch.nn.functional.normalize(tokens, dim=1)
            sim = torch.mm(normed, normed.T)
            sim.fill_diagonal_(-float('inf'))
            knn_vals, _ = torch.topk(sim, min(k, max(1, N-1)), dim=1)
            diversity_score = knn_vals.mean(dim=1)      # lower = more isolated = keep
            
            if audio is not None and audio.numel() > 0:
                # Audio-visual alignment score per video token
                a_norm = torch.nn.functional.normalize(audio, dim=1)
                av_sim = torch.mm(normed, a_norm.T)     # [N_v, N_a]
                audio_sim = av_sim.max(dim=1).values    # [N_v]  max sim to any audio token
                
                # Combined: keep tokens that are diverse AND audio-aligned
                # diversity_score is negative (lower = better isolated), so negate it
                combined = -diversity_score + beta * audio_sim
            else:
                combined = -diversity_score
            
            selected = torch.topk(combined, min(num_keep, N), largest=True).indices
        return selected
    
    # Rest of ISTM loop unchanged except dpcknn → dpcknn_audio_guided
    num_frames = video_feature.shape[0] // num_tokens_per_frame
    mask = torch.zeros(video_feature.shape[0], dtype=torch.bool, device=video_feature.device)
    
    for t in range(num_frames):
        ratio_id = 0 if t < 2 else 1
        keep_ratio = 1.0 - merging_ratio[ratio_id]
        start_idx = t * num_tokens_per_frame
        end_idx = (t + 1) * num_tokens_per_frame
        tokens = video_feature[start_idx:end_idx]
        
        if t % 2 == 0:
            keep_idx = dpcknn_audio_guided(tokens, audio_feature, keep_rate=keep_ratio, beta=audio_anchor_beta)
            mask[start_idx:end_idx][keep_idx] = True
        else:
            prev_tokens = video_feature[(t-1)*num_tokens_per_frame : t*num_tokens_per_frame]
            prev_norm = torch.nn.functional.normalize(prev_tokens, p=2, dim=1)
            curr_norm = torch.nn.functional.normalize(tokens, p=2, dim=1)
            similarity = torch.nn.functional.cosine_similarity(curr_norm, prev_norm, dim=1)
            num_keep = int(num_tokens_per_frame * keep_ratio)
            keep_idx = similarity.topk(num_keep, largest=False).indices if num_keep < num_tokens_per_frame else torch.arange(num_tokens_per_frame, device=tokens.device)
            mask[start_idx:end_idx][keep_idx] = True
    
    return mask
```

**Design rationale:**
- `dpcknn_audio_guided`: blends OmniZip's diversity criterion with audio-visual alignment
- `audio_anchor_beta=0.3`: keeps diversity as primary signal, audio as secondary — ablate this
- Temporal pruning (odd frames) is unchanged: prior frame remains the reference
- This is the "audio → video anchor" direction that makes VideoZip truly bidirectional

---

### 4d. `omnizip_videozip()` — `omnizip_units.py` (main entry)

```python
def omnizip_videozip(
    input_embeds, attn_logits, input_ids,
    audio_token_id, video_token_id, num_input_frames,
    merging_ratio_audio=0.5, merging_ratio_v=0.5,
    contextual_ratio=0.05, g=3, audio_anchor_beta=0.3,
):
    # 1. Extract features and indices
    video_indices = torch.nonzero(flat_ids == video_token_id, as_tuple=True)[0]
    audio_indices = torch.nonzero(flat_ids == audio_token_id, as_tuple=True)[0]
    video_feature = flat_embeds[video_indices]
    audio_feature = flat_embeds[audio_indices]
    video_token_per_frame = video_feature.shape[0] // num_input_frames
    
    # 2. Video saliency → per-group retention
    num_groups = num_input_frames // 4   # same grouping as OmniZip
    video_group_retention = omnizip_video_saliency(
        video_feature, video_indices, attn_logits,
        num_input_frames, video_token_per_frame, num_groups,
    )
    
    # 3. Video retention → audio merging ratios (SWAPPED role)
    audio_merging_ratios = _map_retention_to_ratios(
        video_group_retention, merging_ratio_audio,
        min_ratio=0.35, max_ratio=0.75,
    )  # same formula as OmniZip, roles swapped
    
    # 4. Build audio groups (same structure as OmniZip)
    audio_groups = _build_audio_groups(audio_feature.shape[0], num_groups)
    
    # 5. Per-group audio compression with video-derived ratios
    audio_mask, merge_plan = omnizip_audio_compress(
        audio_feature, video_feature, attn_logits,
        audio_groups, audio_merging_ratios, contextual_ratio, g,
    )
    
    # 6. Merge pruned audio into anchors
    _apply_merge_plan(flat_embeds, audio_feature, audio_indices, video_feature, merge_plan)
    
    # 7. Video compression via ISTM with audio-guided anchor selection
    video_mask = _compress_video_istm(
        video_feature, audio_feature, merging_ratio_v,
        num_groups, video_token_per_frame, audio_anchor_beta,
    )
    
    # 8. Build global mask
    global_mask = torch.ones(flat_embeds.size(0), dtype=torch.bool, device=device)
    global_mask[video_indices] = video_mask
    global_mask[audio_indices] = audio_mask
    
    return input_embeds_out, global_mask
```

---

## 5. Config Changes

### `demo.py` additions
```python
parser.add_argument("--videozip", action="store_true")
parser.add_argument("--guide_mode", choices=["audio", "video", "adaptive"], default="audio")
parser.add_argument("--audio_anchor_beta", type=float, default=0.3)
```

### `omnizip_config` dict
```python
omnizip_config = {
    "rho_audio": 0.4,
    "rho_video": 0.7,
    "g": 3,
    "contextual_ratio": 0.05,
    "guide_mode": "video",                 # NEW: audio | video | adaptive
    "audio_anchor_beta": 0.3,              # NEW
    "video_saliency_source": "attn",       # NEW: attn | l6_cached | sim_only
    "l6_cache_dir": "cache/l6_video/",     # NEW: where per-video L6 snapshots live
}
```

---

## 6. Modeling Changes (`modeling_qwen2_5_omni.py`)

In `forward()`, find where `omnizip()` is called (~line 2566) and add a branch:

```python
guide_mode = getattr(self.thinker, 'omnizip_config', {}).get('guide_mode', 'audio')
if guide_mode == 'video':
    inputs_embeds, global_mask = omnizip_videozip(
        input_embeds, attn_logits, input_ids,
        audio_token_id=..., video_token_id=..., num_input_frames=self.thinker.nframes,
        **{k: v for k, v in cfg.items() if k not in ('guide_mode',)},
    )
else:
    inputs_embeds, global_mask = omnizip(...)  # original path
```

---

## 7. Ablation Plan

### Ablation A: Direction of temporal guidance + saliency source
| Config | Audio guide (OmniZip) | Video guide (VideoZip, attn logits) | Video guide (sim-only, no attn) |
|---|---|---|---|
| VideoMME | baseline | expected +? | isolates attn-logit contribution |
| AIR-Bench | baseline | expected -? | isolates attn-logit contribution |
| WorldSense | baseline | expected ≈ | isolates attn-logit contribution |

**Sim-only variant:** Replace `attn_logits.mean(0).sum(0)` step in `omnizip_video_saliency()` with
frozen cross-modal cosine: `vid_importance[i] = max_j cos(v_i, a_j)`. ~30 LOC. Tests whether
attention logits actually carry info beyond raw embedding alignment (Aurelle's hypothesis).

### Ablation B: Audio anchor beta in ISTM
`beta ∈ {0.0, 0.1, 0.3, 0.5, 1.0}` — 0.0 = original dpcknn, higher = more audio-aligned

### Ablation C: Per-group vs global audio compression
- Global (current OmniZip style, one ratio for all): baseline
- Per-group with video-derived ratios (VideoZip): proposed

### Ablation D: Adaptive guide selection
```python
# Choose guide based on relative attention entropy
audio_entropy = -(audio_attn * audio_attn.log()).sum()
video_entropy = -(video_attn * video_attn.log()).sum()
guide = 'audio' if audio_entropy < video_entropy else 'video'
```

### Ablation E: L6-cached video saliency vs live attention
Empirical anchor: L6 of Qwen2.5-Omni Thinker carries question-invariant cross-modal saliency
(cross-question Spearman 0.9992; AUC 0.65 vs OmniZip mask; within-video std 0.0013).
The same property should hold for **video** saliency — read once per video, reuse across queries.

| Saliency source | Per-query cost | Cache | Expected acc | Prefill |
|---|---|---|---|---|
| Live attn logits (default) | full forward | none | baseline | 1.0× |
| L6 hidden states (cached) | one-time per video | per-video | ≈ baseline | ~1.6× |
| L6 hidden states (no cache) | partial forward to L6 | none | ≈ baseline | ~1.3× |

Adds a free 1.6× prefill speedup on top of VideoZip token reduction — a story that neither
OmniSIFT (training) nor OmniZip (audio-side L6 only) can tell.

---

## 8. Hypotheses and When VideoZip Wins

**VideoZip should outperform OmniZip when:**
1. Audio has high temporal redundancy (background music, silence, ambient noise)
2. Visual events are the primary query target ("what does the person DO?")
3. Audio is secondary/confirmatory: speaker on screen (video tells you WHO speaks → prune non-speaker audio)
4. Tasks: VideoMME (visual-primary), MSVD-QA, activitynet-QA

**OmniZip should outperform VideoZip when:**
1. Audio carries unique information not visible in video (off-screen speech, sound events)
2. Video has high temporal redundancy (static scenes, talking heads)
3. Tasks: AIR-Bench audio tasks, speech QA benchmarks

**Adaptive guide selection** (using attention entropy) should win overall.

---

## 9. Paper Section Mapping

| Section | Content |
|---|---|
| Abstract | Training-free, video-guided audio compression; audio anchors video |
| Intro | OmniZip's limitation: assumes audio is always the information-dense modality |
| Related | OmniSIFT (trained, no anchoring), OmniZip (audio-guided) |
| Method §3.1 | Video saliency scoring (frame-aggregated attention) |
| Method §3.2 | video_group_retention → audio_merging_ratios |
| Method §3.3 | Per-group audio compression with video-derived ratios |
| Method §3.4 | Audio-anchored ISTM (the new cross-modal anchor direction) |
| Method §3.5 | Adaptive guide selection via attention entropy |
| Experiments | VideoMME, AIR-Bench, WorldSense — same benchmarks as OmniZip |
| Ablation | A/B/C/D from §7 above |

---

## 10. Implementation Order

1. **`omnizip_video_saliency()`** — standalone, no deps, easy to unit-test
2. **`_map_retention_to_ratios()`** helper (extract from existing OmniZip code, shared)
3. **`omnizip_audio_compress()`** — wraps existing `omnizip_audio_attn()`, low risk
4. **`omnizip_istm_audio_anchored()`** — modify ISTM, ablate `beta`
5. **`omnizip_videozip()`** — wire together steps 1-4
6. **Config + CLI flags** in `demo.py` and `eval/eval.py`
7. **Model dispatch** in `modeling_qwen2_5_omni.py`
8. **`omnizip_video_saliency_l6()`** — read from cached L6 hidden states; reuse cache
   infra from `docs/l6cache.md`. Headline speedup path. (~80 LOC + cache wiring)
9. **`omnizip_video_saliency_simonly()`** — Aurelle's frozen-cross-modal baseline; one
   ablation row, no production path. (~30 LOC)
10. **Saliency-source dispatch** in `omnizip_videozip()`: read `video_saliency_source` config

Total new code: ~400 lines. No changes to existing functions (backward compatible).

---

## 11. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Video saliency aggregation loses spatial detail | Add max-pooling variant: `frame_imp = vid_importance.reshape(T, S).max(1).values` |
| ISTM audio-anchor cost (N_v × N_a sim matrix) | Only use dominant audio tokens (top-k from audio_mask) as anchors |
| OmniSIFT already published this direction | Our novelty: training-free + audio-anchored ISTM + adaptive selection |
| Video-group retention less predictive than audio | Run ablation D; publish as "task-adaptive guide" if adaptive wins |
| Attention logit alignment for video indices | Use `importance[video_indices]` (exact), not truncation — fixes OmniZip's bug too |
