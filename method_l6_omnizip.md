# Method: L6 Thinker Saliency as OmniZip Audio Importance

## What OmniZip normally does

OmniZip compresses audio and video tokens **before** they enter the Thinker (the LLM decoder). For audio, it computes an importance score for each audio token using **intra-audio encoder self-attention**: the audio encoder's own attention over the audio sequence, averaged across heads and layers. High-scoring tokens are kept; low-scoring tokens are dropped. With `rho_audio=0.3`, 30% of audio tokens are dropped, keeping 70%.

This importance signal is entirely internal to the audio encoder. It has no knowledge of the question or the language model's preferences — it reflects which audio segments are internally salient within the audio stream alone.

## The one thing we change

We replace OmniZip's audio importance signal with **precomputed Layer-6 Thinker Q·K cross-modal scores**.

The integration point is a single argument: `attn_logits` in `omnizip_units.omnizip()`. This is the per-token importance vector that OmniZip uses to rank and prune audio tokens. We monkey-patch this function to swap in our cached scores before the pruning decision is made (`eval_qwen_omni_zip_cached.py`, lines 126–156):

```
wrapped_omnizip():
    external_scores = controller.get_current_scores()   # our L6 scores
    replacement = torch.as_tensor(external_scores, ...)
    return original_omnizip(..., attn_logits=replacement, ...)  # OmniZip runs normally
```

Everything else in OmniZip is unchanged:
- Pruning ratio: `rho_audio=0.3` (keep 70%), `rho_video=0.6` (keep 30%)
- Video token compression (uses OmniZip's own video importance, unmodified)
- Token merging and contextual ratio logic
- The Thinker model, generation, and decoding

## What the L6 scores are

For each video, we compute a single importance vector over all audio tokens using:

```
score[i] = mean over heads and text positions of:
    q_proj(hs_text) @ k_proj(hs_audio[i]).T / sqrt(head_dim)
```

where `hs` are hidden states at **Layer 6** of the Thinker, extracted via a forward hook on `model.thinker.model.layers[6].self_attn`. This measures how strongly text positions attend to each audio token at this layer depth — a cross-modal relevance signal from within the LLM itself.

Key properties of this signal:
- **Question-invariant**: cross-question Spearman ≥ 0.999 at Layer 6 (rank ordering barely changes across different questions on the same video)
- **Cacheable**: computed once per video with a neutral dummy query ("Describe this video."), reused for all questions
- **Aligned with OmniZip**: ROC AUC vs OmniZip's actual keep mask = **0.653** at Layer 6 vs **0.525** at Layer 14

## Why the stock model for precompute

The precompute (`precompute_l6_saliency.py`) uses the **stock Qwen2.5-Omni** model, not the OmniZip-modified version. This matters because OmniZip drops ~30% of audio tokens before the Thinker layers run. If the hook fired during an OmniZip forward pass, it would only see the surviving 70% of tokens — making the scores incomparable to OmniZip's full-sequence keep mask. The stock model keeps all audio tokens intact so the hook at Layer 6 scores the complete audio sequence.

## Full pipeline

```
Offline (once per video):
  stock Qwen2.5-Omni
    → forward pass with dummy query
    → Layer-6 hook extracts Q·K cross-modal scores over ALL audio tokens
    → save to layer_depth_all_full.jsonl

At benchmark time (per question):
  OmniZip-modified Qwen2.5-Omni
    → load cached L6 scores for this video (mean-reduced across dummy queries)
    → patch omnizip_units.omnizip to use L6 scores as attn_logits
    → OmniZip prunes audio tokens using L6 ranking instead of encoder self-attention
    → Thinker runs on compressed tokens, generates answer
```

## What is and is not being claimed

**We are claiming**: Layer-6 Thinker cross-modal saliency is a valid replacement signal for OmniZip's audio importance. The LLM itself, at an early layer, already encodes a ranking of audio tokens that aligns with OmniZip's pruning decisions (AUC 0.653). This signal can be computed offline without the question, which enables true caching.

**We are not claiming**: that our method improves accuracy over OmniZip. The benchmark result (Step 2) tests whether accuracy is preserved when the signal is swapped. Any drop is expected to be small because the AUC alignment is moderate, not perfect.

**Layer 14 comparison**: Layer 14 has similar Gini concentration (0.341 vs 0.327 at L6) but AUC of only 0.525 — near random. It represents something structurally different from what OmniZip selects. Layer 6 is the correct candidate.
