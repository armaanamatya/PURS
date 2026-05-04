# Qwen2.5-Omni + MixKV / DivPrune Implementation Notes

This note explains what was done to create `eval_qwen_omni_mixkv.py` and `eval_qwen_omni_divprune.py`, how they relate to `eval_qwen_omni.py`, what was taken from the MixKV and DivPrune papers, and which defaults were copied versus adapted.

## Scope

- `eval_qwen_omni.py` was used as the baseline template.
- The baseline file itself was not extended in-place for these methods.
- Instead, two sibling evaluation scripts were created:
  - `eval_qwen_omni_mixkv.py`
  - `eval_qwen_omni_divprune.py`
- This kept the plain Qwen2.5-Omni path unchanged while making the compression/pruning logic easy to run and compare.

## Baseline Structure Reused From `eval_qwen_omni.py`

Both new scripts keep the same high-level evaluation flow as the baseline:

- Qwen2.5-Omni loading via `Qwen2_5OmniForConditionalGeneration` and `Qwen2_5OmniProcessor`
- prompt building by dataset
- `process_mm_info(...)` preprocessing for video/audio inputs
- text-only generation mode through `return_audio=False`
- metadata iteration over entries and questions
- per-question JSONL result writing
- VRAM/error logging
- video path resolution and optional audio detection

The baseline pieces mainly come from:

- `eval_qwen_omni.py:210-347` for model loading and inference
- `eval_qwen_omni.py:351-373` for video lookup
- `eval_qwen_omni.py:377-560` for CLI, dataset loop, result logging, and summaries

The inherited defaults that were kept in both new scripts are:

- `--fps 2.0`
- `--max_pixels 360*420`
- `--max_new_tokens 4096`
- `--dtype bfloat16`
- text-only generation with `return_audio=False`
- optional input audio unless `--no_audio` is passed
- `attn_implementation="sdpa"` when loading Qwen2.5-Omni

## What Was Added For MixKV

### New logic

`eval_qwen_omni_mixkv.py` adds four MixKV-specific components on top of the baseline:

1. `load_head_similarity_scores(...)`
   - Loads the per-layer, per-head similarity JSON expected by MixKV-style head-wise selection.

2. `MixKVCompressor`
   - Implements training-free KV compression during prefill.
   - Computes:
     - attention importance from the last `window_size` queries
     - key diversity from negated cosine similarity to the mean key
     - value norm from L2 norms of value states
   - Keeps a recent window untouched and selects top historical tokens for the rest.

3. `_make_mixkv_sdpa_forward(...)`
   - Monkeypatches the Qwen2.5-Omni SDPA attention forward pass.
   - During prefill (`q_len > 1`), it compresses the KV states before they are stored in cache.
   - During decode (`q_len == 1`), it appends to cache normally.

4. `apply_mixkv_to_model(...)`
   - Walks the thinker decoder layers and attaches one compressor per layer.

The core code for this is in:

- `eval_qwen_omni_mixkv.py:224-250`
- `eval_qwen_omni_mixkv.py:253-399`
- `eval_qwen_omni_mixkv.py:404-536`
- `eval_qwen_omni_mixkv.py:541-560`

### What this means relative to the baseline

Compared with `eval_qwen_omni.py`, the MixKV script changes the model path, not the dataset loop:

- model loading now immediately patches attention layers after `from_pretrained(...)`
- generation/inference still uses the same metadata-driven evaluation loop
- output rows add method metadata:
  - `method`
  - `budget`
- log/output filenames are renamed to MixKV-specific defaults

### What was implemented from the MixKV paper/repo

The implemented pieces that directly reflect MixKV or its repo are:

- head-wise redundancy awareness
- combining importance with diversity instead of importance-only scoring
- a preserved recent token window
- selection among earlier KV tokens only
- a head-wise mixture rule for `headwisemixkv`
- support for SnapKV-like attention-only selection as a fallback/default

More concretely, the active `headwisemixkv` rule in the script is:

- `head_score * sim + (1 - head_score) * (attn + vnorm)`

That follows the MixKV repo logic in `MixKV-main/mixkv/mixkv_utils.py:352-389`, especially:

- attention pooling over the last window
- similarity normalization and scaling
- value-norm scaling
- head-wise mixing with precomputed `head_score`

The local MixKV repo also shows:

- internal SnapKVCluster defaults of `window_size=64`, `max_capacity_prompt=256+64`, `kernel_size=5`
- Qwen calibration script defaults of `method="snapkv"` and `budget=128`

See:

- `MixKV-main/mixkv/mixkv_utils.py:212-390`
- `MixKV-main/distribution_qwen.py:12-15`

### What was adapted instead of copied exactly

The MixKV integration is an adaptation for Qwen2.5-Omni, not a literal port of the MixKV repo:

- It targets Hugging Face Qwen2.5-Omni SDPA attention, not the original MixKV FlashAttention-only integration.
- It patches `Qwen2.5-Omni` at runtime instead of modifying a model fork inside `MixKV-main`.
- It keeps the current prefill forward computation on the full just-computed KV, while storing the compressed KV for future decode steps.
- It does not implement flattened cache storage or the repo's lower-level CUDA path.
- It does not generate Qwen2.5-Omni head scores automatically; it only consumes them if `--head_score_path` is provided.
- It does not implement MixKV's pyramidal/adaptive capacity logic.

## MixKV Defaults Used Here

### Defaults inherited from the baseline

- `--fps 2.0`
- `--max_pixels 360*420`
- `--max_new_tokens 4096`
- `--dtype bfloat16`

### Defaults chosen for the new MixKV script

- `--budget 256`
- `--window_size 32`
- `--select_method snapkv`
- `--head_score_path None`
- internal `kernel_size=5`

### Why these defaults were chosen

- `select_method="snapkv"` was chosen because it is the safest zero-calibration default and matches the MixKV repo's Qwen calibration script default (`distribution_qwen.py:13`).
- `kernel_size=5` follows the MixKV repo default (`mixkv_utils.py:213`).
- `budget=256` was chosen as a conservative starting point for Qwen2.5-Omni video evaluation. This is not the repo's calibration-script default of `128`; it is a less aggressive compression setting.
- `window_size=32` was chosen to reserve more of the fixed budget for selected historical KV tokens. This is also an adaptation; the repo's `SnapKVCluster` default is `64`.
- `head_score_path=None` is the default because there is no ready-made Qwen2.5-Omni calibration file in this workspace. Head-wise MixKV is therefore opt-in.

## What Was Added For DivPrune

### New logic

`eval_qwen_omni_divprune.py` adds three main DivPrune-specific pieces:

1. `pairwise_cosine_distance(...)`
   - Computes `1 - cosine_similarity`.

2. `divprune_select(...)`
   - Implements the greedy max-min diversity selection procedure.
   - First pick: the token whose nearest neighbor is farthest away.
   - Later picks: the token farthest from the already selected set.

3. Two application modes
   - `frame` mode:
     - prune decoded video frames before sending them back into the processor
   - `token` mode:
     - prune `pixel_values_videos` after processor output as an approximation to true token pruning

The core code is in:

- `eval_qwen_omni_divprune.py:210-258`
- `eval_qwen_omni_divprune.py:261-306`
- `eval_qwen_omni_divprune.py:388-483`
- `eval_qwen_omni_divprune.py:488-652`

### What this means relative to the baseline

Compared with `eval_qwen_omni.py`, the DivPrune script changes preprocessing, not generation:

- model loading remains basically the same as baseline
- pruning happens before `model.generate(...)`
- result rows add:
  - `method`
  - `subset_ratio`
  - `orig_frames`
  - `pruned_frames`
- summary output also reports total frame reduction

### What was implemented from the DivPrune paper/repo

The parts that were implemented directly from DivPrune are:

- cosine-distance-based diversity scoring
- max-min subset selection
- greedy farthest-point style construction

This matches both:

- the paper's MMDP formulation and algorithm description
- the released LLaVA implementation

In the local DivPrune repo, the closest matching code is:

- `divprune-main/LLaVA/llava/model/llava_arch.py:152-171`

That code:

- computes `1 - cosine_similarity`
- chooses the first token using the second-smallest distance
- then repeatedly selects the token maximizing minimum distance to the selected set

The paper describes the same idea in Section 3.3:

- token pruning is reformulated as a Max-Min Diversity Problem
- cosine distance is used
- a distance matrix is computed once
- greedy subset construction is then applied

### What was adapted instead of copied exactly

The DivPrune integration is more approximate than the MixKV one, because Qwen2.5-Omni does not expose the same token insertion path as the paper's LLaVA-based implementation.

There are two important adaptations:

1. `frame` mode is not the paper's original operating point.
   - It prunes whole decoded frames before they become Omni visual inputs.
   - This is simpler and safer for Qwen2.5-Omni.
   - It is conceptually consistent with diversity pruning, but it is coarser than token-level pruning.

2. `token` mode is still an approximation.
   - It prunes `pixel_values_videos` after processor output.
   - It does not prune true post-encoder visual embeddings inside the Qwen2.5-Omni thinker.
   - `video_grid_thw` is updated proportionally, which is heuristic.

One extra note:

- `_make_divprune_vision_hook(...)` exists in the file, but `load_model(...)` does not register it.
- So the active token-level path today is the `pixel_values_videos` pruning inside `run_inference(...)`, not a live post-encoder hook.

## DivPrune Defaults Used Here

### Defaults inherited from the baseline

- `--fps 2.0`
- `--max_pixels 360*420`
- `--max_new_tokens 4096`
- `--dtype bfloat16`

### Defaults chosen for the new DivPrune script

- `--subset_ratio 0.5`
- `--prune_mode frame`

### Why these defaults were chosen

- The DivPrune repo and paper default to a retained ratio of `0.098` for LLaVA runs.
  - See `divprune-main/README.md:31-33`
  - See `divprune-main/run_Divprune.sh:11`
- That default was not copied directly because Qwen2.5-Omni evaluation here already samples videos at `fps=2.0`, so the visual stream is much smaller than a dense image-token setting.
- Keeping only about 10% of frames from that already reduced stream would be too aggressive as a first default.
- `subset_ratio=0.5` was chosen as a conservative starting point.
- `prune_mode="frame"` was chosen as the default because it does not require fragile model-internal surgery and is easier to validate.

## Practical Summary Of What Was Actually Implemented

### MixKV script

Implemented:

- baseline Qwen2.5-Omni eval flow
- per-head KV scoring
- attention/diversity/value-norm scoring
- recent-window preservation
- optional head-score-driven mixing
- runtime monkeypatching of SDPA attention

Not implemented:

- native MixKV FlashAttention path
- flattened cache CUDA storage
- automatic Omni head-score calibration
- adaptive/pyramidal capacity schedule

### DivPrune script

Implemented:

- baseline Qwen2.5-Omni eval flow
- greedy max-min cosine-distance selection
- frame-level diversity pruning
- approximate token-level pruning on `pixel_values_videos`
- frame reduction logging in outputs

Not implemented exactly as in the paper:

- direct pruning of Qwen2.5-Omni post-encoder visual token embeddings
- the exact LLaVA insertion point used by the released DivPrune code

## External References Used

- MixKV paper: [arXiv 2510.20707](https://arxiv.org/abs/2510.20707)
- DivPrune paper: [arXiv 2503.02175](https://arxiv.org/abs/2503.02175)
- Qwen2.5-Omni docs: [Hugging Face Transformers docs](https://huggingface.co/docs/transformers/v4.52.1/model_doc/qwen2_5_omni)

## Note On Context7 MCP

The original request asked for Context7 MCP docs as well, but there is no configured Context7 MCP server in this session, so the documentation above is grounded in:

- the local code in this workspace
- the local MixKV and DivPrune repos
- the paper pages
- the official Hugging Face Qwen2.5-Omni docs
