# OmniRefine Implementation Log

Date: 2026-05-29

This records the completion pass performed after the initial `paper2code`
reproduction of OmniRefine (arXiv:2605.12056v1).

## Starting state

The generated `omnirefine/` package already contained the model-free NumPy
core:

- `cpcr.py`: Correspondence-Preserving Chunk Refinement (CPCR).
- `video_compress.py`: tree-structured spatio-temporal video compression.
- `audio_compress.py`: semantic-anchor audio compression.
- `budget.py`: cross-modal audio budget coupling from observed video retention.
- `pipeline.py`: pure `compress(ProbeInputs, OmniRefineConfig) -> KeepMask`.
- Synthetic invariant tests for CPCR, MACC, and the full pipeline.

The main missing piece was a runnable Qwen2.5-Omni integration boundary.
`adapter.py` documented the boundary but still raised `NotImplementedError`.

## Added PyTorch prefill bridge

Added `omnirefine/torch_runtime.py`.

The bridge is intentionally conservative and prefill-level only:

- accepts already-merged Qwen-style `inputs_embeds` and `input_ids`;
- extracts video/audio token positions from `video_token_id` and `audio_token_id`;
- reconstructs frame-wise video grids from `video_grid_thw` plus
  `spatial_merge_size`, or from explicit `video_grid_hw`;
- builds `ProbeInputs` for the existing NumPy CPCR+MACC implementation;
- runs `compress(...)`;
- scatters Eq. 11 merged audio/video anchor representations back into
  `inputs_embeds`;
- returns `compressed_embeds`, a boolean `global_mask`, and
  `TorchOmniRefineDiagnostics`.

The bridge returns an identity mask, with diagnostics, when the sequence is too
short, missing a modality, or infeasible under the paper's CPCR bounds.

Files added for this bridge:

- `omnirefine/omnirefine/torch_runtime.py`
- `omnirefine/tests/test_torch_runtime.py`

## Exported runtime API

Updated `omnirefine/__init__.py` to export:

- `TorchOmniRefineDiagnostics`
- `build_probe_inputs_from_qwen_prefill`
- `apply_keep_to_prefill`
- `omnirefine_qwen_prefill`

## Added Qwen hook

Updated `OmniZip-main/omnizip/modeling_qwen2_5_omni.py` with an opt-in
OmniRefine path:

```python
model.thinker.omnirefine_config = {
    "rho_a": 0.3,
    "rho_v": 0.6,
    "layer_probe": 0,
}
model.thinker.omnizip_config = None
```

The hook:

- lives beside the existing OmniZip inference branch;
- runs before OmniZip and uses `elif` for OmniZip, preventing double
  compression;
- accepts either an `OmniRefineConfig` instance or a dict;
- maps compatibility keys `rho_audio -> rho_a` and `rho_video -> rho_v`;
- saves diagnostics in `model.thinker.omnirefine_last_diag`;
- slices `attention_mask`, `position_ids`, and `cache_position` with the
  returned `global_mask`;
- uses the current `omnidrop_input_ids` if a pre-prune step already ran.

File changed for the hook:

- `OmniZip-main/omnizip/modeling_qwen2_5_omni.py`

## Added tests

Added `omnirefine/tests/test_torch_runtime.py`.

The test builds synthetic Qwen-style tensors and checks that:

- the prefill bridge returns a compressed sequence and matching mask;
- text tokens remain kept;
- diagnostics report fewer kept audio/video tokens;
- too-short sequences return an identity mask.

The test uses `pytest.importorskip("torch")`, so it is skipped in environments
without PyTorch.

## Updated docs

Updated `README.md`:

- documents the prefill bridge;
- shows direct use of `omnirefine_qwen_prefill`;
- shows the local Qwen hook via `model.thinker.omnirefine_config`;
- clarifies that internal layer-L KV-cache pruning is still not implemented.

Updated `DESIGN.md`:

- added `torch_runtime.py` to the module table;
- documented the prefill-level design choice and why it avoids pretending
  internal decoder-layer cache pruning is solved.

Updated `REPRODUCTION_NOTES.md`:

- added item `#10 - Runtime bridge scope`;
- explicitly documents batch-size, chunk-id, grid, saliency, and fallback
  assumptions.

Updated `requirements.txt`:

- clarified that `torch` is optional for `omnirefine.torch_runtime`;
- kept the model-free core NumPy-only at import time.

Files updated for docs and package surface:

- `omnirefine/omnirefine/__init__.py`
- `omnirefine/README.md`
- `omnirefine/DESIGN.md`
- `omnirefine/REPRODUCTION_NOTES.md`
- `omnirefine/requirements.txt`

## Verification performed

Ran:

```bash
python -m pytest omnirefine\tests
```

Result:

```text
7 passed, 1 skipped
```

The skipped test is the torch runtime bridge test because `torch` is not
installed in the current environment.

Attempted `python -m py_compile`, but the local sandbox failed to spawn that
command. The available pytest suite still passed.

## Current limitations

- `layer_probe=0` prefill compression is implemented.
- True internal `layer_probe=L` compression is not implemented because it
  requires model-version-specific K/V cache pruning for layers already run.
- Batched compression is not implemented; the bridge supports batch size 1.
- If native Qwen temporal chunk ids are not supplied, the bridge uses equal-rank
  chunk ids as an explicit fallback.
- If frame grid metadata is not supplied, the bridge infers a square-ish grid;
  explicit `video_grid_thw` or `video_grid_hw` is preferred.
- If attention logits are not supplied, audio saliency is uniform.
- Reported paper accuracy is not reproduced yet; this still requires a real
  Qwen2.5-Omni run and benchmark integration.

## Next validation step

Use an environment with PyTorch and Qwen2.5-Omni dependencies, then run:

```bash
python -m pytest omnirefine\tests\test_torch_runtime.py
```

After that, run one short Qwen2.5-Omni sample with
`model.thinker.omnirefine_config` enabled and inspect:

```python
model.thinker.omnirefine_last_diag
```
