# OmniRefine (reference implementation)

Training-free, alignment-aware **cooperative** audio/video token compression
for Omni-LLMs, a citation-anchored reimplementation of:

> **OmniRefine: Alignment-Aware Cooperative Compression for Efficient
> Omnimodal Large Language Models.** Yuchen Deng, Zidang Cai, Hai-Tao Zheng,
> Jie Wang, Feidiao Yang, Yuxing Han. arXiv:2605.12056v1, 12 May 2026.

Two stages, applied to interleaved audio/video tokens before LLM prefill:

1. **CPCR**: refine native chunk boundaries into cross-modally aligned
   compression units via frame-audio similarity and constrained DP.
2. **MACC**: per refined chunk, cooperatively compress video (quadtree +
   temporal merge) and audio (semantic anchors + fusion) under a cross-modal
   budget.

This is **minimal mode** plus a conservative Qwen2.5-Omni prefill bridge:
the two algorithms remain pure, testable functions, and `torch_runtime.py`
converts already-merged Qwen audio/video prefill embeddings into a compressed
sequence mask. Full internal layer-L KV-cache pruning is still a
model-version-specific adapter task. See `DESIGN.md` and
`REPRODUCTION_NOTES.md`.

## Install

```bash
pip install -r requirements.txt   # numpy only for the core
```

The Qwen prefill bridge also needs `torch`.

## Use (model-free core)

```python
from omnirefine import OmniRefineConfig, ProbeInputs, compress

cfg = OmniRefineConfig(layer_probe=8)        # L is UNSPECIFIED; tune it
inputs = ProbeInputs(...)                    # filled by an adapter at layer L
keep = compress(inputs, cfg)                 # returns KeepMask

keep.video_keep_ids   # retained video token ids
keep.audio_keep_ids   # retained audio anchor ids
keep.video_reps       # id -> merged representation
keep.audio_reps       # anchor id -> fused representation (Eq 11)
keep.chunks           # refined CPCR chunks
```

To run on a real model at an internal layer, subclass
`omnirefine.OmniRefineAdapter` and implement `probe()` (prefill to L, extract
hidden states + saliency + native chunk ids) and `apply()` (prune KV cache,
resume prefill). See `adapter.py`.

## Use (Qwen2.5-Omni prefill bridge)

After Qwen has merged text/audio/video into `inputs_embeds`, call:

```python
from omnirefine import OmniRefineConfig, omnirefine_qwen_prefill

cfg = OmniRefineConfig(layer_probe=0)
inputs_embeds, global_mask, diag = omnirefine_qwen_prefill(
    inputs_embeds,
    input_ids,
    audio_token_id=model.config.audio_token_id,
    video_token_id=model.config.video_token_id,
    num_input_frames=model.thinker.nframes,
    cfg=cfg,
    attn_logits=attn_logits,          # optional fused audio importance source
    video_grid_thw=video_grid_thw,    # preferred for frame grid recovery
    spatial_merge_size=model.thinker.spatial_merge_size,
)

attention_mask = attention_mask[:, global_mask]
position_ids = position_ids[:, :, global_mask]
```

This bridge follows the same integration shape as the local OmniZip helper:
it writes Eq. 11 merged anchor reps back into `inputs_embeds`, returns a
boolean sequence mask, and leaves non-audio/video tokens untouched.

In the local `OmniZip-main/omnizip/modeling_qwen2_5_omni.py` fork, the hook is
wired behind `model.thinker.omnirefine_config`:

```python
model.thinker.omnirefine_config = {
    "rho_a": 0.3,
    "rho_v": 0.6,
    "layer_probe": 0,
}
```

Leave `model.thinker.omnizip_config = None` when using this path.

## Tests

```bash
python -m pytest omnirefine/tests
```

`tests/test_torch_runtime.py` is skipped automatically when `torch` is not
installed.

## Layout

```text
omnirefine/
  config.py          hyperparameters (Sec 4.1, Appendix A)
  cpcr.py            Stage 1: CPCR + Algorithm 1 DP
  video_compress.py  Stage 2a: tree-structured spatio-temporal (Compress_v)
  audio_compress.py  Stage 2b: semantic-anchor audio (Compress_a)
  budget.py          cross-modal budget (Eq 13/14)
  pipeline.py        compress(ProbeInputs) -> KeepMask (pure core)
  torch_runtime.py   Qwen prefill bridge: tensors -> ProbeInputs -> mask
  adapter.py         internal layer-L adapter boundary (stubs)
tests/               invariant tests on synthetic input
notebooks/           walkthrough.ipynb (paper to code)
DESIGN.md            data flow + probe-layer boundary
REPRODUCTION_NOTES.md  SPECIFIED / PARTIAL / INVENTED audit
```

## Status & honesty

- Both algorithms are implemented and unit-verified against the paper's
  structural invariants.
- Prefill-level Qwen bridge implemented (`layer_probe=0`): produces an
  `inputs_embeds` slice mask plus diagnostics, with identity fallback for
  too-short sequences.
- **Probe layer L is unspecified by the paper**; `cfg.layer_probe` remains the
  load-bearing knob for a full reproduction. See REPRODUCTION_NOTES #1.
- Internal layer-L application still needs model-specific KV-cache pruning.
- Algorithm 1 (appendix) and Eq 5 (main text) disagree; this implementation
  follows the appendix and defines the undefined `ChunkVariance`. See #2/#3.
- Reported accuracy numbers are not reproduced here; that needs the model,
  benchmarks, and the runtime hook wired into evaluation.
