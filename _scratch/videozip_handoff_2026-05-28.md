# VideoZip Handoff — 2026-05-28

## Current State

- Local branch: `videozip`
- Workspace: `C:\Users\Armaan\Desktop\PURS`
- Remote GPU workspace: `/data/armaan/purs`
- Remote host: `armaan@10.244.120.178`
- Remote env: `omnizip_clean`
- Target GPU command pattern: `CUDA_VISIBLE_DEVICES=7 ... --device cuda:0`

VideoZip is now implemented enough to run the eval harness. It is not benchmarked yet; the first full eval is still pending.

## Files Changed This Session

Tracked/important local changes:

- `docs/videozip_plan.md`
  - Reframed novelty against OmniDrop / OmniRefine / OmniSelect.
  - Replaced "L6 hidden states / L2 norm" with cached 1D L6 Q*K video scores.
  - Updated literature gap, ablations, risks, and paper section mapping.

- `videozip/src/videozip.py`
  - Main `omnizip_videozip()` path implemented.
  - Added `_project_importance_to_audio_scores(attn_logits, audio_indices)`.
  - Uses projected audio-local scores for per-group audio compression.
  - Uses `audio_kept_features = flat_embeds[audio_indices][audio_mask]` for audio-anchored ISTM.
  - Handles non-divisible frame counts instead of falling back to OmniZip.
  - Adds tail-safe ISTM helper for incomplete frame chunks.

- `videozip/src/audio_compress.py`
  - Added projection helper for `[H,T,T]`, `[T,T]`, and 1D attention/score inputs.
  - Per-group audio compression now passes `group_scores` to `omnizip_audio_attn()`.

- `videozip/src/istm_audio_anchored.py`
  - Used by the orchestrator; `beta=0` should match OmniZip ISTM behavior.

- `videozip/tests/test_video_saliency.py`
  - Added tests for:
    - full `[H,T,T]` attention projection,
    - 1D cached scores,
    - non-divisible frame counts,
    - `audio_anchor_beta=0` equals OmniZip ISTM.

- `videozip/eval/eval_videozip.py`
  - Eval harness for cached L6 VideoZip.
  - Added compatibility shims for older remote `eval_qwen_omni_zip.py`:
    - missing OmniZip defaults,
    - `set_run_seed`,
    - VRAM helpers,
    - `Tee` / `StderrTee`,
    - `check_video_has_audio`,
    - `resolve_video_path`.

- `docs/videozip_token_compression_diagram.svg`
  - Standalone explainer diagram of VideoZip token compression.

Also edited on disk but not tracked in this repo because the vendored tree is untracked:

- `OmniZip-main/omnizip/videozip_units.py`
  - Mirrored the attention projection and retained-audio anchoring fixes for the older integrated demo path.

## Verification Done

Local Windows:

```powershell
python -m py_compile videozip\src\videozip.py videozip\src\audio_compress.py videozip\src\video_saliency.py videozip\src\istm_audio_anchored.py videozip\tests\test_video_saliency.py videozip\eval\eval_videozip.py
git diff --check -- docs/videozip_plan.md videozip/src/videozip.py videozip/src/audio_compress.py videozip/tests/test_video_saliency.py videozip/eval/eval_videozip.py
```

Status:

- `py_compile` passed.
- `git diff --check` passed with only CRLF warnings.
- Local runtime tests could not run because local Python did not have `torch`.

Remote GPU:

```bash
cd /data/armaan/purs
PYTHONPATH=/data/armaan/purs python -m videozip.tests.test_video_saliency
```

Result:

```text
all smoke tests passed
```

## Remote Run Trail

Initial remote eval command worked through cache loading and model-load setup, but hit version/path issues.

Important fixes already made locally:

1. Running test by path failed with `ModuleNotFoundError: No module named 'videozip'`.
   - Fixed by running with:
     ```bash
     PYTHONPATH=/data/armaan/purs python -m videozip.tests.test_video_saliency
     ```

2. First eval used wrong model path:
   - Bad: `/data/armaan/purs/Qwen2.5-Omni`
   - This is the source repo, not HF weights.
   - Correct: `/data/armaan/models/Qwen2.5-Omni-7B`

3. Remote `eval_qwen_omni_zip.py` was older and missed helper APIs.
   - Fixed locally in `videozip/eval/eval_videozip.py` with compatibility shims.

Latest remote failure before final local patch:

```text
AttributeError: module 'eval_qwen_omni_zip' has no attribute 'check_video_has_audio'
```

A local patch has now added fallback `check_video_has_audio()` and `resolve_video_path()` to `videozip/eval/eval_videozip.py`. This patched file still needs to be synced to the remote before retrying.

Update after retry:

- Model loaded successfully.
- The run produced `0/118`, but this is invalid: every row failed before inference with:
  ```text
  TypeError("run_inference() got an unexpected keyword argument 'measure_prefill'")
  ```
- Local `videozip/eval/eval_videozip.py` has now been patched with `run_inference_compat()`.
  It inspects `base.run_inference` and passes `measure_prefill` only if the older/newer
  remote signature supports it. If the old function returns only 4 values, timing is `{}`.
- Before rerunning, delete the invalid remote output directory.

## Next Exact Step

From local Windows, sync the latest eval wrapper:

```powershell
scp C:\Users\Armaan\Desktop\PURS\videozip\eval\eval_videozip.py armaan@10.244.120.178:/data/armaan/purs/videozip/eval/eval_videozip.py
```

Then on remote:

```bash
cd /data/armaan/purs
rm -rf videozip/runs/l6_t0p1
mkdir -p videozip/runs/l6_t0p1

PYTHONPATH=/data/armaan/purs CUDA_VISIBLE_DEVICES=7 python videozip/eval/eval_videozip.py \
  --cache vizzing/layer_depth_all_full.jsonl \
  --layer 6 \
  --metadata videos/metadata.json \
  --videos videos \
  --output videozip/runs/l6_t0p1/results.jsonl \
  --log videozip/runs/l6_t0p1/eval.log \
  --vram_log videozip/runs/l6_t0p1/vram.jsonl \
  --errors_log videozip/runs/l6_t0p1/errors.log \
  --model /data/armaan/models/Qwen2.5-Omni-7B \
  --device cuda:0 \
  --dtype bfloat16 \
  --temperature 0.1 \
  --seed 1 \
  --rho_audio 0.3 \
  --rho_video 0.6 \
  --g 3 \
  --contextual_ratio 0.05 \
  --audio_anchor_beta 0.3 \
  --measure_prefill
```

Watch:

```bash
tail -f videozip/runs/l6_t0p1/eval.log
```

## Expected Output Files

Under remote:

```text
/data/armaan/purs/videozip/runs/l6_t0p1/
  results.jsonl
  eval.log
  vram.jsonl
  errors.log
```

## Notes For Resuming

- Always run eval from `/data/armaan/purs`, not `/data/armaan`.
- Prefix remote commands with `PYTHONPATH=/data/armaan/purs`.
- With `CUDA_VISIBLE_DEVICES=7`, pass `--device cuda:0`.
- `vizzing/layer_depth_all_full.jsonl` has 44 cached videos and loaded successfully.
- Model loaded successfully from `/data/armaan/models/Qwen2.5-Omni-7B`, with about `16.6 GB allocated / 20.9 GB reserved` before the last helper mismatch.
- Remote smoke tests already passed, so remaining issues are likely eval-wrapper integration, not core tensor logic.
