# OmniZip Evaluation Reference

## eval.sh template

```bash
export HF_HOME="YOUR_HF_HOME"

# Set WRAPPER=None to disable OmniZip and run baseline
export WRAPPER=OmniZip
OMNIZIP_RHO_AUDIO=0.3
OMNIZIP_RHO_VIDEO=0.6
OMNIZIP_G=3
OMNIZIP_CONTEXTUAL_RATIO=0.05

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --main_process_port=12347 \
    -m lmms_eval \
    --model qwen2_5_omni \
    --model_args "pretrained=Qwen/Qwen2.5-Omni-7B,attn_implementation=flash_attention_2,max_num_frames=768,OMNIZIP_RHO_AUDIO=${OMNIZIP_RHO_AUDIO},OMNIZIP_RHO_VIDEO=${OMNIZIP_RHO_VIDEO},OMNIZIP_G=${OMNIZIP_G},OMNIZIP_CONTEXTUAL_RATIO=${OMNIZIP_CONTEXTUAL_RATIO}" \
    --tasks videomme \
    --batch_size 1 \
    --output_path ./logs/
```

## Disabling OmniZip (baseline comparison)

```bash
export WRAPPER=None
# rest of the eval.sh command unchanged
```

## Changing benchmark task

Replace `--tasks videomme` with any lmms-eval task name:

| Benchmark | `--tasks` value |
|---|---|
| VideoMME | `videomme` |
| MVBench | `mvbench` |
| EgoSchema | `egoschema` |
| ActivityNet-QA | `activitynetqa` |
| PerceptionTest | `perceptiontest` |

Full task list: https://github.com/EvolvingLMMs-Lab/lmms-eval

## Multi-GPU evaluation

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes=2 \
    --main_process_port=12347 \
    -m lmms_eval \
    ...
```

## Output

Results saved to `./logs/` as JSON. Check `results.json` inside the timestamped subfolder.

## Notes

- `batch_size 1` is required for video benchmarks (OOM risk with larger batches)
- `max_num_frames=768` matches paper settings; lower to 256 for faster eval with less VRAM
- OmniZip parameters are passed via `model_args` string, not environment variables in lmms-eval mode
- The `WRAPPER` env var toggles the OmniZip monkey-patch on the model class
