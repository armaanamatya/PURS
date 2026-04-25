# Qwen2.5-Omni Benchmark Matrix

Started output root: `/data/armaan/purs/runs/qwen25_matrix_gpu7_all7_snapkv`
Methods: mixkv
Temperatures: 0.1, 0.9
Repeats per method-temperature: 10
Total eval executions: 1 methods x 2 temperatures x 10 repeats = 20
GPUs used: 7

## Memory Fields

- `model_loaded_alloc_gb`: `torch.cuda.memory_allocated()` right after model load. This is closest to model weights plus persistent buffers, but it excludes CUDA context and non-PyTorch allocations.
- `peak_alloc_gb`: per-question PyTorch allocated peak after resetting peak stats just before inference. This includes loaded model memory plus inference-time allocations such as activations and KV cache.
- `inference_extra_peak_alloc_gb`: `peak_alloc_gb - before_alloc_gb`, an approximate per-question incremental inference overhead.
- `peak_reserved_gb`: PyTorch caching allocator reservation, which can be higher than allocated memory.
- `gpu_process_peak_gb`: peak process VRAM from `nvidia-smi`, sampled by the harness. This is the best top-line VRAM number because it includes CUDA context, allocator reservation, and non-PyTorch allocations.
- `gpu_total_peak_gb`: total memory used on that GPU during the run. Use this carefully if other processes share the GPU.

## Frame Fields

- `orig_frames_*`: mean/max decoded frame count per question after the normal dataset caps (`--fps`, `--max_frames_videomme`, `--max_frames_other`) are applied.
- `used_frames_*`: mean/max frame count actually kept for inference. For baseline, GPTQ, AWQ, OmniZip, and SnapKV MixKV this equals `orig_frames`; for DivPrune/ReDiPrune it reflects the pruned frame count.
- `frame_keep_ratio_mean`: average `used_frames / orig_frames` across questions in a run.

## Summary

| Method | Temp | Successful Runs | Accuracy Mean | Accuracy Std | Prefill Mean ms | Frames Mean | Keep Ratio | Peak Alloc GB | Process Peak GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mixkv | 0.1 | 10/10 | 0.2 | 0.01072 | 2304 | 110 | 1 | 19.69 | 24.11 |
| mixkv | 0.9 | 10/10 | 0.2203 | 0.0339 | 2301 | 110 | 1 | 19.69 | 24.11 |

