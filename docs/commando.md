# Qwen2.5-Omni Benchmark Commands

## 1. Sync updated scripts from Windows to the server

Run these from `C:\Users\Armaan\Desktop\PURS` in PowerShell:

```powershell
scp eval_qwen_omni.py eval_qwen_omni_zip.py eval_qwen_omni_divprune.py eval_qwen_omni_mixkv.py eval_qwen_omni_rediprune.py "armaan@10.244.120.178:/data/armaan/purs/"
```

```powershell
scp scripts/run_qwen_omni_benchmark_matrix.py "armaan@10.244.120.178:/data/armaan/purs/scripts/"
```

## 2. SSH into the server

```bash
ssh armaan@10.244.120.178
```

## 3. Optional sanity check on the remote machine

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python -m py_compile \
  eval_qwen_omni.py \
  eval_qwen_omni_zip.py \
  eval_qwen_omni_divprune.py \
  eval_qwen_omni_mixkv.py \
  eval_qwen_omni_rediprune.py \
  scripts/run_qwen_omni_benchmark_matrix.py
```

## 4. Check GPU status

```bash
nvidia-smi
```

## 5. Full 7-method dry run on GPU 7

This should show `140` executions total.

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python scripts/run_qwen_omni_benchmark_matrix.py \
  --gpus 7 \
  --output_root runs/qwen25_matrix_gpu7_all7_snapkv \
  --methods baseline,gptq,awq,omnizip,divprune,mixkv,rediprune \
  --temperatures 0.1,0.9 \
  --runs 10 \
  --fps 2.0 \
  --max_pixels 100352 \
  --max_frames_videomme 768 \
  --max_frames_other 128 \
  --max_new_tokens 256 \
  --measure_prefill \
  --qwen_model /data/armaan/models/Qwen2.5-Omni-7B \
  --gptq_model /data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4 \
  --awq_model /data/armaan/models/Qwen2.5-Omni-7B-AWQ \
  --mixkv_select_method snapkv \
  --dry_run
```

## 6. Baseline-first dry run

This is `20` executions total: `2 temperatures x 10 repeats`.

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python scripts/run_qwen_omni_benchmark_matrix.py \
  --gpus 7 \
  --output_root runs/qwen25_matrix_gpu7_all7_snapkv \
  --methods baseline \
  --temperatures 0.1,0.9 \
  --runs 10 \
  --fps 2.0 \
  --max_pixels 100352 \
  --max_frames_videomme 768 \
  --max_frames_other 128 \
  --max_new_tokens 256 \
  --measure_prefill \
  --qwen_model /data/armaan/models/Qwen2.5-Omni-7B \
  --dry_run
```

## 7. Baseline-first real run

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python scripts/run_qwen_omni_benchmark_matrix.py \
  --gpus 7 \
  --output_root runs/qwen25_matrix_gpu7_all7_snapkv \
  --methods baseline \
  --temperatures 0.1,0.9 \
  --runs 10 \
  --fps 2.0 \
  --max_pixels 100352 \
  --max_frames_videomme 768 \
  --max_frames_other 128 \
  --max_new_tokens 256 \
  --measure_prefill \
  --qwen_model /data/armaan/models/Qwen2.5-Omni-7B
```

## 8. Remaining 6 methods after baseline

This is `120` executions total.

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python scripts/run_qwen_omni_benchmark_matrix.py \
  --gpus 7 \
  --output_root runs/qwen25_matrix_gpu7_all7_snapkv \
  --methods gptq,awq,omnizip,divprune,mixkv,rediprune \
  --temperatures 0.1,0.9 \
  --runs 10 \
  --fps 2.0 \
  --max_pixels 100352 \
  --max_frames_videomme 768 \
  --max_frames_other 128 \
  --max_new_tokens 256 \
  --measure_prefill \
  --qwen_model /data/armaan/models/Qwen2.5-Omni-7B \
  --gptq_model /data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4 \
  --awq_model /data/armaan/models/Qwen2.5-Omni-7B-AWQ \
  --mixkv_select_method snapkv
```

## 9. Full 7-method real run directly

Use this if you want to launch everything at once instead of doing baseline first.

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python scripts/run_qwen_omni_benchmark_matrix.py \
  --gpus 7 \
  --output_root runs/qwen25_matrix_gpu7_all7_snapkv \
  --methods baseline,gptq,awq,omnizip,divprune,mixkv,rediprune \
  --temperatures 0.1,0.9 \
  --runs 10 \
  --fps 2.0 \
  --max_pixels 100352 \
  --max_frames_videomme 768 \
  --max_frames_other 128 \
  --max_new_tokens 256 \
  --measure_prefill \
  --qwen_model /data/armaan/models/Qwen2.5-Omni-7B \
  --gptq_model /data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4 \
  --awq_model /data/armaan/models/Qwen2.5-Omni-7B-AWQ \
  --mixkv_select_method snapkv
```

## 10. Resume after interruption

This skips completed runs with successful `run_summary.json` files and reruns only incomplete or failed repeats.

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python scripts/run_qwen_omni_benchmark_matrix.py \
  --gpus 7 \
  --output_root runs/qwen25_matrix_gpu7_all7_snapkv \
  --methods baseline,gptq,awq,omnizip,divprune,mixkv,rediprune \
  --temperatures 0.1,0.9 \
  --runs 10 \
  --fps 2.0 \
  --max_pixels 100352 \
  --max_frames_videomme 768 \
  --max_frames_other 128 \
  --max_new_tokens 256 \
  --measure_prefill \
  --qwen_model /data/armaan/models/Qwen2.5-Omni-7B \
  --gptq_model /data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4 \
  --awq_model /data/armaan/models/Qwen2.5-Omni-7B-AWQ \
  --mixkv_select_method snapkv \
  --resume
```

## 11. Resume only the remaining 6 methods

```bash
cd /data/armaan/purs
/data/armaan/venvs/omnizip_clean/bin/python scripts/run_qwen_omni_benchmark_matrix.py \
  --gpus 7 \
  --output_root runs/qwen25_matrix_gpu7_all7_snapkv \
  --methods gptq,awq,omnizip,divprune,mixkv,rediprune \
  --temperatures 0.1,0.9 \
  --runs 10 \
  --fps 2.0 \
  --max_pixels 100352 \
  --max_frames_videomme 768 \
  --max_frames_other 128 \
  --max_new_tokens 256 \
  --measure_prefill \
  --qwen_model /data/armaan/models/Qwen2.5-Omni-7B \
  --gptq_model /data/armaan/models/Qwen2.5-Omni-7B-GPTQ-Int4 \
  --awq_model /data/armaan/models/Qwen2.5-Omni-7B-AWQ \
  --mixkv_select_method snapkv \
  --resume
```

## 12. Output locations

All outputs go under:

```text
/data/armaan/purs/runs/qwen25_matrix_gpu7_all7_snapkv/
```

Per run:

```text
<method>/temp_<temp>-run_<NN>/
```

So each method directory directly contains 20 run folders for the default matrix:

```text
baseline/
  temp_0p1-run_01/
  temp_0p1-run_02/
  ...
  temp_0p1-run_10/
  temp_0p9-run_01/
  ...
  temp_0p9-run_10/
```

New runs use this flat layout. `--resume` still recognizes successful runs written with the older
`<method>/temp_<temp>/run_<NN>/` layout.

Important files:

- `results.jsonl`
- `vram_log.jsonl`
- `gpu_samples.jsonl`
- `run_summary.json`
- `console.log`
- `stderr.log`
- `errors.log`

Top-level summaries:

- `summary_by_config.json`
- `summary_by_config.csv`
- `SUMMARY.md`
- `run_metrics.jsonl`

## 13. Method and combo counts

- `baseline` only: `2 temps x 10 repeats = 20 runs`
- remaining `6` methods: `6 x 2 x 10 = 120 runs`
- full matrix: `7 x 2 x 10 = 140 runs`

## 14. Notes

- Baseline is your full-KV condition.
- Frame caps are applied via `--max_frames_videomme` and `--max_frames_other`.
- Baseline, GPTQ, AWQ, OmniZip, and SnapKV-MixKV now emit per-question frame counts.
- DivPrune and ReDiPrune already emit original and pruned frame counts.
