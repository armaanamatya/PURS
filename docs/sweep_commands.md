# Hyperparameter Sweep: Accuracy vs Speedup

All commands assume:
```bash
source /data/armaan/venvs/omnizip_clean/bin/activate
cd /data/armaan/purs
export META=videos/metadata.json
export VIDS=videos
export FPS=2.0
export MP=100352
export MNT=256
export ALLOC=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Pick a free GPU for each run. Each run takes ~20-40 min on a single RTX 6000 Ada.

---

## Sweep Variables Per Method

| Method | Primary Sweep | Values | Secondary (fixed) |
|-----------|----------------------|--------------------------------|-------------------------------|
| OmniZip | `rho_video` | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 | rho_audio=0.3, g=3, cr=0.05 |
| DivPrune | `subset_ratio` | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 | prune_mode=frame |
| MixKV | `budget` | 64, 128, 256, 512, 1024 | window_size=32, snapkv |
| ReDiPrune | `subset_ratio` | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 | alpha=0.5, tau=0.0, frame |

**Why these variables:**
- **OmniZip `rho_video`**: Controls what fraction of video KV tokens survive compression. This is the main speed/quality knob. `rho_audio` is less impactful (audio tokens are fewer) and g/contextual_ratio are architectural, not retention knobs.
- **DivPrune `subset_ratio`**: Directly controls what fraction of frames are kept. The only meaningful knob.
- **MixKV `budget`**: Total KV cache tokens per attention head. Lower = more aggressive compression. This is the single knob that controls the speed/quality tradeoff.
- **ReDiPrune `subset_ratio`**: Same as DivPrune — frame retention ratio. Alpha/tau are secondary (alpha=0.5 is paper default, tau=0.0 means no filtering).

---

## 1. OmniZip Sweep (rho_video)

```bash
for RV in 0.3 0.4 0.5 0.6 0.7 0.8; do
  DIR=sweep/omnizip_rv${RV}
  echo "=== OmniZip rho_video=$RV ==="
  $ALLOC CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_zip.py \
    --metadata $META --videos $VIDS \
    --output $DIR/results.jsonl --log $DIR/console.log \
    --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log \
    --stderr_log $DIR/stderr.log \
    --fps $FPS --max_pixels $MP --max_new_tokens $MNT \
    --rho_audio 0.3 --rho_video $RV --g 3 --contextual_ratio 0.05 \
    --measure_prefill
done
```

## 2. DivPrune Sweep (subset_ratio)

```bash
for SR in 0.3 0.4 0.5 0.6 0.7 0.8; do
  DIR=sweep/divprune_sr${SR}
  echo "=== DivPrune subset_ratio=$SR ==="
  $ALLOC CUDA_VISIBLE_DEVICES=1 python eval_qwen_omni_divprune.py \
    --metadata $META --videos $VIDS \
    --output $DIR/results.jsonl --log $DIR/console.log \
    --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log \
    --stderr_log $DIR/stderr.log \
    --fps $FPS --max_pixels $MP --max_new_tokens $MNT \
    --subset_ratio $SR --prune_mode frame \
    --measure_prefill
done
```

## 3. MixKV Sweep (budget)

```bash
for B in 64 128 256 512 1024; do
  DIR=sweep/mixkv_b${B}
  echo "=== MixKV budget=$B ==="
  $ALLOC CUDA_VISIBLE_DEVICES=2 python eval_qwen_omni_mixkv.py \
    --metadata $META --videos $VIDS \
    --output $DIR/results.jsonl --log $DIR/console.log \
    --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log \
    --stderr_log $DIR/stderr.log \
    --fps $FPS --max_pixels $MP --max_new_tokens $MNT \
    --budget $B --select_method snapkv \
    --measure_prefill
done
```

## 4. ReDiPrune Sweep (subset_ratio)

```bash
for SR in 0.3 0.4 0.5 0.6 0.7 0.8; do
  DIR=sweep/rediprune_sr${SR}
  echo "=== ReDiPrune subset_ratio=$SR ==="
  $ALLOC CUDA_VISIBLE_DEVICES=3 python eval_qwen_omni_rediprune.py \
    --metadata $META --videos $VIDS \
    --output $DIR/results.jsonl --log $DIR/console.log \
    --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log \
    --stderr_log $DIR/stderr.log \
    --fps $FPS --max_pixels $MP --max_new_tokens $MNT \
    --subset_ratio $SR --prune_mode frame --alpha 0.5 --tau 0.0 \
    --measure_prefill
done
```

---

## Running in Parallel

All 4 sweeps can run simultaneously on GPUs 0-3 (each sweep is sequential within itself since it reloads the model each iteration). Total: 6+6+5+6 = 23 runs.

If only 1 GPU is free, run them sequentially:
```bash
# All sweeps back-to-back on GPU 0
for RV in 0.3 0.4 0.5 0.6 0.7 0.8; do DIR=sweep/omnizip_rv${RV}; echo "=== OmniZip rv=$RV ==="; $ALLOC CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_zip.py --metadata $META --videos $VIDS --output $DIR/results.jsonl --log $DIR/console.log --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log --stderr_log $DIR/stderr.log --fps $FPS --max_pixels $MP --max_new_tokens $MNT --rho_audio 0.3 --rho_video $RV --g 3 --contextual_ratio 0.05 --measure_prefill; done && \
for SR in 0.3 0.4 0.5 0.6 0.7 0.8; do DIR=sweep/divprune_sr${SR}; echo "=== DivPrune sr=$SR ==="; $ALLOC CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_divprune.py --metadata $META --videos $VIDS --output $DIR/results.jsonl --log $DIR/console.log --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log --stderr_log $DIR/stderr.log --fps $FPS --max_pixels $MP --max_new_tokens $MNT --subset_ratio $SR --prune_mode frame --measure_prefill; done && \
for B in 64 128 256 512 1024; do DIR=sweep/mixkv_b${B}; echo "=== MixKV b=$B ==="; $ALLOC CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_mixkv.py --metadata $META --videos $VIDS --output $DIR/results.jsonl --log $DIR/console.log --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log --stderr_log $DIR/stderr.log --fps $FPS --max_pixels $MP --max_new_tokens $MNT --budget $B --select_method snapkv --measure_prefill; done && \
for SR in 0.3 0.4 0.5 0.6 0.7 0.8; do DIR=sweep/rediprune_sr${SR}; echo "=== ReDiPrune sr=$SR ==="; $ALLOC CUDA_VISIBLE_DEVICES=0 python eval_qwen_omni_rediprune.py --metadata $META --videos $VIDS --output $DIR/results.jsonl --log $DIR/console.log --vram_log $DIR/vram_log.jsonl --errors_log $DIR/errors.log --stderr_log $DIR/stderr.log --fps $FPS --max_pixels $MP --max_new_tokens $MNT --subset_ratio $SR --prune_mode frame --alpha 0.5 --tau 0.0 --measure_prefill; done && \
echo "=== ALL SWEEPS DONE ==="
```

---

## Collecting Results

After sweeps complete, copy to Windows:
```powershell
scp -r armaan@10.244.120.178:/data/armaan/purs/sweep "C:\Users\Armaan\Desktop\PURS\"
```

## Output Structure

```
sweep/
  omnizip_rv0.3/   omnizip_rv0.4/   ... omnizip_rv0.8/
  divprune_sr0.3/  divprune_sr0.4/  ... divprune_sr0.8/
  mixkv_b64/       mixkv_b128/      ... mixkv_b1024/
  rediprune_sr0.3/ rediprune_sr0.4/ ... rediprune_sr0.8/
```

Each subfolder contains: results.jsonl, console.log, vram_log.jsonl, errors.log, stderr.log

## Plotting

After collecting, extract summary with:
```bash
# On Lambda or locally with python
python -c "
import json, statistics, glob, os

print('method,param,value,accuracy,prefill_ms,peak_vram_gb')
for d in sorted(glob.glob('sweep/*')):
    name = os.path.basename(d)
    try:
        results = [json.loads(l) for l in open(os.path.join(d,'results.jsonl'))]
        vram = [json.loads(l) for l in open(os.path.join(d,'vram_log.jsonl'))]
        acc = sum(1 for r in results if r['correct']) / len(results) * 100
        pf = statistics.mean([r['prefill_ms'] for r in results if 'prefill_ms' in r])
        peak = max(v['peak_alloc_gb'] for v in vram)
        # Parse method and param from dirname
        parts = name.rsplit('_', 1)
        method = parts[0]
        val = parts[1] if len(parts)>1 else ''
        param = val[0:2] if val else ''
        print(f'{method},{param},{val[2:] if len(val)>2 else val},{acc:.1f},{pf:.0f},{peak:.2f}')
    except Exception as e:
        print(f'# {name}: {e}')
"
```

This CSV can be plotted directly in matplotlib or any charting tool.
