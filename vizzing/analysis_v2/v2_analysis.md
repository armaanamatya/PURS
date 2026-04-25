# Layer Depth Analysis — L6 & L14 Metrics

**Videos:** 21 (worldsense)  
**Pruning rates:** audio drop 40%, video drop 70%  

## Cross-Q Jaccard (pruning decision consistency)

| Layer | Audio | Video |
|-------|-------|-------|
| L0 | 0.9951 ± 0.0020 | 0.9950 ± 0.0023 |
| L1 | 0.9922 ± 0.0051 | 0.9933 ± 0.0036 |
| L6 | 0.9826 ± 0.0111 | 0.9662 ± 0.0213 |
| L14 | 0.9348 ± 0.0258 | 0.8968 ± 0.0330 |

**L6→L14 Jaccard** (same question, do L6 and L14 agree on which tokens to keep?):  
Audio = 0.5432 ± 0.0406  
Video = 0.3485 ± 0.0420  

## Separation score

Uniform baseline ≈ 1.73. Values above 1.73 indicate the signal is more discriminative than random.

| Layer | Audio | Video |
|-------|-------|-------|
| L0 | 1.5900 ± 0.0406 ← below uniform | 1.7033 ± 0.0690 |
| L1 | 1.5535 ± 0.0507 ← below uniform | 1.7386 ± 0.0616 |
| L6 | 1.5816 ± 0.0423 ← below uniform | 1.7197 ± 0.0284 |
| L14 | 1.5882 ± 0.0330 ← below uniform | 1.7063 ± 0.0239 |

## Temporal autocorrelation

| Layer | Audio | Video |
|-------|-------|-------|
| L0 | 0.4452 ± 0.1667 | 0.4278 ± 0.0625 |
| L1 | 0.3530 ± 0.1439 | 0.4785 ± 0.0853 |
| L6 | 0.2926 ± 0.1233 | 0.4025 ± 0.0866 |
| L14 | 0.3786 ± 0.0761 | 0.4398 ± 0.0716 |
