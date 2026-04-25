# Score Analysis — L0 & L1 Metrics

**Source:** `early_layer_relevance/scores.jsonl`  
**Videos with ≥2 questions:** 39  
**Pruning rates:** audio drop 40%, video drop 70%  

## Global aggregate

| Metric | Layer | Audio | Video | N |
|--------|-------|-------|-------|---|
| jaccard | L0 | 0.9951 ± 0.0020 | 0.9950 ± 0.0023 | 39 |
| jaccard | L1 | 0.9922 ± 0.0051 | 0.9933 ± 0.0036 | 39 |
| separation | L0 | 1.5900 ± 0.0406 | 1.7033 ± 0.0690 | 39 |
| separation | L1 | 1.5535 ± 0.0507 | 1.7386 ± 0.0616 | 39 |
| autocorr | L0 | 0.4452 ± 0.1667 | 0.4278 ± 0.0625 | 39 |
| autocorr | L1 | 0.3530 ± 0.1439 | 0.4785 ± 0.0853 | 39 |

## Interpretation

**Top-k Jaccard:** Audio Jaccard at L0=0.995, L1=0.992. High Jaccard confirms the pruning decision is nearly question-invariant.

**Separation score:** Audio at L0=1.590, L1=1.553. Score separation is meaningful even at L0. 

**Temporal autocorrelation:** Audio at L0=0.445, L1=0.353. Moderate autocorrelation — signal has some temporal structure but is not strongly block-structured.

**Plot:** `scores_analysis.png`  
**Per-video data:** `scores_analysis.jsonl`  
