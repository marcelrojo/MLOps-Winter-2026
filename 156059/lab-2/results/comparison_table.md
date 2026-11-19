# Benchmark Comparison: No Batching vs. Adaptive Batching

- **Device:** NVIDIA GeForce RTX 3050 Laptop GPU
- **Model:** `best-hpo-epoch=epoch=17-val_f1_macro=val_f1_macro=0.9613.ckpt`
- **Batching Params:** `max_batch_size=32`, `max_latency_ms=1000`
- **Methodology:** warmup=15.0s, window=60.0s, repeats=3 (mean ± std across trials)

## Required HW2 Comparison Table

| Concurrency | Throughput no batching [samples/s] | Throughput with batching [samples/s] | p95 latency no batching [ms] | p95 latency with batching [ms] |
|---|---|---|---|---|
| 1 | 110.89 ± 0.19 | 107.47 ± 2.07 | 11.05 ± 0.11 | 11.19 ± 0.21 |
| 8 | 131.66 ± 5.97 | 157.48 ± 5.10 | 68.99 ± 12.46 | 114.18 ± 9.39 |
| 32 | 80.01 ± 1.04 | 80.10 ± 0.18 | 1168.24 ± 12.60 | 1170.98 ± 12.14 |

## Extended Metrics (p50 / p99, Speedup & Actual Batch Size)

| Concurrency | Speedup (with/no) | p50 no / with [ms] | p99 no / with [ms] | Avg Batch Size (with) | Errors (no / with) |
|---|---|---|---|---|---|
| 1 | 0.97x | 8.60 ± 0.03 / 8.95 ± 0.17 | 12.78 ± 0.36 / 12.88 ± 0.11 | 1.00 ± 0.00 | 0 / 0 |
| 8 | 1.20x | 39.95 ± 0.45 / 38.08 ± 0.39 | 558.20 ± 142.64 / 265.52 ± 29.29 | 1.49 ± 0.06 | 0 / 0 |
| 32 | 1.00x | 269.21 ± 2.43 / 270.65 ± 6.54 | 1786.53 ± 15.27 / 1800.89 ± 38.18 | 1.05 ± 0.00 | 0 / 0 |
