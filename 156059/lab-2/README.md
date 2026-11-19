# Lab 2: Serving BloodMNIST with BentoML & Adaptive Batching Benchmark

### Marcel Rojewski `156059`

---

## 1. Overview

This lab packages and serves the **BloodMNIST** CNN classifier (`BloodNet`, `base_channels=64`, ~400k parameters, `28x28` RGB images, 8 cell classes) trained in [lab-1](../lab-1/README.md) as a CUDA-accelerated REST API using **BentoML (`1.4.40`)**, and benchmarks **no batching** vs. **adaptive batching** following the [HOMEWORK2.MD](../../HOMEWORK2.MD) methodology.

* **Source HW1 Checkpoint:** `../lab-1/checkpoints/hpo_best/best-hpo-epoch=epoch=17-val_f1_macro=val_f1_macro=0.9613.ckpt`
* **Exported Runtime Format:** Self-contained TorchScript module (`artifacts/bloodnet.ts.pt`, `CUDA max logit diff: 0.00e+00`, full test accuracy `0.9614` on `N=3421`)
* **Hardware:** `NVIDIA GeForce RTX 3050 Laptop GPU` (4096 MiB VRAM, CUDA `12.7` driver, `torch 2.14.1+cu126` under WSL2)

---

## 2. Directory Structure

```text
156059/lab-2/
├── pyproject.toml              # Isolated serving dependencies + explicit pytorch-cu126 index
├── uv.lock                     # Locked environment
├── README.md                   # Setup, run commands, client verification & benchmark results
├── service.py                  # BentoML service (CUDA TorchScript inference + adaptive batching)
├── src/
│   ├── __init__.py
│   └── model.py                # Standalone BloodNet definition & GPU uint8 preprocessing helper
├── scripts/
│   ├── export_model.py         # Extracts weights & train mean/std from HW1 ckpt -> TorchScript
│   ├── client.py               # Python httpx verification client (+ optional curl JSON emitter)
│   └── benchmark.py            # Asyncio + httpx benchmark harness (warm-up, 60s window, 3 repeats)
├── artifacts/
│   ├── metadata.json           # Normalization stats, class names, hyperparameters & parity metrics
│   ├── bloodnet.ts.pt          # Exported TorchScript model (gitignored; regenerated via export_model.py)
│   └── sample_inputs.npz       # 64 BloodMNIST test samples (gitignored; regenerated via export_model.py)
└── results/
    ├── benchmark_no_batching.json
    ├── benchmark_with_batching.json
    └── comparison_table.md
```

---

## 3. Setup & Model Export

```bash
cd 156059/lab-2
uv sync
source .venv/bin/activate

# Export HW1 checkpoint to TorchScript and verify CUDA parity against BloodMNIST test split
python scripts/export_model.py
```

---

## 4. Starting the Serving API

Both benchmark configurations run the exact same [service.py](service.py) code on 1 GPU worker (`workers=1, threads=1`), toggled via environment variables (`BENTOML_ACCESS_LOG=false` disables stdout access-log overhead during load tests):

### Mode A: No Batching (`batchable=False`)
```bash
BENTOML_BATCHING_ENABLED=false BENTOML_ACCESS_LOG=false bentoml serve service:BloodMNISTService --port 3000
```

### Mode B: Adaptive Batching (`batchable=True`, `max_batch_size=32`, `max_latency_ms=1000`)
```bash
BENTOML_BATCHING_ENABLED=true BENTOML_MAX_BATCH_SIZE=32 BENTOML_MAX_LATENCY_MS=1000 BENTOML_ACCESS_LOG=false bentoml serve service:BloodMNISTService --port 3000
```

---

## 5. Client Request Example & Verified Prediction

### Python Client (`httpx`) & `curl`
```bash
# Sends /info + /predict for test sample 0 and saves JSON payload for curl
python scripts/client.py --index 0 --save-payload artifacts/sample_request.json

# Equivalent cURL request
curl -s -X POST http://127.0.0.1:3000/predict \
  -H "Content-Type: application/json" \
  -d @artifacts/sample_request.json | python -m json.tool
```

### Observed Output (with `BENTOML_BATCHING_ENABLED=true`)
```text
=== Service Info ===
{
  "service": "bloodmnist_classifier",
  "device": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "source_checkpoint": "best-hpo-epoch=epoch=17-val_f1_macro=val_f1_macro=0.9613.ckpt",
  "batching_enabled": true,
  "max_batch_size": 32,
  "max_latency_ms": 1000,
  "num_classes": 8,
  "class_names": [
    "Basophil", "Eosinophil", "Erythroblast", "Immature Granulocyte",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
  ]
}

=== Verification Result ===
Sample index  : 0
Ground truth  : 0 (Basophil)
Predicted     : 0 (Basophil) | confidence=0.9992
Correct match : True
Latency       : 527.60 ms   # Cold first request (subsequent warm cURL request: 5.24 ms)

=== Raw JSON Response ===
[
  {
    "class_id": 0,
    "class_name": "Basophil",
    "confidence": 0.999203,
    "probabilities": {
      "Basophil": 0.999203,
      "Eosinophil": 0.000402,
      "Erythroblast": 4e-06,
      "Immature Granulocyte": 0.000112,
      "Lymphocyte": 2e-06,
      "Monocyte": 0.000276,
      "Neutrophil": 1e-06,
      "Platelet": 0.0
    }
  }
]
```

---

## 6. Benchmark Methodology & Results

Run against each server mode using [scripts/benchmark.py](scripts/benchmark.py):
```bash
python scripts/benchmark.py
```

* **Methodology:** `15.0 s` warm-up per concurrency level (discarded), followed by `3 × 60.0 s` measurement runs per concurrency (`1`, `8`, `32`), fixed pre-serialized lossless base64 PNG payload, persistent HTTP keep-alive pool (`httpx.AsyncClient`), `0` errors across all runs.
* **Server batch size tracking:** Sampled live from BentoML's Prometheus `/metrics` (`bentoml_service_adaptive_batch_size_{sum,count}`).

### Required Comparison Table (Mean ± Std across 3 × 60 s Runs)

| Concurrency | Throughput no batching [samples/s] | Throughput with batching [samples/s] | p95 latency no batching [ms] | p95 latency with batching [ms] |
|---|---|---|---|---|
| 1 | 110.89 ± 0.19 | 107.47 ± 2.07 | 11.05 ± 0.11 | 11.19 ± 0.21 |
| 8 | 131.66 ± 5.97 | 157.48 ± 5.10 | 68.99 ± 12.46 | 114.18 ± 9.39 |
| 32 | 80.01 ± 1.04 | 80.10 ± 0.18 | 1168.24 ± 12.60 | 1170.98 ± 12.14 |

### Extended Latency, Speedup & Actual Batch Size

| Concurrency | Speedup (with/no) | p50 no / with [ms] | p99 no / with [ms] | Avg Batch Size (with) | GPU-Util (`nvidia-smi`) | Errors (no / with) |
|---|---|---|---|---|---|---|
| 1 | 0.97x | 8.60 ± 0.03 / 8.95 ± 0.17 | 12.78 ± 0.36 / 12.88 ± 0.11 | 1.00 ± 0.00 | — | 0 / 0 |
| 8 | 1.20x | 39.95 ± 0.45 / 38.08 ± 0.39 | 558.20 ± 142.64 / 265.52 ± 29.29 | 1.49 ± 0.06 | 13% / 16–26% | 0 / 0 |
| 32 | 1.00x | 269.21 ± 2.43 / 270.65 ± 6.54 | 1786.53 ± 15.27 / 1800.89 ± 38.18 | 1.05 ± 0.00 | 12% / 9–14% | 0 / 0 |

### Conclusions
1. **Concurrency `1`:** Adaptive batching adds minimal queueing overhead (~`0.35 ms` p50), yielding `0.97x` throughput since no peer requests exist to batch (`avg_batch = 1.00`).
2. **Concurrency `8`:** Adaptive batching increases throughput by **+19.6%** (`131.66 -> 157.48 samples/s`) with `avg_batch = 1.49`, lifts GPU utilization (`13% -> 16–26%`), and cuts `p99` tail latency in half (`558.20 -> 265.52 ms`), while trading off higher `p95` latency (`68.99 -> 114.18 ms`).
3. **Concurrency `32`:** Both configurations saturate at `~80 samples/s` (`p95 ≈ 1170 ms`) and `avg_batch` drops back to `1.05`. Meanwhile `nvidia-smi` shows the GPU stays in idle power state `P8` (`5–6 W / 35 W`, `144 MiB / 4096 MiB` VRAM, `9–14%` util). Because `BloodNet` is tiny (~400k params, `28x28` images), the bottleneck at high localhost concurrency is **CPU / HTTP event-loop & PNG-decoding overhead upstream of the batcher**, starving `CorkDispatcher` before requests can accumulate into large batches.
