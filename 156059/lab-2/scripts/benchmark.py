"""Reproducible asyncio + httpx load-testing and comparison harness for HW2.

Implements the HOMEWORK2.MD benchmark methodology:
- Fixed pre-serialized BloodMNIST payload reused across all trials.
- Per-concurrency warm-up phase discarded before measuring (warms CUDA/cuDNN and
  allows BentoML's CorkDispatcher.Optimizer to adapt to the active concurrency).
- Fixed measurement window (default 60 s) across >= 3 concurrency levels (1, 8, 32).
- Multiple repetitions (default 3) reporting mean and std dev for throughput
  [samples/s] and latency [p50, p95, p99 in ms], plus server-side average batch size
  sampled from BentoML's `/metrics` Prometheus endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np

LAB2_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = LAB2_DIR / "artifacts"
SAMPLES_PATH = ARTIFACTS_DIR / "sample_inputs.npz"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from client import encode_image_to_b64_png  # noqa: E402

HEADERS = {"Content-Type": "application/json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BloodMNIST BentoML service with/without adaptive batching."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:3000",
        help="Base URL of the running BentoML service.",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 8, 32],
        help="Concurrency levels to sweep (default: 1 8 32).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Measurement window duration in seconds per trial (default: 60.0).",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=15.0,
        help="Warm-up duration in seconds before measuring each concurrency (default: 15.0).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated trials per concurrency level (default: 3).",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Image index in artifacts/sample_inputs.npz to send (default: 0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP request timeout in seconds (default: 30.0).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke test (5s warmup, 10s duration, 1 repeat; saves to .tmp/).",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip benchmarking and generate comparison table from saved JSON files.",
    )
    return parser.parse_args()


def load_fixed_payload(sample_index: int) -> bytes:
    """Load one BloodMNIST sample and serialize it once to avoid client-side noise."""
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(
            f"Sample archive not found at '{SAMPLES_PATH}'. "
            "Run `python scripts/export_model.py` first."
        )
    samples = np.load(SAMPLES_PATH)
    images = samples["images"]
    if not (0 <= sample_index < len(images)):
        raise IndexError(
            f"--sample-index {sample_index} out of bounds for {len(images)} samples."
        )
    b64_png = encode_image_to_b64_png(images[sample_index])
    return json.dumps({"images": [b64_png]}).encode("utf-8")


async def fetch_service_info(client: httpx.AsyncClient) -> dict[str, Any]:
    """Query POST /info to record active device and batching configuration."""
    resp = await client.post("/info", json={})
    resp.raise_for_status()
    return dict(resp.json())


async def fetch_batch_metric_totals(client: httpx.AsyncClient) -> tuple[float, float]:
    """Fetch cumulative adaptive_batch_size (count, sum) from BentoML /metrics."""
    try:
        resp = await client.get("/metrics")
        if resp.status_code != 200:
            return 0.0, 0.0
    except httpx.HTTPError:
        return 0.0, 0.0

    total_count = 0.0
    total_sum = 0.0
    for raw_line in resp.text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("bentoml_service_adaptive_batch_size_count"):
            try:
                total_count += float(line.split()[-1])
            except ValueError:
                continue
        elif line.startswith("bentoml_service_adaptive_batch_size_sum"):
            try:
                total_sum += float(line.split()[-1])
            except ValueError:
                continue
    return total_count, total_sum


async def run_window(
    client: httpx.AsyncClient,
    concurrency: int,
    duration_s: float,
    payload_bytes: bytes,
    *,
    collect_metrics: bool,
) -> dict[str, Any]:
    """Send sustained concurrent requests over a fixed duration window."""
    if collect_metrics:
        count_before, sum_before = await fetch_batch_metric_totals(client)
    else:
        count_before, sum_before = 0.0, 0.0

    latencies_ms: list[float] = []
    success_count = 0
    error_count = 0

    start_time = time.perf_counter()
    end_time = start_time + duration_s

    async def worker() -> None:
        nonlocal success_count, error_count
        while time.perf_counter() < end_time:
            t0 = time.perf_counter()
            try:
                resp = await client.post("/predict", content=payload_bytes, headers=HEADERS)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    success_count += 1
                    if collect_metrics:
                        latencies_ms.append(dt_ms)
                else:
                    error_count += 1
            except httpx.HTTPError:
                error_count += 1

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    elapsed_s = time.perf_counter() - start_time

    if not collect_metrics:
        return {
            "elapsed_s": elapsed_s,
            "successes": success_count,
            "errors": error_count,
        }

    count_after, sum_after = await fetch_batch_metric_totals(client)
    delta_count = max(0.0, count_after - count_before)
    delta_sum = max(0.0, sum_after - sum_before)
    avg_batch_size = (delta_sum / delta_count) if delta_count > 0.0 else 1.0

    if not latencies_ms:
        raise RuntimeError(
            f"All requests failed during measurement window (errors={error_count})."
        )

    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "elapsed_s": round(elapsed_s, 4),
        "successes": int(success_count),
        "errors": int(error_count),
        "throughput_samples_per_s": float(success_count / elapsed_s),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "avg_batch_size": float(avg_batch_size),
        "batch_calls": int(round(delta_count)),
    }


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean and sample std dev across repeated trials."""
    metric_keys = [
        "throughput_samples_per_s",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "avg_batch_size",
    ]
    ddof = 1 if len(trials) > 1 else 0
    summary: dict[str, Any] = {
        "repeats": len(trials),
        "total_successes": sum(int(t["successes"]) for t in trials),
        "total_errors": sum(int(t["errors"]) for t in trials),
    }
    for key in metric_keys:
        vals = np.asarray([float(t[key]) for t in trials], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals, ddof=ddof))
    return summary


async def run_benchmark_suite(args: argparse.Namespace) -> dict[str, Any]:
    """Execute warm-up + repeated measurement sweeps across all concurrency levels."""
    payload_bytes = load_fixed_payload(args.sample_index)

    max_conns = max(128, max(args.concurrencies) * 2)
    limits = httpx.Limits(
        max_connections=max_conns,
        max_keepalive_connections=max_conns,
    )

    async with httpx.AsyncClient(
        base_url=args.url, timeout=args.timeout, limits=limits
    ) as client:
        service_info = await fetch_service_info(client)
        batching_enabled = bool(service_info["batching_enabled"])
        mode_label = "with_batching" if batching_enabled else "no_batching"

        print("=== Target Service Configuration ===")
        print(
            f"Mode: {mode_label} | Device: {service_info.get('device')} | "
            f"max_batch_size={service_info.get('max_batch_size')} | "
            f"max_latency_ms={service_info.get('max_latency_ms')}"
        )
        print(
            f"Sweep: concurrencies={args.concurrencies} | warmup={args.warmup}s | "
            f"duration={args.duration}s | repeats={args.repeats}\n"
        )

        results_by_concurrency: dict[str, Any] = {}
        for concurrency in args.concurrencies:
            print(
                f"[Concurrency {concurrency}] Warming up for {args.warmup:.1f}s..."
            )
            warmup_stats = await run_window(
                client, concurrency, args.warmup, payload_bytes, collect_metrics=False
            )
            print(
                f"  Warm-up finished: {warmup_stats['successes']} reqs "
                f"({warmup_stats['errors']} errors)"
            )

            trials: list[dict[str, Any]] = []
            for trial_idx in range(1, args.repeats + 1):
                await asyncio.sleep(1.0)
                trial = await run_window(
                    client, concurrency, args.duration, payload_bytes, collect_metrics=True
                )
                trials.append(trial)
                print(
                    f"  Trial {trial_idx}/{args.repeats}: "
                    f"{trial['throughput_samples_per_s']:.2f} samples/s | "
                    f"p50={trial['p50_ms']:.2f} ms | p95={trial['p95_ms']:.2f} ms | "
                    f"p99={trial['p99_ms']:.2f} ms | avg_batch={trial['avg_batch_size']:.2f} | "
                    f"errors={trial['errors']}"
                )

            agg = _aggregate_trials(trials)
            results_by_concurrency[str(concurrency)] = {
                "concurrency": concurrency,
                "summary": agg,
                "trials": trials,
            }
            print(
                f"  -> Summary (c={concurrency}): "
                f"{agg['throughput_samples_per_s_mean']:.2f} ± {agg['throughput_samples_per_s_std']:.2f} samples/s | "
                f"p95={agg['p95_ms_mean']:.2f} ± {agg['p95_ms_std']:.2f} ms\n"
            )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "mode": mode_label,
        "service_info": service_info,
        "methodology": {
            "sample_index": args.sample_index,
            "concurrencies": list(args.concurrencies),
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "repeats": args.repeats,
        },
        "results": results_by_concurrency,
    }


def _fmt_stat(summary: dict[str, Any], key: str, precision: int = 2) -> str:
    mean = float(summary[f"{key}_mean"])
    std = float(summary[f"{key}_std"])
    if int(summary.get("repeats", 1)) > 1:
        return f"{mean:.{precision}f} ± {std:.{precision}f}"
    return f"{mean:.{precision}f}"


def print_single_mode_summary(report: dict[str, Any]) -> None:
    """Print a concise table for a single completed benchmark run."""
    print(f"=== Single-Mode Summary ({report['mode']}) ===")
    print(
        "| Concurrency | Throughput [samples/s] | p50 [ms] | p95 [ms] | p99 [ms] | Avg Batch Size | Errors |"
    )
    print("|---|---|---|---|---|---|---|")
    for c_key, entry in report["results"].items():
        s = entry["summary"]
        print(
            f"| {c_key} | {_fmt_stat(s, 'throughput_samples_per_s')} | "
            f"{_fmt_stat(s, 'p50_ms')} | {_fmt_stat(s, 'p95_ms')} | "
            f"{_fmt_stat(s, 'p99_ms')} | {_fmt_stat(s, 'avg_batch_size')} | "
            f"{s['total_errors']} |"
        )
    print()


def build_comparison_markdown(
    no_batch: dict[str, Any], with_batch: dict[str, Any]
) -> str:
    """Render the HW2 required comparison table plus extended latency/batch stats."""
    no_info = no_batch["service_info"]
    with_info = with_batch["service_info"]
    meth = with_batch["methodology"]

    lines: list[str] = [
        "# Benchmark Comparison: No Batching vs. Adaptive Batching",
        "",
        f"- **Device:** {with_info.get('device')}",
        f"- **Model:** `{with_info.get('source_checkpoint')}`",
        (
            f"- **Batching Params:** `max_batch_size={with_info.get('max_batch_size')}`, "
            f"`max_latency_ms={with_info.get('max_latency_ms')}`"
        ),
        (
            f"- **Methodology:** warmup={meth['warmup_s']}s, window={meth['duration_s']}s, "
            f"repeats={meth['repeats']} (mean ± std across trials)"
        ),
        "",
        "## Required HW2 Comparison Table",
        "",
        "| Concurrency | Throughput no batching [samples/s] | Throughput with batching [samples/s] | p95 latency no batching [ms] | p95 latency with batching [ms] |",
        "|---|---|---|---|---|",
    ]

    common_concurrencies = [
        c for c in no_batch["results"] if c in with_batch["results"]
    ]
    for c_key in common_concurrencies:
        s_no = no_batch["results"][c_key]["summary"]
        s_with = with_batch["results"][c_key]["summary"]
        lines.append(
            f"| {c_key} | {_fmt_stat(s_no, 'throughput_samples_per_s')} | "
            f"{_fmt_stat(s_with, 'throughput_samples_per_s')} | "
            f"{_fmt_stat(s_no, 'p95_ms')} | {_fmt_stat(s_with, 'p95_ms')} |"
        )

    lines.extend(
        [
            "",
            "## Extended Metrics (p50 / p99, Speedup & Actual Batch Size)",
            "",
            "| Concurrency | Speedup (with/no) | p50 no / with [ms] | p99 no / with [ms] | Avg Batch Size (with) | Errors (no / with) |",
            "|---|---|---|---|---|---|",
        ]
    )

    for c_key in common_concurrencies:
        s_no = no_batch["results"][c_key]["summary"]
        s_with = with_batch["results"][c_key]["summary"]
        tput_no = float(s_no["throughput_samples_per_s_mean"])
        tput_with = float(s_with["throughput_samples_per_s_mean"])
        speedup = (tput_with / tput_no) if tput_no > 0 else 0.0
        lines.append(
            f"| {c_key} | {speedup:.2f}x | "
            f"{_fmt_stat(s_no, 'p50_ms')} / {_fmt_stat(s_with, 'p50_ms')} | "
            f"{_fmt_stat(s_no, 'p99_ms')} / {_fmt_stat(s_with, 'p99_ms')} | "
            f"{_fmt_stat(s_with, 'avg_batch_size')} | "
            f"{s_no['total_errors']} / {s_with['total_errors']} |"
        )

    if no_info.get("device") != with_info.get("device"):
        lines.append(
            "\n> **Warning:** Device mismatch between runs: "
            f"`{no_info.get('device')}` vs `{with_info.get('device')}`."
        )

    lines.append("")
    return "\n".join(lines)


def maybe_write_comparison(out_dir: Path, prefix: str) -> None:
    """If both mode reports exist in out_dir, generate and print the comparison table."""
    no_path = out_dir / f"{prefix}_no_batching.json"
    with_path = out_dir / f"{prefix}_with_batching.json"
    if not (no_path.exists() and with_path.exists()):
        missing = "with_batching" if no_path.exists() else "no_batching"
        print(
            f"Saved results. Run the `{missing}` configuration to generate the side-by-side comparison table."
        )
        return

    no_data = json.loads(no_path.read_text(encoding="utf-8"))
    with_data = json.loads(with_path.read_text(encoding="utf-8"))
    md = build_comparison_markdown(no_data, with_data)

    comp_name = "smoke_comparison.md" if prefix == "smoke" else "comparison_table.md"
    comp_path = out_dir / comp_name
    comp_path.write_text(md, encoding="utf-8")
    print(f"Saved comparison table -> {comp_path}\n")
    print(md)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.warmup = 5.0
        args.duration = 10.0
        args.repeats = 1
        out_dir = LAB2_DIR / ".tmp"
        prefix = "smoke"
    else:
        out_dir = LAB2_DIR / "results"
        prefix = "benchmark"

    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_only:
        maybe_write_comparison(out_dir, prefix)
        return

    report = asyncio.run(run_benchmark_suite(args))
    out_file = out_dir / f"{prefix}_{report['mode']}.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved benchmark JSON -> {out_file}\n")

    print_single_mode_summary(report)
    maybe_write_comparison(out_dir, prefix)


if __name__ == "__main__":
    main()
