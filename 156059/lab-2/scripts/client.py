"""Verification client for the BloodMNIST BentoML service.

Loads a sample test image from `artifacts/sample_inputs.npz`, encodes it as a
lossless base64 PNG payload, optionally writes the JSON payload for `curl`,
queries `/info` and `/predict`, and prints the prediction and latency.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

LAB2_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = LAB2_DIR / "artifacts"
SAMPLES_PATH = ARTIFACTS_DIR / "sample_inputs.npz"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def encode_image_to_b64_png(image_hwc: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 RGB NumPy array into a lossless base64 PNG string."""
    img = Image.fromarray(image_hwc, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a verification prediction request to the BloodMNIST BentoML service."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:3000",
        help="Base URL of the running BentoML service.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index in artifacts/sample_inputs.npz (0..63).",
    )
    parser.add_argument(
        "--save-payload",
        type=Path,
        default=None,
        help="Optional path to save the JSON request payload for curl testing.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP client timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not SAMPLES_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing exported artifacts in '{ARTIFACTS_DIR}'. "
            "Run `python scripts/export_model.py` first."
        )

    samples = np.load(SAMPLES_PATH)
    images = samples["images"]
    labels = samples["labels"]

    if not (0 <= args.index < len(images)):
        raise IndexError(
            f"--index {args.index} out of bounds for {len(images)} saved samples."
        )

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    class_names: list[str] = list(metadata["class_names"])

    gt_id = int(labels[args.index])
    gt_name = class_names[gt_id]
    b64_png = encode_image_to_b64_png(images[args.index])
    payload = {"images": [b64_png]}

    if args.save_payload is not None:
        args.save_payload.parent.mkdir(parents=True, exist_ok=True)
        args.save_payload.write_text(json.dumps(payload), encoding="utf-8")
        print(f"Saved request payload -> {args.save_payload}")

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        info_resp = client.post("/info", json={})
        info_resp.raise_for_status()
        info_data = info_resp.json()

        t0 = time.perf_counter()
        pred_resp = client.post("/predict", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        pred_resp.raise_for_status()
        pred_list = pred_resp.json()

    if not isinstance(pred_list, list) or len(pred_list) != 1:
        raise ValueError(f"Expected 1-element prediction list, got: {pred_list}")

    pred = pred_list[0]
    pred_id = int(pred["class_id"])
    pred_name = str(pred["class_name"])
    confidence = float(pred["confidence"])
    matched = pred_id == gt_id

    print("\n=== Service Info ===")
    print(json.dumps(info_data, indent=2))

    print("\n=== Verification Result ===")
    print(f"Sample index  : {args.index}")
    print(f"Ground truth  : {gt_id} ({gt_name})")
    print(f"Predicted     : {pred_id} ({pred_name}) | confidence={confidence:.4f}")
    print(f"Correct match : {matched}")
    print(f"Latency       : {elapsed_ms:.2f} ms")

    print("\n=== Raw JSON Response ===")
    print(json.dumps(pred_list, indent=2))


if __name__ == "__main__":
    main()
