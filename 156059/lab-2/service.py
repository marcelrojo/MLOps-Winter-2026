"""BentoML service for BloodMNIST classification on CUDA with adaptive batching.

Exposes:
- POST /info: Service runtime, device, and batching configuration metadata.
- POST /predict: Single- or batched-image classification endpoint accepting
  base64-encoded 28x28 RGB PNG images (`{"images": ["<b64_png>"]}`).

Batching mode and parameters are controlled via environment variables so
no-batching vs. adaptive-batching benchmarks run identical model & code:
- BENTOML_BATCHING_ENABLED (default: "true")
- BENTOML_MAX_BATCH_SIZE (default: "32", clamped >= 2 by BentoML validation)
- BENTOML_MAX_LATENCY_MS (default: "1000")
- BENTOML_ACCESS_LOG (default: "true")
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import torch
from PIL import Image

LAB2_DIR = Path(__file__).resolve().parent
if str(LAB2_DIR) not in sys.path:
    sys.path.insert(0, str(LAB2_DIR))

from src.model import preprocess_uint8_batch  # noqa: E402


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


BATCHING_ENABLED: bool = _env_bool("BENTOML_BATCHING_ENABLED", True)
MAX_BATCH_SIZE: int = max(2, int(os.getenv("BENTOML_MAX_BATCH_SIZE", "32")))
MAX_LATENCY_MS: int = int(os.getenv("BENTOML_MAX_LATENCY_MS", "1000"))
ACCESS_LOG_ENABLED: bool = _env_bool("BENTOML_ACCESS_LOG", True)

ARTIFACTS_DIR = LAB2_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "bloodnet.ts.pt"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


def _decode_b64_png(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded image string into a (28, 28, 3) uint8 RGB array."""
    raw_bytes = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(raw_bytes)) as img:
        rgb = img.convert("RGB")
        if rgb.size != (28, 28):
            rgb = rgb.resize((28, 28), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


@bentoml.service(
    name="bloodmnist_classifier",
    workers=1,
    threads=1,
    traffic={"timeout": 60},
    metrics={"enabled": True},
    logging={"access": {"enabled": ACCESS_LOG_ENABLED}},
)
class BloodMNISTService:
    """BentoML service wrapping the TorchScript BloodNet classifier on CUDA."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for BloodMNISTService, but torch.cuda.is_available() is False."
            )
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Missing exported artifacts in '{ARTIFACTS_DIR}'. "
                "Run `python scripts/export_model.py` first."
            )

        self.device = torch.device("cuda")
        self.device_name = torch.cuda.get_device_name(0)

        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.class_names: list[str] = list(metadata["class_names"])
        self.source_checkpoint: str = str(metadata["source_checkpoint"])

        self.model = torch.jit.load(str(MODEL_PATH), map_location=self.device)
        self.model.eval()

        self.mean = torch.tensor(
            metadata["train_mean"], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            metadata["train_std"], dtype=torch.float32, device=self.device
        ).view(1, 3, 1, 1)

        # Warm up CUDA context and cuDNN kernel selection for single and batched shapes.
        warmup_sizes = sorted({1, min(32, MAX_BATCH_SIZE)})
        with torch.inference_mode():
            for bs in warmup_sizes:
                dummy_u8 = torch.zeros(
                    (bs, 28, 28, 3), dtype=torch.uint8, device=self.device
                )
                x = preprocess_uint8_batch(dummy_u8, mean=self.mean, std=self.std)
                _ = self.model(x)
        torch.cuda.synchronize()

    @bentoml.api(route="/info")
    def info(self) -> dict[str, Any]:
        """Return active device, model, and batching configuration."""
        return {
            "service": "bloodmnist_classifier",
            "device": self.device_name,
            "source_checkpoint": self.source_checkpoint,
            "batching_enabled": BATCHING_ENABLED,
            "max_batch_size": MAX_BATCH_SIZE,
            "max_latency_ms": MAX_LATENCY_MS,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
        }

    @bentoml.api(
        route="/predict",
        batchable=BATCHING_ENABLED,
        batch_dim=0,
        max_batch_size=MAX_BATCH_SIZE,
        max_latency_ms=MAX_LATENCY_MS,
    )
    def predict(self, images: list[str]) -> list[dict[str, Any]]:
        """Classify a list of base64-encoded PNG images into BloodMNIST cell types."""
        if not images:
            raise ValueError("Input 'images' list must not be empty.")

        decoded = [_decode_b64_png(b64_str) for b64_str in images]
        batch_np = np.stack(decoded, axis=0)  # (B, 28, 28, 3) uint8

        with torch.inference_mode():
            x_u8 = torch.from_numpy(batch_np).to(self.device)
            x = preprocess_uint8_batch(x_u8, mean=self.mean, std=self.std)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidences, pred_ids = probs.max(dim=1)

        pred_ids_list: list[int] = pred_ids.cpu().tolist()
        conf_list: list[float] = confidences.cpu().tolist()
        probs_list: list[list[float]] = probs.cpu().tolist()

        results: list[dict[str, Any]] = []
        for cid, conf, prob_row in zip(pred_ids_list, conf_list, probs_list):
            results.append(
                {
                    "class_id": int(cid),
                    "class_name": self.class_names[cid],
                    "confidence": round(float(conf), 6),
                    "probabilities": {
                        name: round(float(p), 6)
                        for name, p in zip(self.class_names, prob_row)
                    },
                }
            )
        return results
