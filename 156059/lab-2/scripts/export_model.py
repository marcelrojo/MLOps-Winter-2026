"""Export the trained HW1 BloodClassifier checkpoint to a standalone TorchScript artifact.

Produces inside `artifacts/`:
- `bloodnet.ts.pt`: Self-contained TorchScript model in eval mode (no Lightning dependency).
- `metadata.json`: Training-split normalization stats, class names, and model hyperparameters.
- `sample_inputs.npz`: Sample BloodMNIST test images and labels for client tests and benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

LAB2_DIR = Path(__file__).resolve().parent.parent
LAB1_DIR = LAB2_DIR.parent / "lab-1"

if str(LAB2_DIR) not in sys.path:
    sys.path.insert(0, str(LAB2_DIR))

from src.model import CLASS_NAMES, BloodNet, preprocess_uint8_batch  # noqa: E402

DEFAULT_CKPT = (
    LAB1_DIR
    / "checkpoints"
    / "hpo_best"
    / "best-hpo-epoch=epoch=17-val_f1_macro=val_f1_macro=0.9613.ckpt"
)
DEFAULT_DATA = LAB1_DIR / "data" / "bloodmnist.npz"
DEFAULT_OUT_DIR = LAB2_DIR / "artifacts"


def compute_train_stats(data_path: Path, grayscale: bool) -> tuple[list[float], list[float]]:
    """Compute exact training-split channel mean and std matching Lab 1 BloodDataModule."""
    archive = np.load(data_path)
    train_images = archive["train_images"]
    train_float = train_images.astype(np.float32) / 255.0

    if grayscale:
        luma = (
            0.299 * train_float[..., 0]
            + 0.587 * train_float[..., 1]
            + 0.114 * train_float[..., 2]
        )
        return [float(luma.mean())], [float(luma.std())]

    mean_vals = train_float.mean(axis=(0, 1, 2))
    std_vals = train_float.std(axis=(0, 1, 2))
    return [float(m) for m in mean_vals], [float(s) for s in std_vals]


def load_bloodnet_from_ckpt(ckpt_path: Path) -> tuple[BloodNet, dict]:
    """Extract BloodNet architecture and weights from a PyTorch Lightning checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))

    in_channels = int(hparams.get("in_channels", 3))
    num_classes = int(hparams.get("num_classes", 8))
    base_channels = int(hparams.get("base_channels", 64))
    dropout = float(hparams.get("dropout", 0.15))

    model = BloodNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout,
    )

    raw_state_dict: dict[str, torch.Tensor] = ckpt["state_dict"]
    net_state_dict = {
        key.removeprefix("net."): value
        for key, value in raw_state_dict.items()
        if key.startswith("net.")
    }
    if not net_state_dict:
        raise KeyError(
            f"No 'net.*' weights found in checkpoint {ckpt_path}. "
            "Expected a BloodClassifier Lightning checkpoint."
        )

    model.load_state_dict(net_state_dict, strict=True)
    model.eval()
    return model, {
        "in_channels": in_channels,
        "num_classes": num_classes,
        "base_channels": base_channels,
        "dropout": dropout,
    }


def export_and_verify(ckpt_path: Path, data_path: Path, out_dir: Path) -> None:
    """Export TorchScript model, save metadata & test samples, and verify CUDA parity."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"BloodMNIST archive not found: {data_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for verification but torch.cuda.is_available() is False.")

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    eager_model, model_hparams = load_bloodnet_from_ckpt(ckpt_path)
    grayscale = model_hparams["in_channels"] == 1
    if grayscale:
        raise ValueError("Expected RGB checkpoint (in_channels=3) for this pipeline.")

    train_mean, train_std = compute_train_stats(data_path, grayscale=False)

    eager_model = eager_model.to(device)
    dummy_input = torch.randn(1, model_hparams["in_channels"], 28, 28, device=device)
    with torch.inference_mode():
        traced_model = torch.jit.trace(eager_model, dummy_input)
        traced_model.eval()

    ts_path = out_dir / "bloodnet.ts.pt"
    traced_model.save(str(ts_path))

    reloaded_ts = torch.jit.load(str(ts_path), map_location=device)
    reloaded_ts.eval()

    archive = np.load(data_path)
    test_images_np = archive["test_images"]  
    test_labels_np = archive["test_labels"].flatten()

    mean_t = torch.tensor(train_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(train_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    with torch.inference_mode():
        x_test = preprocess_uint8_batch(
            torch.from_numpy(test_images_np).to(device), mean=mean_t, std=std_t
        )
        eager_logits = eager_model(x_test)
        ts_logits = reloaded_ts(x_test)

    max_diff = float((eager_logits - ts_logits).abs().max().item())
    preds = ts_logits.argmax(dim=1).cpu().numpy()
    test_acc = float((preds == test_labels_np).mean())

    metadata = {
        "source_checkpoint": ckpt_path.name,
        "input_shape": [28, 28, 3],
        "input_dtype": "uint8",
        "train_mean": train_mean,
        "train_std": train_std,
        "class_names": CLASS_NAMES,
        "hyper_parameters": model_hparams,
        "verification": {
            "device": torch.cuda.get_device_name(0),
            "test_samples": int(len(test_labels_np)),
            "test_accuracy": round(test_acc, 6),
            "max_logit_diff_vs_eager": max_diff,
        },
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    selected_indices: list[int] = []
    for class_idx in range(model_hparams["num_classes"]):
        matches = np.where(test_labels_np == class_idx)[0][:8]
        selected_indices.extend(matches.tolist())

    samples_path = out_dir / "sample_inputs.npz"
    np.savez(
        samples_path,
        images=test_images_np[selected_indices],
        labels=test_labels_np[selected_indices],
    )

    print(f"Exported TorchScript model -> {ts_path}")
    print(f"Saved metadata             -> {metadata_path}")
    print(f"Saved sample inputs        -> {samples_path} ({len(selected_indices)} images)")
    print(f"Hyperparameters            : {model_hparams}")
    print(f"Train mean / std           : {train_mean} / {train_std}")
    print(f"CUDA max logit diff        : {max_diff:.2e}")
    print(f"Full test accuracy (N={len(test_labels_np)}): {test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HW1 BloodNet model to TorchScript.")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="Path to HW1 .ckpt file.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to bloodmnist.npz.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_and_verify(args.ckpt, args.data, args.out_dir)
