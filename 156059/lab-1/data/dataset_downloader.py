"""BloodMNIST Dataset Downloader.

Downloads the BloodMNIST dataset (part of MedMNIST v2) for MLOps Lab 1.
Supports automatic download via the `medmnist` Python package, with a fallback
to direct Zenodo HTTP download using standard library tools.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

ZENODO_URL = "https://zenodo.org/records/6496656/files/bloodmnist.npz?download=1"
EXPECTED_MD5 = "7053d0359d879ad8a5505303e11de1dc"
LEGACY_MD5 = "ac60eb7f0e3f05022378a3ca8fa66893"
FILE_NAME = "bloodmnist.npz"


def check_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file against expected hashes."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    return digest == expected_md5 or digest == LEGACY_MD5


def download_via_medmnist(target_dir: Path) -> bool:
    """Attempt download using the medmnist package."""
    try:
        from medmnist import BloodMNIST

        print("Attempting download via 'medmnist' package...")
        BloodMNIST(split="train", download=True, root=str(target_dir))
        return True
    except ImportError:
        print("'medmnist' package not found in current environment. Using HTTP fallback...")
        return False
    except Exception as e:
        print(f"medmnist download encountered an issue ({e}). Using HTTP fallback...")
        return False


def download_via_http(target_path: Path) -> None:
    """Download bloodmnist.npz directly from Zenodo with a progress indicator."""
    print(f"Downloading from Zenodo:\n  {ZENODO_URL}")
    print(f"Destination: {target_path}")

    def report_progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:5.1f}% [{mb_downloaded:.2f} MB / {mb_total:.2f} MB]")
        else:
            sys.stdout.write(f"\rDownloaded: {downloaded / (1024 * 1024):.2f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(ZENODO_URL, target_path, reporthook=report_progress)
    sys.stdout.write("\n")


def inspect_dataset(file_path: Path) -> None:
    """Print dataset shape and verification summary."""
    try:
        import numpy as np

        data = np.load(file_path)
        print("\n--- BloodMNIST Dataset Verification ---")
        print(f"Archive file:   {file_path} ({file_path.stat().st_size / (1024 * 1024):.2f} MB)")
        for split in ["train", "val", "test"]:
            imgs = data[f"{split}_images"]
            labels = data[f"{split}_labels"]
            print(f"  {split.capitalize():5s} split: {imgs.shape[0]:5d} images | shape: {imgs.shape} | labels: {labels.shape}")
        print("Classes:        8 classes (0..7)")
        print("Status:         Ready for training.")
    except ImportError:
        print("\nNumPy not installed in this environment; skipping array inspection.")
        print(f"File verified at {file_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download BloodMNIST dataset into the data directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Target directory to store bloodmnist.npz (default: data/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if bloodmnist.npz already exists.",
    )
    args = parser.parse_args()

    target_dir: Path = args.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / FILE_NAME

    if target_file.exists() and not args.force:
        print(f"Dataset already exists at: {target_file}")
        if check_md5(target_file, EXPECTED_MD5):
            print("Checksum verified (MD5 match).")
            inspect_dataset(target_file)
            return
        print("Existing file checksum mismatch. Re-downloading...")

    downloaded = download_via_medmnist(target_dir)

    if not downloaded or not target_file.exists():
        download_via_http(target_file)

    if check_md5(target_file, EXPECTED_MD5):
        print("Download complete. Checksum successfully verified!")
    else:
        print("Warning: MD5 checksum did not match the expected Zenodo hash.")

    inspect_dataset(target_file)


if __name__ == "__main__":
    main()

