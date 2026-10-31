# Lab 1: PyTorch Lightning, Model Monitoring & Hyperparameter Optimization

Student ID: **156059**  
Course: **MLOps Winter 2026**  
Task: **BloodMNIST (28x28 Microscopic Blood Cell Classification - 8 Classes)**  

---

## 1. Setup Environment

Sync dependencies into a virtual environment with CUDA 12.6 support using `uv`:

```bash
cd 156059/lab-1

# Optional (WSL): Suppress cross-filesystem hardlink warning on /mnt/c
export UV_LINK_MODE=copy

# Create & sync virtual environment with CUDA 12.6
uv sync --extra cu126

# Activate the virtual environment
source .venv/bin/activate
```

*(For CPU-only machines: replace `cu126` with `cpu`: `uv sync --extra cpu`)*

---

## 2. Download Dataset

Once your virtual environment is activated (`source .venv/bin/activate`), download and verify the **BloodMNIST** dataset (~15.4 MB) into `data/`:

```bash
python data/dataset_downloader.py
```

*(Alternatively, without activating the venv: `uv run --extra cu126 python data/dataset_downloader.py`)*

The script will fetch `bloodmnist.npz`, verify the MD5 checksum, and display the train/val/test split shapes.  
For detailed dataset documentation and class descriptions, see [data/README.md](file:///mnt/c/Studia/Masters/2nd_term/MLOps/MLOps-Winter-2026/156059/lab-1/data/README.md).

---

## 3. Project Structure

```text
156059/
├── .gitignore
└── lab-1/
    ├── pyproject.toml
    ├── README.md
    ├── data/
    │   ├── README.md               # Dataset documentation & characteristics
    │   ├── dataset_downloader.py   # Automated downloader & checksum verifier
    │   └── bloodmnist.npz          # Downloaded data (gitignored)
    └── src/
        ├── datamodule.py           # PyTorch LightningDataModule
        ├── model.py                # PyTorch LightningModule (architecture & metrics)
        ├── train.py                # Baseline training script with WandB logger
        └── optimize.py             # Optuna hyperparameter optimization script
```
