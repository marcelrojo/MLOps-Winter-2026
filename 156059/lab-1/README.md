# Lab 1: PyTorch Lightning, Model Monitoring & Hyperparameter Optimization

### Marcel Rojewski `156059`  

---

## 1. Executive Summary & Verification Results

Used Optuna search space that combined data representation (color space, domain-safe augmentations, inverse-frequency class weighting) with network capacity and optimization dynamics:

| Metric | Baseline (`01_baseline_training.ipynb`) | Tuned Model (`02_hpo_optuna.ipynb`) | Absolute Lift | Relative Gain | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Macro-F1** (Primary) | `0.9059` | **`0.9613`** | **$+0.0554$** ($+5.54\text{ pts}$) | **$+6.11\%$** | **PASSED** ($\ge 3\%$) |
| **Validation Accuracy** | `0.9039` | **`0.9639`** | **$+0.0600$** ($+6.00\text{ pts}$) | **$+6.64\%$** | — |
| **Validation Loss** | `0.2493` | **`0.1130`** | **$-0.1363$** | **$-54.7\%$** | — |
| **Test Set Macro-F1** | `0.8952` | **`0.9591`** | **$+0.0639$** ($+6.39\text{ pts}$) | **$+7.14\%$** | — |
| **Test Set Accuracy** | `0.8912` | **`0.9596`** | **$+0.0684$** ($+6.84\text{ pts}$) | **$+7.67\%$** | — |
| **Test Set CrossEntropy Loss** | `0.2578` | **`0.1268`** | **$-0.1310$** | **Loss Halved ($-50.8\%$)** | — |

---

## 2. Environment Setup & Data Download

### Virtual Environment with `uv`
The project environment is managed via `pyproject.toml` using `uv`:

```bash
cd 156059/lab-1

# Sync environment with CUDA 12.6 support (for CPU-only, use --extra cpu)
uv sync --extra cu126

# Activate environment
source .venv/bin/activate
```

### Dataset Download
Fetch and verify the MD5 checksum of the BloodMNIST dataset (~15.4 MB, 17,092 images):

```bash
python data/dataset_downloader.py
```

The split distribution is:
- **Train:** 11,959 images
- **Validation:** 1,712 images
- **Test:** 3,421 images

---

## 3. Project Architecture & Repository Structure

The repository separates concerns between modular, testable PyTorch Lightning code (`src/`) and narrative experimental notebooks (`notebooks/`):

```text
156059/
└── lab-1/
    ├── pyproject.toml               # Project metadata & pinned dependencies
    ├── README.md                    # Project documentation & presentation cheat sheet
    ├── data/
    │   ├── README.md                # Dataset characteristics & class descriptions
    │   ├── dataset_downloader.py    # Automated download and MD5 verification
    │   └── bloodmnist.npz           # Cached dataset archive (gitignored)
    ├── src/
    │   ├── __init__.py
    │   ├── data.py                  # BloodDataset & BloodDataModule (in-memory caching, stats, augs)
    │   ├── model.py                 # BloodNet (ConvNet backbone) & BloodClassifier (LightningModule)
    │   └── tuning.py                # BloodObjective, OptunaPruningCallback & study runners
    ├── tests/
    │   └── test_tuning.py           # Unit tests for tuning components & parameter sampling
    ├── logs/
    │   ├── baseline/                # CSV & WandB logs for baseline run
    │   ├── hpo_best_v2/             # CSV logs & curves for final tuned model
    │   └── optuna_study_v2.db       # Persistent SQLite database of all Optuna trials
    ├── checkpoints/
    │   ├── baseline/                # Best baseline model checkpoint
    │   └── hpo_best/                # Best tuned model checkpoint (val_f1_macro = 0.9613)
    └── notebooks/
        ├── 01_baseline_training.ipynb  # Baseline training, WandB tracking, curves & error analysis
        └── 02_hpo_optuna.ipynb         # Unified Optuna study, visual analytics & >=3% lift verification
```

---

## 4. Baseline Experiment & Monitoring

In `notebooks/01_baseline_training.ipynb`
- **Architecture:** `BloodNet` with `base_channels = 32`, `dropout = 0.2` (~102k params).
- **Optimization:** `AdamW` (`lr = 1e-3`, `weight_decay = 1e-4`), `CosineAnnealingLR` over 20 epochs, `batch_size = 64`.
- **Data Configuration:** Standard augmentations (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15)`, `ColorJitter`), unweighted loss.
- **Monitoring Tools:**
  - `WandbLogger` (logging metrics and run artifacts to Weights & Biases under project `mlops-lab-1-bloodmnist`).
  - `CSVLogger` (local metric curves in `logs/baseline/version_0/`).
  - `ModelCheckpoint` monitoring `val_f1_macro` (`mode="max"`).
  - `LearningRateMonitor` tracking learning rate schedules.
- **Learning Verification:** Training and validation loss curves confirm steady descent and stable convergence without severe divergence.
- **Baseline Result:** Best validation score was **`val_f1_macro = 0.9059`** (at epoch 18).

---

## 5. Unified Optuna Hyperparameter Optimization

In `notebooks/02_hpo_optuna.ipynb`,a **Unified Search Space** was formulated that let Optuna optimize both data preprocessing and model architecture simultaneously across 25 trials:

### Search Space
- **Color Representation:** `grayscale` $\in \{\text{False (RGB)}, \text{True (Grayscale)}\}$
- **Augmentation Strategy:** `augmentation_type` $\in \{\text{"none"}, \text{"flips only"}, \text{"full"}\}$
- **Class Loss Weighting:** `use_class_weights` $\in \{\text{False}, \text{True}\}$
- **Model Capacity:** `base_channels` $\in \{32, 64\}$ (~102k to ~400k parameters)
- **Learning Rate:** `lr` $\in [1\times 10^{-4}, 1\times 10^{-2}]$ (log-uniform)
- **Regularization:** `weight_decay` $\in [1\times 10^{-6}, 1\times 10^{-2}]$, `dropout` $\in [0.10, 0.40]$
- **Optimization:** `optimizer_name` $\in \{\text{"adamw"}, \text{"adam"}, \text{"sgd"}\}$, `scheduler_name` $\in \{\text{"cosine"}, \text{"plateau"}\}$, `batch_size` $\in \{32, 64\}$

### Pruning & Persistence Strategy
- Monitored `val_f1_macro` at every epoch via custom `OptunaPruningCallback`
- Pruner: `MedianPruner(n_startup_trials=4, n_warmup_steps=3)` — terminated 12 underperforming trials around epoch 3, saving **$>60\%$ of GPU compute time**.
- Persistent SQLite storage: `logs/optuna_study_v2.db`.

### Discovered Winning Configuration (Trial #11)
- `grayscale`: `False` (RGB retained)
- `augmentation_type`: `"none"` (Clean inputs without continuous rotation/color jitter)
- `use_class_weights`: `True` (Inverse-frequency class-weighted CrossEntropy)
- `base_channels`: `64` (~400k parameters)
- `lr`: `0.000408` (~$4\times 10^{-4}$)
- `optimizer_name`: `"adamw"`
- `scheduler_name`: `"cosine"`
- `batch_size`: `32`
- `dropout`: `0.15`
- `weight_decay`: `0.000782`