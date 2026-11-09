# BloodMNIST Dataset

This directory contains the **BloodMNIST** dataset, used for **MLOps Lab 1**.

BloodMNIST is part of the standardized [MedMNIST v2](https://medmnist.com/) benchmark for 2D biomedical image classification.

---

## 1. Dataset Characteristics

| Property | Details |
| :--- | :--- |
| **Task** | Multi-class image classification (8 classes) |
| **Domain** | Biomedical / Hematology (Microscopic blood cell analysis) |
| **Image Resolution** | $28 \times 28$ pixels, RGB (3 channels) |
| **File Format** | NumPy zipped archive (`bloodmnist.npz`), ~15.4 MB |
| **Total Samples** | 17,092 images |
| **Benchmark Baseline** | ~78%–81% accuracy / macro-F1 |

---

## 2. Splits & Class Distribution

The dataset is pre-split into standardized training, validation, and testing sets:

* **Training Set:** 11,959 samples (~70%)
* **Validation Set:** 1,712 samples (~10%)
* **Test Set:** 3,421 samples (~20%)

### Class Breakdown (8 Classes)

| Label ID | Cell Type | Description & Diagnostic Relevance |
| :---: | :--- | :--- |
| `0` | **Basophil** | Rare granulocyte involved in severe inflammatory and allergic reactions. |
| `1` | **Eosinophil** | Granulocyte that attacks parasitic infections and modulates allergic responses. |
| `2` | **Erythroblast** | Nucleated immature precursor of red blood cells (erythrocytes). |
| `3` | **Immature Granulocyte** | Precursor cells (e.g. myelocytes, promyelocytes) indicating active infection or leukemia. |
| `4` | **Lymphocyte** | Key mononuclear white blood cell of the adaptive immune system (B-cells, T-cells). |
| `5` | **Monocyte** | Large circulating white blood cell that migrates to tissue and becomes a macrophage. |
| `6` | **Neutrophil** | Most abundant circulating white blood cell, first line of defense against pathogens. |
| `7` | **Platelet** | Small cell fragments essential for hemostasis and blood clot formation. |

---

## 3. How to Download

Activate the virtual environment and run the dedicated downloader script from within `156059/lab-1/`:

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Run downloader
python data/dataset_downloader.py
```

The script will automatically download `bloodmnist.npz` directly to this folder, verify its MD5 checksum (`7053d0359d879ad8a5505303e11de1dc`), and inspect the data arrays.

---

## 4. Citation & References

* **MedMNIST v2 Benchmark:**
  Yang, J. et al. (2023). *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data, 10(1), 41. [arXiv:2110.14763](https://arxiv.org/abs/2110.14763)
* **Original Blood Cell Data:**
  Acevedo, A. et al. (2020). *A dataset of microscopic peripheral blood cell images for development of computer vision algorithms.* Data in Brief, 30, 106040. [DOI: 10.1016/j.dib.2020.106040](https://doi.org/10.1016/j.dib.2020.106040)

