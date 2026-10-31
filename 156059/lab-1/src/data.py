"""Data pipeline for BloodMNIST dataset.

Provides:
- BloodDataset: PyTorch Dataset with in-memory caching, grayscale/RGB support,
  and customizable augmentations.
- BloodDataModule: PyTorch Lightning DataModule handling train/val/test splits,
  exact training-set standardization statistics, class weights for imbalance,
  and DataLoader creation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

CLASS_NAMES: list[str] = [
    "Basophil",
    "Eosinophil",
    "Erythroblast",
    "Immature Granulocyte",
    "Lymphocyte",
    "Monocyte",
    "Neutrophil",
    "Platelet",
]


def resolve_default_data_path() -> Path:
    """Locate the default bloodmnist.npz archive relative to this file."""
    candidate = Path(__file__).resolve().parent.parent / "data" / "bloodmnist.npz"
    return candidate


class BloodDataset(Dataset):
    """PyTorch Dataset for BloodMNIST stored fully in-memory.

    Args:
        images: NumPy array of shape (N, 28, 28, 3) with uint8 values in [0, 255].
        labels: NumPy array of shape (N, 1) or (N,) with integer class labels.
        transform: Optional torchvision transform (e.g. geometric augmentations).
        normalize: Optional torchvision transform (e.g. transforms.Normalize).
        grayscale: If True, converts RGB to single-channel grayscale (1, 28, 28).
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose | None = None,
        normalize: transforms.Normalize | None = None,
        grayscale: bool = False,
    ) -> None:
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatch: {images.shape[0]} images vs {labels.shape[0]} labels"
            )

        self.transform = transform
        self.normalize = normalize
        self.grayscale = grayscale

        img_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().float() / 255.0

        if self.grayscale:
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1)
            img_tensor = (img_tensor * weights).sum(dim=1, keepdim=True)  # (N, 1, 28, 28)

        self.images: Tensor = img_tensor
        self.labels: Tensor = torch.from_numpy(labels).flatten().long()

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.normalize is not None:
            x = self.normalize(x)

        return x, y


class BloodDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for BloodMNIST.

    Features:
    - Preloads train/val/test splits entirely into memory for high throughput.
    - Computes exact channel mean & std from the training split for normalization.
    - Exposes inverse-frequency `class_weights` for weighted CrossEntropyLoss.
    - Toggleable augmentations via `use_augmentations` (flips, rotation, mild jitter).
    - Toggleable `grayscale` mode (1-channel vs 3-channel RGB).

    Args:
        data_path: Path to `bloodmnist.npz`. If None, automatically resolved.
        batch_size: Batch size for training.
        val_batch_size: Batch size for validation/test (defaults to `batch_size`).
        use_augmentations: Whether to apply data augmentations to training set.
        grayscale: If True, convert images to 1 channel; otherwise 3 channels.
        num_workers: Number of DataLoader subprocesses (0 recommended for RAM datasets).
        pin_memory: Whether DataLoader pins memory to CUDA page-locked memory.
        drop_last: Whether to drop the last incomplete training batch.
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        batch_size: int = 64,
        val_batch_size: int | None = None,
        use_augmentations: bool = True,
        grayscale: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path) if data_path else resolve_default_data_path()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        self.use_augmentations = use_augmentations
        self.grayscale = grayscale
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.drop_last = drop_last

        self.num_classes: int = len(CLASS_NAMES)
        self.class_names: list[str] = list(CLASS_NAMES)

        self.train_dataset: BloodDataset | None = None
        self.val_dataset: BloodDataset | None = None
        self.test_dataset: BloodDataset | None = None

        self._train_mean: tuple[float, ...] | None = None
        self._train_std: tuple[float, ...] | None = None
        self._class_counts: Tensor | None = None
        self._class_weights: Tensor | None = None

    @property
    def in_channels(self) -> int:
        """Number of image channels (1 if grayscale, else 3)."""
        return 1 if self.grayscale else 3

    @property
    def train_mean(self) -> tuple[float, ...]:
        """Channel-wise mean of training split."""
        if self._train_mean is None:
            raise RuntimeError("DataModule has not been setup yet. Call setup() first.")
        return self._train_mean

    @property
    def train_std(self) -> tuple[float, ...]:
        """Channel-wise standard deviation of training split."""
        if self._train_std is None:
            raise RuntimeError("DataModule has not been setup yet. Call setup() first.")
        return self._train_std

    @property
    def class_counts(self) -> Tensor:
        """Sample count per class in the training split."""
        if self._class_counts is None:
            raise RuntimeError("DataModule has not been setup yet. Call setup() first.")
        return self._class_counts

    @property
    def class_weights(self) -> Tensor:
        """Normalized inverse-frequency class weights for loss balancing."""
        if self._class_weights is None:
            raise RuntimeError("DataModule has not been setup yet. Call setup() first.")
        return self._class_weights

    def prepare_data(self) -> None:
        """Verify that the dataset archive exists."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"BloodMNIST archive not found at '{self.data_path}'. "
                "Please run `python data/dataset_downloader.py` first."
            )

    def _build_transforms(
        self, train_mean: Sequence[float], train_std: Sequence[float]
    ) -> tuple[transforms.Compose | None, transforms.Normalize]:
        """Construct augmentation and normalization transforms."""
        normalizer = transforms.Normalize(mean=list(train_mean), std=list(train_std))

        if not self.use_augmentations:
            return None, normalizer

        aug_list: list[torch.nn.Module] = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
        ]

        # Subtle color jitter only makes sense for RGB images
        if not self.grayscale:
            aug_list.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))

        train_transform = transforms.Compose(aug_list)
        return train_transform, normalizer

    def setup(self, stage: str | None = None) -> None:
        """Load arrays, calculate train statistics, and instantiate datasets."""
        if self.train_dataset is not None:
            return  

        self.prepare_data()
        data = np.load(self.data_path)

        train_images = data["train_images"]
        train_labels = data["train_labels"].flatten()
        val_images = data["val_images"]
        val_labels = data["val_labels"].flatten()
        test_images = data["test_images"]
        test_labels = data["test_labels"].flatten()

        counts = np.bincount(train_labels, minlength=self.num_classes)
        self._class_counts = torch.from_numpy(counts).long()

        total_samples = float(len(train_labels))
        
        weights = total_samples / (self.num_classes * counts.astype(np.float32))
        weights = weights / weights.mean()
        self._class_weights = torch.from_numpy(weights).float()

        train_float = train_images.astype(np.float32) / 255.0
        if self.grayscale:
            luma = (
                0.299 * train_float[..., 0]
                + 0.587 * train_float[..., 1]
                + 0.114 * train_float[..., 2]
            )
            self._train_mean = (float(luma.mean()),)
            self._train_std = (float(luma.std()),)
        else:
            mean_vals = train_float.mean(axis=(0, 1, 2))
            std_vals = train_float.std(axis=(0, 1, 2))
            self._train_mean = tuple(float(m) for m in mean_vals)
            self._train_std = tuple(float(s) for s in std_vals)

        train_transform, normalizer = self._build_transforms(
            self._train_mean, self._train_std
        )

        self.train_dataset = BloodDataset(
            images=train_images,
            labels=train_labels,
            transform=train_transform,
            normalize=normalizer,
            grayscale=self.grayscale,
        )
        self.val_dataset = BloodDataset(
            images=val_images,
            labels=val_labels,
            transform=None, 
            normalize=normalizer,
            grayscale=self.grayscale,
        )
        self.test_dataset = BloodDataset(
            images=test_images,
            labels=test_labels,
            transform=None,  
            normalize=normalizer,
            grayscale=self.grayscale,
        )

        logger.info(
            "BloodDataModule setup complete. "
            f"Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, "
            f"Test: {len(self.test_dataset)}, "
            f"Channels: {self.in_channels}, "
            f"Mean: {self._train_mean}, Std: {self._train_std}"
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

