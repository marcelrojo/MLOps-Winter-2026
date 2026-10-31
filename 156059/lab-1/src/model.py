"""Model architecture and PyTorch Lightning module for BloodMNIST classification.

Provides:
- BloodNet: Modular CNN backbone with configurable base channels, dropout,
  and input channels (1 for grayscale, 3 for RGB).
- BloodClassifier: LightningModule managing forward pass, training/validation/test
  loops, Macro-F1 & Accuracy metric computation, optional class-weighted CrossEntropyLoss,
  and optimizer/scheduler configuration.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

logger = logging.getLogger(__name__)


class BloodNet(nn.Module):
    """Convolutional neural network backbone tailored for 28x28 cell images.

    Architecture:
    - 3 Conv-BN-ReLU-MaxPool blocks with doubling channel widths.
    - Global Adaptive Average Pooling (collapses spatial dimensions to 1x1).
    - Fully connected classification head with dropout regularization.

    Args:
        in_channels: Number of input image channels (1 for grayscale, 3 for RGB).
        num_classes: Number of target cell classes (default: 8).
        base_channels: Number of filters in the initial convolution block.
        dropout: Dropout probability applied in the classification head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        base_channels: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_rate = dropout

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c3, c2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(c2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass extracting convolutional features and returning class logits."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits


class BloodClassifier(pl.LightningModule):
    """PyTorch LightningModule for BloodMNIST multi-class cell classification.

    Features:
    - Encapsulates BloodNet backbone with configurable hyperparameters.
    - Supports weighted CrossEntropyLoss via persistent class weight buffer.
    - Computes Macro-F1 (primary metric) and Accuracy across all stages.
    - Configurable optimizers (AdamW, Adam, SGD) and schedulers (Cosine, Plateau, None).
    - Automated hyperparameter logging via `save_hyperparameters()`.

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB).
        num_classes: Number of target classes (8 for BloodMNIST).
        base_channels: Base filter count for BloodNet.
        dropout: Dropout rate in classification head.
        lr: Initial learning rate.
        weight_decay: L2 regularization factor.
        optimizer_name: Name of optimizer ('adamw', 'adam', 'sgd').
        scheduler_name: Name of learning rate scheduler ('cosine', 'plateau', or None).
        max_epochs: Total training epochs (used for CosineAnnealingLR horizon).
        class_weights: Optional 1D tensor or sequence of inverse class frequencies.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        base_channels: int = 32,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        scheduler_name: str | None = "cosine",
        max_epochs: int = 20,
        class_weights: Tensor | Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        serializable_weights = (
            [float(w) for w in class_weights] if class_weights is not None else None
        )
        self.save_hyperparameters(ignore=["class_weights"])
        self.hparams["class_weights"] = serializable_weights

        self.net = BloodNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            dropout=dropout,
        )

        if class_weights is not None:
            if not isinstance(class_weights, Tensor):
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            else:
                weight_tensor = class_weights.detach().clone().float()
            self.register_buffer("class_weights", weight_tensor)
        else:
            self.register_buffer("class_weights", None)

        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)

        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward inference through the underlying convolutional backbone."""
        return self.net(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Execute a single training step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.train_f1.update(logits, y)
        self.train_acc.update(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1_macro", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Execute a single validation step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_f1.update(logits, y)
        self.val_acc.update(logits, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1_macro", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Execute a single test step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.test_f1.update(logits, y)
        self.test_acc.update(logits, y)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_f1_macro", self.test_f1, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> Any:
        """Configure optimizer and learning rate scheduler based on hyperparameters."""
        opt_name = self.hparams.optimizer_name.lower()
        lr = self.hparams.lr
        weight_decay = self.hparams.weight_decay

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: '{opt_name}'. Choose 'adamw', 'adam', or 'sgd'."
            )

        sched_name = self.hparams.scheduler_name
        if sched_name is None or str(sched_name).lower() == "none":
            return optimizer

        sched_name = str(sched_name).lower()
        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        elif sched_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(
                f"Unsupported scheduler: '{sched_name}'. Choose 'cosine', 'plateau', or None."
            )
