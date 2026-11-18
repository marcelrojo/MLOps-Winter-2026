"""Standalone BloodNet backbone and preprocessing constants for inference.

Decoupled from PyTorch Lightning, training losses, and metrics so the serving
service has a minimal runtime footprint.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

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


class BloodNet(nn.Module):
    """Convolutional neural network backbone for 28x28 BloodMNIST images."""

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
        """Return raw class logits for an input batch of shape (B, C, 28, 28)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        return self.classifier(x)


def preprocess_uint8_batch(
    images: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Convert a batch of NHWC uint8 images (B, H, W, C) to normalized NCHW float32.

    Matches Lab 1's BloodDataset + transforms.Normalize pipeline exactly:
    1. Permute (B, H, W, C) -> (B, C, H, W)
    2. Scale [0, 255] -> [0.0, 1.0] float32
    3. Channel-wise standardization: (x - mean) / std
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
    x = images.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32) / 255.0
    return (x - mean) / std
