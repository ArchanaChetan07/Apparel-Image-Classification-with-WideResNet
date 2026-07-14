"""WideResNet architecture for Fashion-MNIST (grayscale 28x28)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .labels import NUM_CLASSES


class CBRBlock(nn.Module):
    """Conv → BatchNorm → ReLU."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=1,
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbr(x)


class ConvBlock(nn.Module):
    """Residual block with optional 1x1 channel projection."""

    def __init__(
        self, input_channels: int, output_channels: int, scale_input: bool
    ) -> None:
        super().__init__()
        self.scale_input = scale_input
        self.scale = (
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                padding="same",
            )
            if scale_input
            else None
        )
        self.layer1 = CBRBlock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = CBRBlock(output_channels, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale is not None:
            residual = self.scale(residual)
        return out + residual


class WideResNet(nn.Module):
    """Wide residual CNN used for the published 90.5% Fashion-MNIST result."""

    # Channel schedule matching the original notebook (portfolio baseline).
    DEFAULT_CHANNELS = (1, 16, 160, 320, 640)
    # Narrow schedule for CI / smoke trains (same topology, far fewer params).
    NARROW_CHANNELS = (1, 8, 16, 32, 64)

    def __init__(self, num_classes: int = NUM_CLASSES, narrow: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        n = list(self.NARROW_CHANNELS if narrow else self.DEFAULT_CHANNELS)
        self.narrow = narrow
        self.num_classes = num_classes

        self.input_block = CBRBlock(n[0], n[1])
        self.block1 = ConvBlock(n[1], n[2], True)
        self.block2 = ConvBlock(n[2], n[2], False)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = ConvBlock(n[2], n[3], True)
        self.block4 = ConvBlock(n[3], n[3], False)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = ConvBlock(n[3], n[4], True)
        self.block6 = ConvBlock(n[4], n[4], False)
        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(n[4], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected NCHW tensor, got shape {tuple(x.shape)}")
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        return self.fc(out)


def build_model(
    num_classes: int = NUM_CLASSES,
    *,
    narrow: bool = False,
    device: torch.device | str | None = None,
) -> WideResNet:
    model = WideResNet(num_classes=num_classes, narrow=narrow)
    if device is not None:
        model = model.to(device)
    return model
