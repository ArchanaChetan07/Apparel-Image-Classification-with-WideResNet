"""Dataset helpers — torchvision Fashion-MNIST only (no TensorFlow)."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),  # [0, 1], shape CxHxW
        ]
    )


def get_datasets(data_dir: str | Path = "data") -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = default_transform()
    train_ds = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_ds = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    return train_ds, test_ds


def get_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 0,
    subset_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_ds, test_ds = get_datasets(data_dir)

    if subset_size is not None:
        if subset_size < 1:
            raise ValueError("subset_size must be >= 1")
        train_ds = Subset(train_ds, list(range(min(subset_size, len(train_ds)))))
        test_n = min(max(subset_size // 5, 1), len(test_ds))
        test_ds = Subset(test_ds, list(range(test_n)))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def tensor_from_grayscale_array(image) -> torch.Tensor:
    """Accept HxW or HxWx1 numeric array / list; return 1x1x28x28 float tensor in [0, 1]."""
    import numpy as np

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"expected HxW grayscale image, got shape {arr.shape}")
    if arr.max() > 1.0:
        arr = arr / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # NCHW
    return tensor
