"""Latency ceiling checks — measures real forward-pass times on this machine."""

from __future__ import annotations

import statistics
import time

import pytest
import torch

from apparel_classifier.model import build_model


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_ms(
    model, batch: torch.Tensor, device: torch.device, *, warmup: int = 5, repeats: int = 15
) -> float:
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)
            _sync(device)
        times = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(batch)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


# Generous ceilings so CI runners / slow machines stay stable; measured numbers
# for README come from scripts/benchmark_latency.py, not these bounds.
CPU_SINGLE_FULL_CEILING_MS = 500.0
CPU_BATCH32_FULL_CEILING_MS = 5000.0


@pytest.mark.parametrize(
    "narrow,batch_size,ceiling_ms",
    [
        (False, 1, CPU_SINGLE_FULL_CEILING_MS),
        (False, 32, CPU_BATCH32_FULL_CEILING_MS),
        (True, 1, CPU_SINGLE_FULL_CEILING_MS),
    ],
)
def test_cpu_inference_under_ceiling(narrow: bool, batch_size: int, ceiling_ms: float):
    device = torch.device("cpu")
    model = build_model(narrow=narrow, device=device)
    batch = torch.randn(batch_size, 1, 28, 28)
    med = _median_ms(model, batch, device)
    assert med < ceiling_ms, (
        f"CPU {'narrow' if narrow else 'full'} batch={batch_size} "
        f"median {med:.2f}ms exceeds ceiling {ceiling_ms}ms"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_single_image_under_ceiling():
    device = torch.device("cuda")
    model = build_model(narrow=False, device=device)
    batch = torch.randn(1, 1, 28, 28)
    # GPU should be fast; keep a loose 100ms ceiling for CI variance
    med = _median_ms(model, batch, device)
    assert med < 100.0, f"GPU full single-image median {med:.2f}ms exceeds 100ms"
