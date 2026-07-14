#!/usr/bin/env python3
"""Measure WideResNet inference latency (CPU and GPU if available).

Prints a JSON report and optionally writes it under artifacts/.
Does not fabricate numbers — only records timed forward passes on this machine.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from apparel_classifier.model import WideResNet, build_model


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(
    model: WideResNet,
    batch: torch.Tensor,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch)
            _sync(device)
        ms: list[float] = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(batch)
            _sync(device)
            ms.append((time.perf_counter() - t0) * 1000.0)
    return ms


def summarize(times_ms: list[float]) -> dict:
    return {
        "n": len(times_ms),
        "mean_ms": round(statistics.mean(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(sorted(times_ms)[max(0, int(0.95 * (len(times_ms) - 1)))], 3),
        "min_ms": round(min(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
    }


def benchmark_device(
    device: torch.device,
    *,
    narrow: bool,
    warmup: int,
    repeats: int,
) -> dict:
    model = build_model(narrow=narrow, device=device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    single = torch.randn(1, 1, 28, 28)
    batch32 = torch.randn(32, 1, 28, 28)
    single_times = timed_forward(
        model, single, device=device, warmup=warmup, repeats=repeats
    )
    batch_times = timed_forward(
        model, batch32, device=device, warmup=warmup, repeats=repeats
    )
    return {
        "device": str(device),
        "narrow": narrow,
        "params": params,
        "single_image": summarize(single_times),
        "batch_32": {
            **summarize(batch_times),
            "per_image_mean_ms": round(statistics.mean(batch_times) / 32.0, 3),
            "per_image_median_ms": round(statistics.median(batch_times) / 32.0, 3),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("artifacts/latency_benchmark.json"))
    p.add_argument(
        "--narrow-only",
        action="store_true",
        help="Skip the full 17.1M model (faster CI)",
    )
    args = p.parse_args()

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    results = {
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "runs": [],
    }

    for device in devices:
        for narrow in ([True] if args.narrow_only else [False, True]):
            run = benchmark_device(
                device, narrow=narrow, warmup=args.warmup, repeats=args.repeats
            )
            results["runs"].append(run)
            label = "narrow" if narrow else "full"
            s = run["single_image"]
            b = run["batch_32"]
            print(
                f"[{run['device']}|{label}] single median={s['median_ms']}ms "
                f"p95={s['p95_ms']}ms | batch32 median={b['median_ms']}ms "
                f"({b['per_image_median_ms']}ms/img)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
