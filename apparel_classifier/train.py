"""Training loop with early stopping and checkpointing."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .data import get_dataloaders
from .labels import NUM_CLASSES
from .model import build_model


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 40
    lr: float = 0.01
    target_accuracy: float = 0.85
    patience: int = 2
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    narrow: bool = False
    subset_size: int | None = None
    seed: int = 42
    num_workers: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * images.size(0)
        n += images.size(0)
    return running / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    batches = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += int((preds == labels).sum().item())
        loss_sum += float(loss.item())
        batches += 1
    accuracy = correct / max(total, 1)
    avg_loss = loss_sum / max(batches, 1)
    return accuracy, avg_loss


def should_early_stop(history: list[float], target: float, patience: int) -> bool:
    if patience < 1:
        return False
    if len(history) < patience:
        return False
    return all(acc >= target for acc in history[-patience:])


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    metrics: dict,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "num_classes": getattr(model, "num_classes", NUM_CLASSES),
        "narrow": bool(getattr(model, "narrow", False)),
        "metrics": metrics,
        "config": asdict(config),
    }
    torch.save(payload, path)
    meta = path.with_suffix(".json")
    with meta.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if k != "model_state"}, f, indent=2)


def run_training(config: TrainConfig) -> dict:
    set_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        subset_size=config.subset_size,
    )

    model = build_model(NUM_CLASSES, narrow=config.narrow, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

    history: list[float] = []
    best_acc = -1.0
    artifacts = Path(config.artifacts_dir)
    ckpt_path = artifacts / ("model_narrow.pt" if config.narrow else "model.pt")
    total_time = 0.0

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, loss_fn, device)
        epoch_time = time.time() - t0
        total_time += epoch_time
        acc, val_loss = evaluate(model, test_loader, loss_fn, device)
        history.append(acc)
        images_per_sec = len(train_loader) * config.batch_size / max(epoch_time, 1e-9)
        print(
            f"Epoch {epoch:02d}: time={epoch_time:.3f}s total={total_time:.3f}s "
            f"img/s={images_per_sec:.1f} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={acc:.4f}"
        )

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(
                model,
                ckpt_path,
                metrics={
                    "val_accuracy": acc,
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "device": str(device),
                },
                config=config,
            )

        if should_early_stop(history, config.target_accuracy, config.patience):
            print(f"Early stopping after epoch {epoch}")
            break

    final_acc, final_loss = evaluate(model, test_loader, loss_fn, device)
    result = {
        "best_val_accuracy": best_acc,
        "final_val_accuracy": final_acc,
        "final_val_loss": final_loss,
        "epochs_ran": len(history),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "history": history,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train WideResNet on Fashion-MNIST")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--target-accuracy", type=float, default=0.85)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--artifacts-dir", default="artifacts")
    p.add_argument("--narrow", action="store_true", help="Use slim channels for smoke/CI")
    p.add_argument("--subset-size", type=int, default=None, help="Use first N train samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    ns = p.parse_args(argv)
    return TrainConfig(
        batch_size=ns.batch_size,
        epochs=ns.epochs,
        lr=ns.lr,
        target_accuracy=ns.target_accuracy,
        patience=ns.patience,
        data_dir=ns.data_dir,
        artifacts_dir=ns.artifacts_dir,
        narrow=ns.narrow,
        subset_size=ns.subset_size,
        seed=ns.seed,
        num_workers=ns.num_workers,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    run_training(config)


if __name__ == "__main__":
    main()
