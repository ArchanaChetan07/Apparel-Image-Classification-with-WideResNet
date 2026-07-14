"""Tests for training helpers and short smoke train."""

from pathlib import Path

import torch

from apparel_classifier.train import TrainConfig, run_training, should_early_stop


def test_early_stop_logic():
    assert should_early_stop([0.8, 0.86, 0.87], target=0.85, patience=2) is True
    assert should_early_stop([0.8, 0.86], target=0.85, patience=2) is False
    assert should_early_stop([0.9], target=0.85, patience=2) is False
    assert should_early_stop([0.9, 0.9], target=0.85, patience=0) is False


def test_smoke_train_narrow(tmp_path: Path):
    """One-epoch narrow train on a tiny subset — validates end-to-end wiring."""
    cfg = TrainConfig(
        batch_size=16,
        epochs=1,
        lr=0.05,
        target_accuracy=0.99,
        patience=5,
        data_dir=str(tmp_path / "data"),
        artifacts_dir=str(tmp_path / "artifacts"),
        narrow=True,
        subset_size=64,
        seed=0,
    )
    result = run_training(cfg)
    assert result["epochs_ran"] == 1
    ckpt = Path(result["checkpoint"])
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state" in payload
    assert payload["narrow"] is True
