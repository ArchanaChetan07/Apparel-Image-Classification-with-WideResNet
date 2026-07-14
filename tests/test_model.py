"""Unit tests for WideResNet model."""

import pytest
import torch

from apparel_classifier.labels import CLASS_NAMES, NUM_CLASSES, class_name
from apparel_classifier.model import WideResNet, build_model


def test_class_catalog():
    assert NUM_CLASSES == 10
    assert len(CLASS_NAMES) == 10
    assert class_name(0) == "T-shirt/top"
    assert class_name(9) == "Ankle boot"
    with pytest.raises(ValueError):
        class_name(10)


def test_wideresnet_forward_default():
    model = build_model(narrow=False)
    x = torch.randn(2, 1, 28, 28)
    y = model(x)
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()


def test_wideresnet_forward_narrow():
    model = build_model(narrow=True)
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    assert y.shape == (4, NUM_CLASSES)


def test_reject_bad_input_rank():
    model = WideResNet(narrow=True)
    with pytest.raises(ValueError):
        model(torch.randn(1, 28, 28))


def test_narrow_has_fewer_params():
    full = build_model(narrow=False)
    slim = build_model(narrow=True)
    n_full = sum(p.numel() for p in full.parameters())
    n_slim = sum(p.numel() for p in slim.parameters())
    assert n_slim < n_full
    assert n_slim > 0
