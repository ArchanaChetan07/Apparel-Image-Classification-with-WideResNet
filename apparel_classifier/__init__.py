"""Apparel image classification with WideResNet (Fashion-MNIST)."""

from .labels import CLASS_NAMES, NUM_CLASSES
from .model import WideResNet, build_model

__version__ = "1.0.0"
__all__ = ["CLASS_NAMES", "NUM_CLASSES", "WideResNet", "build_model", "__version__"]
