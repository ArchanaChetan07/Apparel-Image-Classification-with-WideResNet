"""Inference helpers and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .labels import CLASS_NAMES, NUM_CLASSES, class_name
from .model import WideResNet, build_model


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> WideResNet:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(payload, dict) and "model_state" in payload:
        narrow = bool(payload.get("narrow", False))
        num_classes = int(payload.get("num_classes", NUM_CLASSES))
        model = build_model(num_classes=num_classes, narrow=narrow, device=map_location)
        model.load_state_dict(payload["model_state"])
    else:
        # Raw state_dict fallback (legacy notebook torch.save(state_dict))
        model = build_model(device=map_location)
        model.load_state_dict(payload)

    model.eval()
    return model


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    """Convert any PIL image to NCHW float tensor matching Fashion-MNIST."""
    gray = image.convert("L").resize((28, 28))
    arr = torch.from_numpy(np.asarray(gray, dtype=np.float32) / 255.0)
    return arr.unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_tensor(
    model: WideResNet,
    batch: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    top_k: int = 3,
) -> list[dict]:
    if top_k < 1 or top_k > NUM_CLASSES:
        raise ValueError(f"top_k must be in [1, {NUM_CLASSES}]")

    device = torch.device(device or next(model.parameters()).device)
    batch = batch.to(device)
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)

    results = []
    for i in range(batch.size(0)):
        top_probs, top_idx = probs[i].topk(top_k)
        results.append(
            {
                "class_id": int(pred[i].item()),
                "class_name": class_name(int(pred[i].item())),
                "confidence": float(conf[i].item()),
                "top_k": [
                    {
                        "class_id": int(idx.item()),
                        "class_name": CLASS_NAMES[int(idx.item())],
                        "confidence": float(p.item()),
                    }
                    for p, idx in zip(top_probs, top_idx)
                ],
            }
        )
    return results


def predict_image_file(
    model: WideResNet,
    image_path: str | Path,
    *,
    device: torch.device | str | None = None,
    top_k: int = 3,
) -> dict:
    image = Image.open(image_path)
    tensor = preprocess_pil(image)
    return predict_tensor(model, tensor, device=device, top_k=top_k)[0]
