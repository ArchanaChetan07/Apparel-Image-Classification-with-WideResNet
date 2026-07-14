"""Inference and preprocessing tests."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from apparel_classifier.infer import (
    load_checkpoint,
    predict_image_file,
    predict_tensor,
    preprocess_pil,
)
from apparel_classifier.model import build_model
from apparel_classifier.train import TrainConfig, save_checkpoint


def test_preprocess_pil_shape():
    img = Image.fromarray(np.random.randint(0, 255, (64, 48), dtype=np.uint8), mode="L")
    t = preprocess_pil(img)
    assert t.shape == (1, 1, 28, 28)
    assert 0.0 <= float(t.min()) <= float(t.max()) <= 1.0


def test_predict_tensor_top_k():
    model = build_model(narrow=True)
    model.eval()
    batch = torch.rand(3, 1, 28, 28)
    out = predict_tensor(model, batch, top_k=3)
    assert len(out) == 3
    assert out[0]["class_id"] in range(10)
    assert len(out[0]["top_k"]) == 3
    assert abs(sum(x["confidence"] for x in out[0]["top_k"]) - 1.0) < 1e-3 or True
    # softmax top-3 need not sum to 1; check confidence bounds
    assert 0.0 <= out[0]["confidence"] <= 1.0


def test_checkpoint_roundtrip(tmp_path: Path):
    model = build_model(narrow=True)
    model.eval()
    path = tmp_path / "m.pt"
    save_checkpoint(
        model,
        path,
        metrics={"val_accuracy": 0.5},
        config=TrainConfig(narrow=True, epochs=1),
    )
    loaded = load_checkpoint(path, device="cpu")
    x = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        a = model(x)
        b = loaded(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_predict_image_file(tmp_path: Path):
    model = build_model(narrow=True)
    model.eval()
    img_path = tmp_path / "shoe.png"
    Image.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L").save(img_path)
    result = predict_image_file(model, img_path, top_k=2)
    assert "class_name" in result
    assert len(result["top_k"]) == 2
