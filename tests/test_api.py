"""API contract tests."""

import io
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Allow API boot without a trained checkpoint for wiring tests.
os.environ["ALLOW_UNTRAINED"] = "1"
os.environ["MODEL_PATH"] = "artifacts/does_not_exist.pt"

from apparel_classifier import api as api_mod  # noqa: E402
from apparel_classifier.api import app  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_model_cache():
    api_mod.get_model_bundle.cache_clear()
    yield
    api_mod.get_model_bundle.cache_clear()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(size=(28, 28)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, size, dtype=np.uint8), mode="L").save(
        buf, format="PNG"
    )
    return buf.getvalue()


def test_root(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "apparel-classification"


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["num_classes"] == 10


def test_classes(client: TestClient):
    r = client.get("/classes")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 10


def test_predict(client: TestClient):
    r = client.post(
        "/predict",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        params={"top_k": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert "class_name" in body
    assert len(body["top_k"]) == 3


def test_predict_rejects_empty(client: TestClient):
    r = client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400
