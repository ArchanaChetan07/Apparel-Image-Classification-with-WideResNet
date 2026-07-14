"""FastAPI inference service for apparel classification."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import __version__
from .infer import load_checkpoint, predict_tensor, preprocess_pil
from .labels import CLASS_NAMES, NUM_CLASSES
from .model import build_model

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
ALLOW_UNTRAINED = os.getenv("ALLOW_UNTRAINED", "0") == "1"


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_path: str
    num_classes: int
    device: str


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: list[dict]


def _resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_model_bundle():
    device = _resolve_device()
    path = Path(DEFAULT_MODEL_PATH)
    if path.exists():
        model = load_checkpoint(path, device=device)
        return model, device, str(path), True

    if not ALLOW_UNTRAINED:
        raise RuntimeError(
            f"No checkpoint at {path}. Train first "
            f"(python -m apparel_classifier.train) or set ALLOW_UNTRAINED=1."
        )

    # Dev-only fallback: random weights so the API can boot for wiring tests.
    model = build_model(narrow=True, device=device)
    model.eval()
    return model, device, str(path), False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager load on startup when possible; surface a clear error otherwise.
    try:
        get_model_bundle()
    except RuntimeError as exc:
        app.state.model_error = str(exc)
    else:
        app.state.model_error = None
    yield


app = FastAPI(
    title="Apparel Classification API",
    description="WideResNet Fashion-MNIST inference service",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        _model, device, path, loaded = get_model_bundle()
        return HealthResponse(
            status="ok" if loaded else "degraded",
            version=__version__,
            model_loaded=loaded,
            model_path=path,
            num_classes=NUM_CLASSES,
            device=device,
        )
    except RuntimeError:
        return HealthResponse(
            status="degraded",
            version=__version__,
            model_loaded=False,
            model_path=DEFAULT_MODEL_PATH,
            num_classes=NUM_CLASSES,
            device=_resolve_device(),
        )


@app.get("/classes")
def classes() -> dict:
    return {"num_classes": NUM_CLASSES, "classes": list(CLASS_NAMES)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), top_k: int = 3) -> PredictResponse:
    if top_k < 1 or top_k > NUM_CLASSES:
        raise HTTPException(status_code=400, detail=f"top_k must be 1..{NUM_CLASSES}")

    try:
        model, device, _, _ = get_model_bundle()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    try:
        image = Image.open(io.BytesIO(raw))
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="invalid image file") from exc

    tensor = preprocess_pil(image)
    result = predict_tensor(model, tensor, device=device, top_k=top_k)[0]
    return PredictResponse(**result)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "service": "apparel-classification",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }
    )
