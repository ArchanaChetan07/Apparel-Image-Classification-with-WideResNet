# Production inference image for Apparel WideResNet classifier
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/artifacts/model.pt \
    ALLOW_UNTRAINED=0

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY apparel_classifier ./apparel_classifier
COPY artifacts ./artifacts

RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && pip install -e .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "apparel_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
