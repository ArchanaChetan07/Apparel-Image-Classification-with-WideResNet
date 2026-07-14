"""CLI entrypoints."""

from __future__ import annotations

import argparse
import json
import sys

from .infer import load_checkpoint, predict_image_file
from .train import main as train_main


def cmd_predict(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict apparel class for an image")
    p.add_argument("image", help="Path to input image")
    p.add_argument("--model", default="artifacts/model.pt", help="Checkpoint path")
    p.add_argument("--top-k", type=int, default=3)
    ns = p.parse_args(argv)
    model = load_checkpoint(ns.model)
    result = predict_image_file(model, ns.image, top_k=ns.top_k)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m apparel_classifier.cli [train|predict|serve] ...")
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        train_main(rest)
        return 0
    if cmd == "predict":
        return cmd_predict(rest)
    if cmd == "serve":
        import uvicorn

        host = "0.0.0.0"
        port = 8000
        # lightweight flag parse
        if "--port" in rest:
            i = rest.index("--port")
            port = int(rest[i + 1])
        if "--host" in rest:
            i = rest.index("--host")
            host = rest[i + 1]
        uvicorn.run("apparel_classifier.api:app", host=host, port=port, reload=False)
        return 0
    print(f"Unknown command: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
