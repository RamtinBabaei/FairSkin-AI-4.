from __future__ import annotations

from pathlib import Path

import torch

from .model import build_model


def save_checkpoint(path, model, model_name, class_to_idx, image_size, dropout=0.25):
    payload = {
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "class_to_idx": class_to_idx,
        "image_size": int(image_size),
        "dropout": float(dropout),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(
        payload["model_name"],
        num_classes=len(payload["class_to_idx"]),
        pretrained=False,
        dropout=float(payload.get("dropout", 0.25)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
