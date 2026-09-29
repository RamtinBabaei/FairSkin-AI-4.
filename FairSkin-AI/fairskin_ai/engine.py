from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .fairness import bootstrap_group_metric, group_metrics
from .utils import save_json


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total = 0
    correct = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * targets.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += targets.size(0)
    if total == 0:
        raise ValueError("DataLoader is empty.")
    return total_loss / total, correct / total


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs: int,
    patience: int,
):
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    history = []
    stalled = 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device, None)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_accuracy": tr_acc,
                "val_loss": va_loss,
                "val_accuracy": va_acc,
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )
        if va_loss < best_loss - 1e-5:
            best_loss = va_loss
            best_state = copy.deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), time.time() - start


@torch.no_grad()
def predict(model, loader, device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        for i in range(targets.size(0)):
            rows.append(
                {
                    "true_idx": int(targets[i].cpu()),
                    "pred_idx": int(pred[i].cpu()),
                    "confidence": float(conf[i].cpu()),
                    "fitzpatrick": int(batch["fitzpatrick"][i]),
                    "skin_group": str(batch["skin_group"][i]),
                    "image_path": str(batch["image_path"][i]),
                }
            )
    return pd.DataFrame(rows)


def write_evaluation(predictions: pd.DataFrame, output_dir: str | Path, bootstrap_samples: int = 200) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = group_metrics(predictions.true_idx, predictions.pred_idx, predictions.skin_group)
    report["bootstrap_macro_f1"] = bootstrap_group_metric(
        predictions.true_idx,
        predictions.pred_idx,
        predictions.skin_group,
        metric="macro_f1",
        n_bootstrap=bootstrap_samples,
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_json(report, output_dir / "fairness_report.json")
    return report
