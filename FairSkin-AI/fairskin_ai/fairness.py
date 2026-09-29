from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> dict[str, float]:
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    # Balanced accuracy is macro recall over a fixed label set. Using recall_score
    # directly avoids sklearn warnings in small bootstrap resamples.
    balanced = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def group_metrics(y_true: Iterable[int], y_pred: Iterable[int], groups: Iterable[str]) -> dict:
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    gr = np.asarray(list(groups), dtype=object)
    if not (len(yt) == len(yp) == len(gr)):
        raise ValueError("y_true, y_pred and groups must have the same length.")
    labels = np.unique(np.concatenate([yt, yp]))
    result = {"overall": _metric_dict(yt, yp, labels), "groups": {}}
    for group in sorted(set(gr.tolist())):
        mask = gr == group
        result["groups"][str(group)] = _metric_dict(yt[mask], yp[mask], labels)
    disparity: dict[str, dict[str, float | str]] = {}
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_recall"]:
        values = {g: m[metric] for g, m in result["groups"].items() if m["n"] > 0}
        if values:
            max_g = max(values, key=values.get)
            min_g = min(values, key=values.get)
            disparity[metric] = {
                "max_group": max_g,
                "min_group": min_g,
                "gap": float(values[max_g] - values[min_g]),
            }
    result["disparity"] = disparity
    return result


def bootstrap_group_metric(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    groups: Iterable[str],
    metric: str = "macro_f1",
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    gr = np.asarray(list(groups), dtype=object)
    rng = np.random.default_rng(seed)
    labels = np.unique(np.concatenate([yt, yp]))
    out: dict[str, dict[str, float]] = {}
    for group in sorted(set(gr.tolist())):
        idx = np.where(gr == group)[0]
        if len(idx) < 2:
            continue
        vals: list[float] = []
        for _ in range(n_bootstrap):
            sample = rng.choice(idx, size=len(idx), replace=True)
            metrics = _metric_dict(yt[sample], yp[sample], labels)
            vals.append(float(metrics[metric]))
        out[str(group)] = {
            "mean": float(np.mean(vals)),
            "ci_low": float(np.quantile(vals, 0.025)),
            "ci_high": float(np.quantile(vals, 0.975)),
        }
    return out
