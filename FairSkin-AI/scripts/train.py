from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairskin_ai.checkpoint import save_checkpoint
from fairskin_ai.data import (
    DermatologyDataset,
    build_transforms,
    class_weights,
    intersectional_sampler,
    make_label_map,
    split_metadata,
    validate_metadata,
)
from fairskin_ai.engine import predict, train_model, write_evaluation
from fairskin_ai.model import build_model
from fairskin_ai.utils import ensure_dir, load_yaml, resolve_device, save_json, set_seed


def main():
    p = argparse.ArgumentParser(description="Train FairSkin-AI.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--csv", default=None)
    p.add_argument("--image-root", default=None)
    p.add_argument("--mode", choices=["baseline", "class_weighted", "intersectional_sampler"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.csv: cfg["data"]["csv"] = args.csv
    if args.image_root: cfg["data"]["image_root"] = args.image_root
    if args.mode: cfg["mitigation"]["mode"] = args.mode
    if args.model: cfg["model"]["name"] = args.model
    if args.epochs is not None: cfg["training"]["epochs"] = args.epochs
    if args.output_dir: cfg["output"]["dir"] = args.output_dir

    seed = int(cfg["seed"])
    set_seed(seed)
    device = resolve_device(str(cfg.get("device", "auto")))
    print(f"Device: {device}")

    df = pd.read_csv(cfg["data"]["csv"])
    validate_metadata(df)
    train_df, val_df, test_df = split_metadata(
        df,
        float(cfg["data"]["val_size"]),
        float(cfg["data"]["test_size"]),
        seed,
        str(cfg["fairness"]["skin_grouping"]),
    )
    label_map = make_label_map(train_df, val_df, test_df)
    out = ensure_dir(cfg["output"]["dir"])
    split_dir = ensure_dir(out / "splits")
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    save_json(label_map.class_to_idx, out / "class_to_idx.json")
    save_json(cfg, out / "resolved_config.json")

    image_size = int(cfg["data"]["image_size"])
    root = cfg["data"]["image_root"]
    train_ds = DermatologyDataset(train_df, root, label_map, build_transforms(image_size, True))
    val_ds = DermatologyDataset(val_df, root, label_map, build_transforms(image_size, False))
    test_ds = DermatologyDataset(test_df, root, label_map, build_transforms(image_size, False))

    sampler = None
    shuffle = True
    mode = str(cfg["mitigation"]["mode"])
    if mode == "intersectional_sampler":
        sampler = intersectional_sampler(train_df)
        shuffle = False

    batch_size = int(cfg["data"]["batch_size"])
    workers = int(cfg["data"].get("num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = build_model(
        str(cfg["model"]["name"]),
        len(label_map.class_to_idx),
        bool(cfg["model"]["pretrained"]),
        float(cfg["model"]["dropout"]),
    ).to(device)

    criterion_weights = class_weights(train_df, label_map).to(device) if mode == "class_weighted" else None
    criterion = nn.CrossEntropyLoss(
        weight=criterion_weights,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    model, history, seconds = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        int(cfg["training"]["epochs"]),
        int(cfg["training"]["patience"]),
    )
    history.to_csv(out / "history.csv", index=False)
    save_checkpoint(
        out / "best_model.pt",
        model,
        str(cfg["model"]["name"]),
        label_map.class_to_idx,
        image_size,
        float(cfg["model"]["dropout"]),
    )

    preds = predict(model, test_loader, device)
    report = write_evaluation(preds, out, int(cfg["fairness"].get("bootstrap_samples", 200)))
    summary = {
        "mode": mode,
        "model": cfg["model"]["name"],
        "device": str(device),
        "train_n": len(train_df), "val_n": len(val_df), "test_n": len(test_df),
        "seconds": seconds,
        "overall": report["overall"],
        "disparity": report["disparity"],
    }
    save_json(summary, out / "run_summary.json")
    print("\nTraining complete.")
    print(f"Checkpoint: {out / 'best_model.pt'}")
    print(f"Fairness report: {out / 'fairness_report.json'}")


if __name__ == "__main__":
    main()
