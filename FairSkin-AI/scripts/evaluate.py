from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairskin_ai.checkpoint import load_checkpoint
from fairskin_ai.data import DermatologyDataset, LabelMap, build_transforms
from fairskin_ai.engine import predict, write_evaluation
from fairskin_ai.utils import resolve_device


def main():
    p = argparse.ArgumentParser(description="Evaluate a FairSkin-AI checkpoint on standardized metadata.")
    p.add_argument("--checkpoint", default="artifacts/best_model.pt")
    p.add_argument("--csv", default="artifacts/splits/test.csv")
    p.add_argument("--image-root", required=True)
    p.add_argument("--output-dir", default="artifacts/evaluation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap", type=int, default=200)
    args = p.parse_args()

    device = resolve_device("auto")
    model, payload = load_checkpoint(args.checkpoint, device)
    df = pd.read_csv(args.csv)
    labels = LabelMap({str(k): int(v) for k, v in payload["class_to_idx"].items()})
    ds = DermatologyDataset(df, args.image_root, labels, build_transforms(int(payload["image_size"]), False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds = predict(model, loader, device)
    report = write_evaluation(preds, args.output_dir, args.bootstrap)
    print(report["overall"])
    print("Disparity:", report["disparity"])


if __name__ == "__main__":
    main()
