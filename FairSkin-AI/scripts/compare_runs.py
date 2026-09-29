from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def flatten(path: str) -> dict:
    p = Path(path)
    data = json.loads((p / "fairness_report.json").read_text(encoding="utf-8"))
    row = {"run": p.name}
    for k, v in data["overall"].items(): row[f"overall_{k}"] = v
    for metric, obj in data.get("disparity", {}).items(): row[f"gap_{metric}"] = obj["gap"]
    for group, obj in data.get("groups", {}).items():
        row[f"{group}_macro_f1"] = obj["macro_f1"]
        row[f"{group}_balanced_accuracy"] = obj["balanced_accuracy"]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="artifacts/run_comparison.csv")
    args = p.parse_args()
    df = pd.DataFrame([flatten(x) for x in args.runs])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
