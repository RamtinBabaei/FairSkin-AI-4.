from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Create shareable plots and a Markdown summary from a completed run.")
    p.add_argument("run_dir", help="e.g. artifacts/baseline")
    args = p.parse_args()
    run = Path(args.run_dir)
    report_path = run / "fairness_report.json"
    history_path = run / "history.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if history_path.exists():
        h = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(h["epoch"], h["train_loss"], marker="o", label="Train loss")
        ax.plot(h["epoch"], h["val_loss"], marker="o", label="Validation loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.set_title("Training history")
        ax.legend(); fig.tight_layout(); fig.savefig(run / "training_curves.png", dpi=180); plt.close(fig)

    groups = report.get("groups", {})
    if groups:
        df = pd.DataFrame([
            {"skin_group": g, **m} for g, m in groups.items()
        ]).sort_values("skin_group")
        df.to_csv(run / "subgroup_metrics.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(df["skin_group"], df["macro_f1"])
        ax.set_ylim(0, 1); ax.set_ylabel("Macro F1"); ax.set_xlabel("Fitzpatrick group")
        ax.set_title("Subgroup macro-F1")
        fig.tight_layout(); fig.savefig(run / "subgroup_macro_f1.png", dpi=180); plt.close(fig)

    lines = [
        "# FairSkin-AI run report", "",
        "## Overall metrics", "",
    ]
    for k, v in report.get("overall", {}).items():
        lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
    lines += ["", "## Subgroup disparity gaps", ""]
    for metric, obj in report.get("disparity", {}).items():
        lines.append(f"- **{metric} gap**: {obj['gap']:.4f} ({obj['max_group']} vs {obj['min_group']})")
    lines += ["", "> Research prototype only. These results are not evidence of clinical validity or safety.", ""]
    (run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report assets written to {run}")


if __name__ == "__main__":
    main()
