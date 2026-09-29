from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description="Create a small synthetic dataset to smoke-test the full pipeline.")
    p.add_argument("--out", default="data/synthetic_demo")
    p.add_argument("--samples", type=int, default=180)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    labels = ["benign", "malignant", "non-neoplastic"]
    rows = []

    # Synthetic geometric images are for software validation only, not scientific results.
    for i in range(args.samples):
        label = labels[i % len(labels)]
        fitz = int((i % 6) + 1)
        base = int(235 - (fitz - 1) * 24)
        noise = rng.integers(-8, 9, size=3)
        bg = tuple(int(np.clip(base + n, 40, 245)) for n in noise)
        img = Image.new("RGB", (args.size, args.size), bg)
        d = ImageDraw.Draw(img)
        cx, cy = args.size // 2, args.size // 2
        jitter = int(rng.integers(-7, 8))
        if label == "benign":
            r = args.size // 5
            d.ellipse((cx-r+jitter, cy-r, cx+r+jitter, cy+r), fill=(110, 65, 45))
        elif label == "malignant":
            pts = [(cx-20, cy+18), (cx-10, cy-23), (cx+3, cy-12), (cx+22, cy-2), (cx+14, cy+23)]
            pts = [(x+jitter, y) for x,y in pts]
            d.polygon(pts, fill=(75, 38, 50))
        else:
            for _ in range(8):
                x = int(rng.integers(12, args.size-12)); y = int(rng.integers(12, args.size-12))
                r = int(rng.integers(3, 7))
                d.ellipse((x-r, y-r, x+r, y+r), fill=(170, 85, 70))
        name = f"sample_{i:04d}.png"
        img.save(images / name)
        rows.append({"image_path": name, "target": label, "fitzpatrick": fitz})

    pd.DataFrame(rows).to_csv(out / "metadata_standardized.csv", index=False)
    print(f"Created {len(rows)} synthetic samples in {out}")


if __name__ == "__main__":
    main()
