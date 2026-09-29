#!/usr/bin/env bash
set -euo pipefail
python scripts/make_synthetic_demo.py --samples 180
python scripts/train.py --csv data/synthetic_demo/metadata_standardized.csv --image-root data/synthetic_demo/images --model tiny_cnn --epochs 2 --output-dir artifacts/demo
echo "Demo completed successfully. Results: artifacts/demo"
