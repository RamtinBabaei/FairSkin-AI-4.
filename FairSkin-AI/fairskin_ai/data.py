from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

TARGET_COL = "target"
FITZ_COL = "fitzpatrick"
IMAGE_COL = "image_path"


def skin_group(value: int, grouping: str = "three_groups") -> str:
    v = int(value)
    if v not in {1, 2, 3, 4, 5, 6}:
        return "Unknown"
    if grouping == "binary":
        return "I-III" if v <= 3 else "IV-VI"
    if grouping == "six_groups":
        return f"Type {v}"
    if grouping != "three_groups":
        raise ValueError(f"Unsupported skin grouping: {grouping}")
    if v <= 2:
        return "I-II"
    if v <= 4:
        return "III-IV"
    return "V-VI"


def add_skin_group(df: pd.DataFrame, grouping: str = "three_groups") -> pd.DataFrame:
    out = df.copy()
    out["skin_group"] = out[FITZ_COL].map(lambda x: skin_group(x, grouping))
    return out


def validate_metadata(df: pd.DataFrame) -> None:
    required = {IMAGE_COL, TARGET_COL, FITZ_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Metadata is empty.")
    invalid = ~df[FITZ_COL].astype(int).isin([1, 2, 3, 4, 5, 6])
    if invalid.any():
        raise ValueError("fitzpatrick values must be integers from 1 to 6.")


def _safe_stratify_key(df: pd.DataFrame) -> pd.Series:
    key = df[TARGET_COL].astype(str) + "__" + df["skin_group"].astype(str)
    counts = key.value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        return key
    return df[TARGET_COL].astype(str)


def split_metadata(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
    grouping: str = "three_groups",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be >0 and sum to <1.")
    df = add_skin_group(df.reset_index(drop=True), grouping)
    stratify = _safe_stratify_key(df)
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    relative_val = val_size / (1.0 - test_size)
    stratify_tv = _safe_stratify_key(train_val)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=stratify_tv,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def build_transforms(image_size: int, train: bool) -> Callable:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@dataclass(frozen=True)
class LabelMap:
    class_to_idx: dict[str, int]

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {v: k for k, v in self.class_to_idx.items()}


class DermatologyDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        label_map: LabelMap,
        transform: Callable | None = None,
    ) -> None:
        validate_metadata(frame)
        self.frame = frame.reset_index(drop=True).copy()
        self.root = Path(image_root)
        self.label_map = label_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        path = Path(str(row[IMAGE_COL]))
        if not path.is_absolute():
            path = self.root / path
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target_name = str(row[TARGET_COL])
        target = self.label_map.class_to_idx[target_name]
        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.long),
            "fitzpatrick": int(row[FITZ_COL]),
            "skin_group": str(row.get("skin_group", skin_group(int(row[FITZ_COL])))),
            "image_path": str(path),
        }


def make_label_map(*frames: pd.DataFrame) -> LabelMap:
    labels: set[str] = set()
    for frame in frames:
        labels.update(frame[TARGET_COL].astype(str).unique().tolist())
    ordered = sorted(labels)
    return LabelMap({name: i for i, name in enumerate(ordered)})


def class_weights(frame: pd.DataFrame, label_map: LabelMap) -> torch.Tensor:
    counts = frame[TARGET_COL].astype(str).value_counts()
    total = counts.sum()
    n = len(label_map.class_to_idx)
    weights = []
    for name, _ in sorted(label_map.class_to_idx.items(), key=lambda kv: kv[1]):
        c = max(int(counts.get(name, 0)), 1)
        weights.append(total / (n * c))
    return torch.tensor(weights, dtype=torch.float32)


def intersectional_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    if "skin_group" not in frame.columns:
        raise ValueError("skin_group column is required for intersectional sampling.")
    keys = frame[TARGET_COL].astype(str) + "__" + frame["skin_group"].astype(str)
    counts = keys.value_counts()
    weights = keys.map(lambda k: 1.0 / float(counts[k])).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
