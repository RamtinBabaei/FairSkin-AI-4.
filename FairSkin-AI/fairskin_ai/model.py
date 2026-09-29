from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision import models


class TinyCNN(nn.Module):
    """Small model for CI/smoke tests; not the default research model."""

    def __init__(self, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _safe_weights(builder, weights_enum, pretrained: bool):
    if not pretrained:
        return builder(weights=None)
    try:
        return builder(weights=weights_enum.DEFAULT)
    except Exception as exc:  # network/cache failures should not crash the project
        warnings.warn(
            f"Could not load pretrained weights ({exc}). Falling back to random initialization.",
            RuntimeWarning,
        )
        return builder(weights=None)


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.25) -> nn.Module:
    name = name.lower()
    if name == "tiny_cnn":
        return TinyCNN(num_classes, dropout)
    if name == "efficientnet_b0":
        model = _safe_weights(models.efficientnet_b0, models.EfficientNet_B0_Weights, pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        return model
    if name == "resnet18":
        model = _safe_weights(models.resnet18, models.ResNet18_Weights, pretrained)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        return model
    if name == "resnet50":
        model = _safe_weights(models.resnet50, models.ResNet50_Weights, pretrained)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        return model
    raise ValueError(f"Unsupported model: {name}")
