import torch

from fairskin_ai.model import build_model


def test_tiny_cnn_forward():
    model = build_model("tiny_cnn", 3, pretrained=False)
    x = torch.randn(2,3,64,64)
    y = model(x)
    assert y.shape == (2,3)
