"""Classifier model factories (torchvision + Ultralytics YOLO-cls)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_torchvision_classifier(
    architecture: str,
    num_classes: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    from torchvision import models

    weights_resnet = models.ResNet18_Weights.DEFAULT if pretrained else None
    weights_vgg = models.VGG16_Weights.DEFAULT if pretrained else None

    if architecture == "resnet18":
        model = models.resnet18(weights=weights_resnet)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if architecture == "vgg16":
        model = models.vgg16(weights=weights_vgg)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported torchvision architecture: {architecture}")


def load_torchvision_checkpoint(
    path: str | Any,
    architecture: str,
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_torchvision_classifier(architecture, num_classes, pretrained=False)
    state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}
