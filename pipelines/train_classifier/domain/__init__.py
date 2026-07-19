"""Domain config for classifier training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_ARCHITECTURES = ("resnet18", "vgg16", "yolo26x-cls")


@dataclass(frozen=True)
class ClassifierTrainConfig:
    data_dir: Path
    output_dir: Path
    architecture: str = "resnet18"
    epochs: int = 50
    imgsz: int = 256
    batch: int = 32
    device: str = ""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 4
    pretrained: bool = True
    yolo_model: str = "yolo26x-cls.pt"
