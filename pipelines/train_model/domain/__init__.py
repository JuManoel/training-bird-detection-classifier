"""Domain config for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    data_yaml: Path
    output_dir: Path
    model_name: str = "yolo26n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = ""
    optimizer: str = "AdamW"
    seed: int = 42
    iou_match: float = 0.5
    conf: float = 0.25
