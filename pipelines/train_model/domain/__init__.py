"""Domain config for training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrainConfig:
    data_yaml: Path
    output_dir: Path
    model_name: str = "yolo26x.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    device: str = ""
    optimizer: str = "AdamW"
    seed: int = 42
    iou_match: float = 0.5
    conf: float = 0.25
    workers: int = 2
    amp: bool = True
    cache: bool | Literal["ram", "disk"] = False
    mosaic: float = 1.0
    close_mosaic: int = 5
    skip_cls_eval: bool = False
