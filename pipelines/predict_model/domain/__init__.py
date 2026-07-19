"""Domain config for two-stage prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictConfig:
    detector_weights: Path
    source: Path
    output_dir: Path
    classifier_weights: Path | None = None
    classifier_architecture: str = "resnet18"
    classifier_weights_map: tuple[tuple[str, Path], ...] = ()
    compare: bool = False
    conf: float = 0.25
    imgsz: int = 640
    crop_size: int = 256
    pad_ratio: float = 0.1
    top_k: int = 5
    device: str = ""
