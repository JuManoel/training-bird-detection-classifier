"""Domain config for prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictConfig:
    weights: Path
    source: Path
    output_dir: Path
    conf: float = 0.25
    imgsz: int = 640
    device: str = ""
