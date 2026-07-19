"""Domain types for extract_bb."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_name: str


@dataclass(frozen=True)
class ExtractConfig:
    manifest_path: Path
    species_path: Path
    catalog_path: Path
    detect_root: Path
    classify_root: Path
    coverage_json: Path
    model: str = "yolo26x.pt"
    device: str | None = None
    threshold: float = 0.4
    imgsz: int = 640
    crop_size: int = 256
    pad_ratio: float = 0.1
    jpeg_quality: int = 90
    train_ratio: float = 0.8
    seed: int = 42
    min_images: int = 125
    bird_class_names: tuple[str, ...] = ("bird",)
