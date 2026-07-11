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
    dataset_root: Path
    engine_path: Path
    threshold: float = 0.4
    keep_all: bool = False
    train_ratio: float = 0.8
    seed: int = 42
    bird_class_names: tuple[str, ...] = ("bird",)
