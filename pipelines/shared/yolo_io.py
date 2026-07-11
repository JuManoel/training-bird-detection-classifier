"""YOLO label / data.yaml helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_line(self) -> str:
        return (
            f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    class_id: int,
) -> YoloBox:
    """Convert absolute xyxy pixels to normalized YOLO xywh."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image dimensions must be positive")
    x1 = max(0.0, min(float(x1), img_w))
    y1 = max(0.0, min(float(y1), img_h))
    x2 = max(0.0, min(float(x2), img_w))
    y2 = max(0.0, min(float(y2), img_h))
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    return YoloBox(
        class_id=class_id,
        x_center=min(max(xc, 0.0), 1.0),
        y_center=min(max(yc, 0.0), 1.0),
        width=min(max(bw / img_w, 0.0), 1.0),
        height=min(max(bh / img_h, 0.0), 1.0),
    )


def write_label_file(path: Path, boxes: list[YoloBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(b.to_line() for b in boxes) + ("\n" if boxes else ""), encoding="utf-8")


def write_data_yaml(
    path: Path,
    dataset_root: Path,
    class_names: list[str],
) -> None:
    """Write Ultralytics data.yaml with relative train/val image dirs."""
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_classes_txt(path: Path, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
