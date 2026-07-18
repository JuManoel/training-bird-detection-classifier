"""Ultralytics YOLO detector adapter for COCO bird boxes."""

from __future__ import annotations

import os
from pathlib import Path

from pipelines.extract_bb.domain import BBox
from pipelines.shared.paths import get_project_root


class YoloBirdDetector:
    """Detect COCO birds and return boxes in xyxy pixel coordinates."""

    def __init__(
        self,
        model: str = "yolo26x.pt",
        threshold: float = 0.4,
        imgsz: int = 640,
        device: str | None = None,
        bird_class_names: tuple[str, ...] = ("bird",),
    ) -> None:
        cfg = get_project_root() / "artifacts" / "ultralytics_config"
        cfg.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))

        from ultralytics import YOLO

        self._model = YOLO(model)
        self.threshold = threshold
        self.imgsz = imgsz
        self.device = device

        wanted = {name.lower() for name in bird_class_names}
        self._bird_class_ids = [
            int(class_id)
            for class_id, name in self._model.names.items()
            if str(name).lower() in wanted
        ]
        if not self._bird_class_ids:
            raise ValueError(
                f"Model {model!r} has no bird class. Available classes: "
                f"{tuple(self._model.names.values())}"
            )

    def detect(self, image_path: Path) -> tuple[list[BBox], tuple[int, int]]:
        kwargs: dict[str, object] = {
            "source": str(image_path),
            "conf": self.threshold,
            "imgsz": self.imgsz,
            "classes": self._bird_class_ids,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self._model.predict(**kwargs)
        if not results:
            return [], (0, 0)

        result = results[0]
        height, width = result.orig_shape
        boxes: list[BBox] = []
        if result.boxes is None:
            return boxes, (width, height)

        names = result.names or self._model.names
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            boxes.append(
                BBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    score=float(box.conf.item()),
                    class_name=str(names.get(class_id, class_id)),
                )
            )
        return boxes, (width, height)

    def __enter__(self) -> YoloBirdDetector:
        return self

    def __exit__(self, *args: object) -> None:
        return None
