"""dfine-cpp detector adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pipelines.extract_bb.domain import BBox


class DfineBirdDetector:
    """Wraps dfine Detector and returns bird boxes in xyxy pixel space."""

    def __init__(
        self,
        engine_path: Path,
        threshold: float = 0.4,
        bird_class_names: tuple[str, ...] = ("bird",),
    ) -> None:
        try:
            from dfine import Detector
        except ImportError as exc:
            raise ImportError(
                "dfine is required for extract_bb. Install the wheel from "
                "https://github.com/PogChamper/dfine-cpp/releases and TensorRT "
                "(uv sync --extra extract), then build an engine with `dfine build`."
            ) from exc

        self._detector = Detector(str(engine_path), threshold=threshold)
        self.threshold = threshold
        self.bird_class_names = {n.lower() for n in bird_class_names}

    def detect(self, image_path: Path) -> tuple[list[BBox], tuple[int, int]]:
        with Image.open(image_path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            arr = np.asarray(rgb, dtype=np.uint8)

        boxes: list[BBox] = []
        for det in self._detector.detect(arr):
            name = getattr(det, "class_name", None) or str(getattr(det, "label", ""))
            if name.lower() not in self.bird_class_names:
                continue
            box = det.box
            if hasattr(box, "as_tuple"):
                x1, y1, x2, y2 = box.as_tuple()
            else:
                x1, y1, x2, y2 = box
            score = float(getattr(det, "score", getattr(det, "confidence", 0.0)))
            if score < self.threshold:
                continue
            boxes.append(
                BBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=score,
                    class_name=name,
                )
            )
        return boxes, (width, height)

    def close(self) -> None:
        close = getattr(self._detector, "close", None) or getattr(self._detector, "__exit__", None)
        if callable(close):
            try:
                close()
            except TypeError:
                # context-manager __exit__ signature
                self._detector.__exit__(None, None, None)

    def __enter__(self) -> DfineBirdDetector:
        return self

    def __exit__(self, *args) -> None:
        self.close()
