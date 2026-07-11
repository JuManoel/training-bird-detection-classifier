"""Run YOLO26 inference and persist annotated images + JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pipelines.predict_model.domain import PredictConfig
from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.paths import get_project_root

logger = setup_logging(name="avesia.predict")


def run_predict(config: PredictConfig) -> Path:
    cfg = get_project_root() / "artifacts" / "ultralytics_config"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))

    from ultralytics import YOLO

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(config.weights))
    kwargs = {
        "source": str(config.source),
        "conf": config.conf,
        "imgsz": config.imgsz,
        "project": str(config.output_dir.parent),
        "name": config.output_dir.name,
        "exist_ok": True,
        "save": True,
    }
    if config.device:
        kwargs["device"] = config.device

    results = model.predict(**kwargs)
    detections: list[dict] = []
    for r in results:
        path = Path(getattr(r, "path", ""))
        names = r.names or {}
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            detections.append(
                {
                    "image": str(path),
                    "class_id": cls_id,
                    "species": names.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf.item()),
                    "xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
            )

    out_json = config.output_dir / "predictions.json"
    out_json.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    logger.info("Wrote %d detections → %s", len(detections), out_json)
    return out_json
