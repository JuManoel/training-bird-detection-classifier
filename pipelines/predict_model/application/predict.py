"""Two-stage inference: YOLO bird detect → species classifier(s)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from pipelines.extract_bb.infrastructure import YoloBirdDetector
from pipelines.predict_model.domain import PredictConfig
from pipelines.shared.crop import crop_bird_to_square
from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.paths import get_project_root
from pipelines.train_classifier.infrastructure.models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_torchvision_checkpoint,
)

logger = setup_logging(name="avesia.predict")


def _collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)


def _load_class_names(ckpt: dict, weights: Path) -> list[str]:
    class_to_idx = ckpt.get("class_to_idx")
    if not class_to_idx:
        sidecar = weights.parent / "class_to_idx.json"
        if sidecar.exists():
            class_to_idx = json.loads(sidecar.read_text(encoding="utf-8"))
    if not class_to_idx:
        raise ValueError(f"No class_to_idx found for classifier weights {weights}")
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]


class _TorchvisionClassifier:
    def __init__(self, architecture: str, weights: Path, device: torch.device, imgsz: int) -> None:
        # Probe num classes from checkpoint
        raw = torch.load(weights, map_location="cpu", weights_only=False)
        class_to_idx = raw.get("class_to_idx") or {}
        if not class_to_idx:
            sidecar = weights.parent / "class_to_idx.json"
            if sidecar.exists():
                class_to_idx = json.loads(sidecar.read_text(encoding="utf-8"))
                raw["class_to_idx"] = class_to_idx
        num_classes = len(class_to_idx)
        if num_classes <= 0:
            raise ValueError(f"Cannot infer num_classes from {weights}")
        self.model, self.ckpt = load_torchvision_checkpoint(
            weights, architecture, num_classes, device
        )
        self.class_names = _load_class_names(self.ckpt if self.ckpt else raw, weights)
        self.device = device
        self.tf = transforms.Compose(
            [
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.architecture = architecture

    @torch.no_grad()
    def predict(self, crop_rgb: Image.Image, top_k: int) -> list[dict]:
        tensor = self.tf(crop_rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        k = min(top_k, probs.numel())
        values, indices = torch.topk(probs, k)
        return [
            {
                "species": self.class_names[int(i)],
                "confidence": float(v),
                "class_id": int(i),
            }
            for v, i in zip(values.tolist(), indices.tolist())
        ]


class _YoloClsClassifier:
    def __init__(self, weights: Path, device: str, imgsz: int) -> None:
        cfg = get_project_root() / "artifacts" / "ultralytics_config"
        cfg.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device
        self.architecture = "yolo26x-cls"
        sidecar = weights.parent / "class_to_idx.json"
        if sidecar.exists():
            class_to_idx = json.loads(sidecar.read_text(encoding="utf-8"))
            self.class_names = [
                name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])
            ]
        else:
            names = getattr(self.model, "names", None) or {}
            if isinstance(names, dict):
                self.class_names = [names[i] for i in sorted(names)]
            else:
                self.class_names = list(names)

    def predict(self, crop_path: Path, top_k: int) -> list[dict]:
        kwargs = {"source": str(crop_path), "imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        if not results or results[0].probs is None:
            return []
        probs = results[0].probs
        k = min(top_k, len(self.class_names) or probs.data.numel())
        top_idx = probs.top5[:k] if hasattr(probs, "top5") else probs.data.topk(k).indices.tolist()
        top_conf = (
            probs.top5conf[:k].tolist()
            if hasattr(probs, "top5conf")
            else probs.data.topk(k).values.tolist()
        )
        out = []
        for idx, conf in zip(top_idx, top_conf):
            i = int(idx)
            name = self.class_names[i] if i < len(self.class_names) else str(i)
            out.append({"species": name, "confidence": float(conf), "class_id": i})
        return out


def _load_classifier(architecture: str, weights: Path, device: torch.device, imgsz: int, device_str: str):
    if architecture == "yolo26x-cls":
        return _YoloClsClassifier(weights, device_str, imgsz)
    return _TorchvisionClassifier(architecture, weights, device, imgsz)


def run_predict(config: PredictConfig) -> Path:
    device = torch.device(config.device) if config.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    device_str = config.device or (str(device) if device.type == "cuda" else "cpu")

    classifiers: list[tuple[str, object]] = []
    if config.compare and config.classifier_weights_map:
        for arch, weights in config.classifier_weights_map:
            classifiers.append(
                (arch, _load_classifier(arch, weights, device, config.crop_size, device_str))
            )
    elif config.classifier_weights is not None:
        classifiers.append(
            (
                config.classifier_architecture,
                _load_classifier(
                    config.classifier_architecture,
                    config.classifier_weights,
                    device,
                    config.crop_size,
                    device_str,
                ),
            )
        )
    else:
        raise ValueError("Provide classifier_weights or compare with classifier_weights_map")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = config.output_dir / "crops"
    annotated_dir = config.output_dir / "annotated"
    crops_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(config.source)
    detections: list[dict] = []

    with YoloBirdDetector(
        model=str(config.detector_weights),
        threshold=config.conf,
        imgsz=config.imgsz,
        device=config.device or None,
    ) as detector:
        for image_path in images:
            boxes, (w, h) = detector.detect(image_path)
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
                draw = ImageDraw.Draw(rgb)
                try:
                    font = ImageFont.load_default()
                except Exception:  # noqa: BLE001
                    font = None

                for i, box in enumerate(boxes):
                    crop_name = f"{image_path.stem}_{i}.jpg"
                    crop_path = crops_dir / crop_name
                    crop_bird_to_square(
                        image_path,
                        crop_path,
                        box.x1,
                        box.y1,
                        box.x2,
                        box.y2,
                        size=config.crop_size,
                        pad_ratio=config.pad_ratio,
                    )
                    # Read crop for torchvision classifiers
                    with Image.open(crop_path) as crop_im:
                        crop_rgb = crop_im.convert("RGB")

                    per_clf = {}
                    primary = None
                    for arch, clf in classifiers:
                        if arch == "yolo26x-cls":
                            top = clf.predict(crop_path, config.top_k)
                        else:
                            top = clf.predict(crop_rgb, config.top_k)
                        per_clf[arch] = top
                        if primary is None:
                            primary = (arch, top)

                    species = primary[1][0]["species"] if primary and primary[1] else "unknown"
                    cls_conf = primary[1][0]["confidence"] if primary and primary[1] else 0.0
                    arch_name = primary[0] if primary else config.classifier_architecture

                    record = {
                        "image": str(image_path),
                        "xyxy": [box.x1, box.y1, box.x2, box.y2],
                        "det_conf": box.score,
                        "species": species,
                        "cls_conf": cls_conf,
                        "classifier": arch_name,
                        "top_k": primary[1] if primary else [],
                        "crop_path": str(crop_path),
                    }
                    if config.compare:
                        record["classifiers"] = per_clf
                    detections.append(record)

                    label = f"{species} {cls_conf:.2f}"
                    draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(0, 200, 80), width=3)
                    draw.text(
                        (box.x1, max(0, box.y1 - 12)),
                        label,
                        fill=(0, 200, 80),
                        font=font,
                    )

                rgb.save(annotated_dir / image_path.name, quality=90)

    out_json = config.output_dir / "predictions.json"
    out_json.write_text(json.dumps(detections, indent=2), encoding="utf-8")
    logger.info("Wrote %d detections → %s", len(detections), out_json)
    return out_json
