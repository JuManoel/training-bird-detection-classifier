"""Ultralytics YOLO training adapter with best-by-val-loss checkpointing."""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

import yaml

from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.metrics import (
    compute_classification_metrics,
    match_detections_to_gt,
    save_confusion_heatmap,
)
from pipelines.shared.paths import get_project_root
from pipelines.shared.split import early_stop_patience
from pipelines.train_model.domain import TrainConfig

logger = setup_logging(name="avesia.train")


def _release_cuda_memory() -> None:
    """Drop Python refs and free cached CUDA allocations."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _configure_ultralytics_env() -> None:
    cfg = get_project_root() / "artifacts" / "ultralytics_config"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))


def _extract_val_loss(trainer) -> float | None:
    """Best-effort read of validation detection loss from an Ultralytics trainer."""
    metrics = getattr(trainer, "metrics", {}) or {}
    if isinstance(metrics, dict):
        for key in ("val/loss", "val/box_loss", "train/box_loss"):
            if key in metrics:
                # Prefer summing box+cls+dfl when available
                break
        parts = [
            float(metrics[k])
            for k in ("val/box_loss", "val/cls_loss", "val/dfl_loss")
            if k in metrics
        ]
        if parts:
            return sum(parts)
        if "val/loss" in metrics:
            return float(metrics["val/loss"])

    validator = getattr(trainer, "validator", None)
    if validator is not None:
        vloss = getattr(validator, "loss", None)
        if vloss is not None:
            try:
                import torch

                if isinstance(vloss, torch.Tensor):
                    return float(vloss.detach().sum().item())
                return float(vloss)
            except Exception:  # noqa: BLE001
                pass
        vmetrics = getattr(validator, "metrics", None)
        if vmetrics is not None:
            rd = getattr(vmetrics, "results_dict", None) or {}
            if isinstance(rd, dict):
                parts = [
                    float(v)
                    for k, v in rd.items()
                    if isinstance(k, str) and "loss" in k.lower() and isinstance(v, (int, float))
                ]
                if parts:
                    return sum(parts)

    # Fallback: epoch training loss (less ideal than val)
    tloss = getattr(trainer, "tloss", None)
    if tloss is not None:
        try:
            import torch

            if isinstance(tloss, torch.Tensor):
                return float(tloss.detach().mean().item())
            return float(tloss)
        except Exception:  # noqa: BLE001
            return None
    return None


class BestByValLossCallback:
    """Track lowest validation loss and copy weights to output_dir/best.pt."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.history: list[dict] = []

    def on_fit_epoch_end(self, trainer) -> None:
        metrics = getattr(trainer, "metrics", {}) or {}
        loss = _extract_val_loss(trainer)
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        entry = {
            "epoch": epoch,
            "val_loss": loss,
            "metrics": {
                k: float(v)
                for k, v in (metrics.items() if isinstance(metrics, dict) else [])
                if isinstance(v, (int, float))
            },
        }
        self.history.append(entry)

        if loss is None:
            logger.warning(
                "Epoch %d: could not read val loss; skipping best checkpoint update", epoch
            )
            return

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.output_dir.mkdir(parents=True, exist_ok=True)
            dest = self.output_dir / "best.pt"
            weights_dir = Path(trainer.save_dir) / "weights"
            # Prefer last.pt (current epoch) over mAP-based best.pt
            for candidate in (weights_dir / "last.pt", weights_dir / "best.pt"):
                if candidate.exists():
                    shutil.copy2(candidate, dest)
                    break
            else:
                try:
                    trainer.save_model()
                except Exception:  # noqa: BLE001
                    pass
                last = weights_dir / "last.pt"
                if last.exists():
                    shutil.copy2(last, dest)
            logger.info("New best val_loss=%.6f @ epoch %d → %s", loss, epoch, dest)


def _load_class_names(data_yaml: Path) -> list[str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload.get("names", {})
    if isinstance(names, dict):
        return [names[i] for i in sorted(names, key=lambda x: int(x))]
    return list(names)


def _yolo_label_to_xyxy(line: str, w: int, h: int) -> tuple[int, float, float, float, float]:
    parts = line.strip().split()
    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return cls, x1, y1, x2, y2


def evaluate_classification(
    model,
    data_yaml: Path,
    output_dir: Path,
    conf: float,
    iou_match: float,
) -> dict[str, float]:
    """Run predictions on val images and build confusion heatmap metrics."""
    from PIL import Image

    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    val_img_dir = root / payload["val"]
    val_lbl_dir = root / "labels" / "val"
    class_names = _load_class_names(data_yaml)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    images = sorted(
        [
            p
            for p in val_img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
    )
    for img_path in images:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            w, h = im.size
        gt: list[tuple[int, float, float, float, float]] = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    gt.append(_yolo_label_to_xyxy(line, w, h))

        results = model.predict(source=str(img_path), conf=conf, verbose=False, stream=False)
        preds: list[tuple[int, float, float, float, float, float]] = []
        if results:
            r0 = results[0]
            if r0.boxes is not None and len(r0.boxes):
                for box in r0.boxes:
                    cls_id = int(box.cls.item())
                    score = float(box.conf.item())
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    preds.append((cls_id, score, x1, y1, x2, y2))
        del results

        yt, yp = match_detections_to_gt(gt, preds, iou_threshold=iou_match)
        y_true_all.extend(yt)
        y_pred_all.extend(yp)

    metrics = compute_classification_metrics(y_true_all, y_pred_all, class_names)
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    heat_path = save_confusion_heatmap(
        metrics,
        plots / "confusion_heatmap.png",
        title="Species confusion (IoU-matched)",
    )
    summary = {
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision,
        "recall_macro": metrics.recall,
        "f1_macro": metrics.f1,
        "matched_pairs": float(len(y_true_all)),
    }
    (plots / "classification_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Classification metrics: %s | heatmap=%s", summary, heat_path)
    return summary


def run_train(config: TrainConfig) -> Path:
    _configure_ultralytics_env()
    from ultralytics import YOLO

    config.output_dir.mkdir(parents=True, exist_ok=True)
    patience = early_stop_patience(config.epochs, fraction=0.05)
    logger.info(
        "Training %s for %d epochs | optimizer=%s | patience=%d | "
        "batch=%s workers=%d amp=%s cache=%s mosaic=%s close_mosaic=%d",
        config.model_name,
        config.epochs,
        config.optimizer,
        patience,
        config.batch,
        config.workers,
        config.amp,
        config.cache,
        config.mosaic,
        config.close_mosaic,
    )

    model = YOLO(config.model_name)
    tracker = BestByValLossCallback(config.output_dir)

    def on_fit_epoch_end(trainer):
        tracker.on_fit_epoch_end(trainer)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    train_kwargs = {
        "data": str(config.data_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "optimizer": config.optimizer,
        "patience": patience,
        "seed": config.seed,
        "project": str(config.output_dir.parent),
        "name": config.output_dir.name,
        "exist_ok": True,
        "save": True,
        "plots": True,
        "workers": config.workers,
        "amp": config.amp,
        "cache": config.cache,
        "mosaic": config.mosaic,
        "close_mosaic": config.close_mosaic,
    }
    if config.device:
        train_kwargs["device"] = config.device

    model.train(**train_kwargs)

    # Ensure best.pt exists (fallback to ultralytics best)
    best = config.output_dir / "best.pt"
    ultra_best = config.output_dir / "weights" / "best.pt"
    if not best.exists() and ultra_best.exists():
        shutil.copy2(ultra_best, best)
    if not best.exists():
        last = config.output_dir / "weights" / "last.pt"
        if last.exists():
            shutil.copy2(last, best)
            logger.warning("best.pt missing; copied last.pt as best.pt")

    (config.output_dir / "loss_history.json").write_text(
        json.dumps(tracker.history, indent=2), encoding="utf-8"
    )
    logger.info(
        "Best by val loss: epoch=%d loss=%s path=%s",
        tracker.best_epoch,
        tracker.best_loss if tracker.best_loss < float("inf") else None,
        best,
    )

    # Free training model before loading best.pt for classification eval
    del model
    _release_cuda_memory()

    if config.skip_cls_eval:
        logger.info("Skipping classification eval (--skip-cls-eval)")
        return best

    if not best.exists():
        logger.warning("best.pt missing; skipping classification eval")
        return best

    eval_model = YOLO(str(best))
    try:
        evaluate_classification(
            eval_model,
            config.data_yaml,
            config.output_dir,
            conf=config.conf,
            iou_match=config.iou_match,
        )
    finally:
        del eval_model
        _release_cuda_memory()

    return best
