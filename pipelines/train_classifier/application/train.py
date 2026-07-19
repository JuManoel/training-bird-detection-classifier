"""Train ResNet18 / VGG16 / YOLO26x-cls on 256×256 bird crops."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.metrics import compute_classification_metrics, save_confusion_heatmap
from pipelines.shared.paths import get_project_root
from pipelines.train_classifier.domain import (
    SUPPORTED_ARCHITECTURES,
    ClassifierTrainConfig,
)
from pipelines.train_classifier.infrastructure.models import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_torchvision_classifier,
)

logger = setup_logging(name="avesia.train_cls")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device: str) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders(config: ClassifierTrainConfig) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((config.imgsz, config.imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((config.imgsz, config.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected ImageFolder splits at {train_dir} and {val_dir}"
        )
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    if train_ds.classes != val_ds.classes:
        # Align val to train class_to_idx by rebuilding samples if folders match
        if set(train_ds.classes) != set(val_ds.classes):
            raise ValueError("train/val class folders differ")
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds.class_to_idx


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.numel()
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    summary = {
        "accuracy": metrics.accuracy,
        "precision_macro": metrics.precision,
        "recall_macro": metrics.recall,
        "f1_macro": metrics.f1,
        "n": total,
    }
    return summary, y_true, y_pred


def _train_torchvision(config: ClassifierTrainConfig) -> dict:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    train_loader, val_loader, class_to_idx = _build_loaders(config)
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    num_classes = len(class_names)
    model = build_torchvision_classifier(
        config.architecture, num_classes, pretrained=config.pretrained
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_f1 = -1.0
    best_path = config.output_dir / "best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for images, labels in tqdm(train_loader, desc=f"{config.architecture} ep{epoch}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        train_loss = running_loss / max(n_batches, 1)
        val_metrics, y_true, y_pred = _evaluate(model, val_loader, device, class_names)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(row)
        logger.info(
            "%s epoch %d loss=%.4f acc=%.4f f1=%.4f",
            config.architecture,
            epoch,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "history": history,
                    "class_to_idx": class_to_idx,
                    "architecture": config.architecture,
                    "imgsz": config.imgsz,
                    "metrics": val_metrics,
                    "config": {
                        "epochs": config.epochs,
                        "batch": config.batch,
                        "lr": config.lr,
                        "seed": config.seed,
                        "data_dir": str(config.data_dir),
                    },
                },
                best_path,
            )
            metrics_obj = compute_classification_metrics(y_true, y_pred, class_names)
            save_confusion_heatmap(
                metrics_obj,
                config.output_dir / "confusion_matrix.png",
                title=f"{config.architecture} val confusion",
            )

    (config.output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (config.output_dir / "metrics.json").write_text(
        json.dumps({"best_f1_macro": best_f1, "history_tail": history[-1] if history else {}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2) + "\n", encoding="utf-8"
    )
    return {"architecture": config.architecture, "best_f1_macro": best_f1, "best_path": str(best_path)}


def _train_yolo_cls(config: ClassifierTrainConfig) -> dict:
    """Train Ultralytics YOLO classification model on ImageFolder layout."""
    cfg = get_project_root() / "artifacts" / "ultralytics_config"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cfg))

    from ultralytics import YOLO

    _set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(config.yolo_model)
    kwargs: dict = {
        "data": str(config.data_dir),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "project": str(config.output_dir.parent),
        "name": config.output_dir.name,
        "exist_ok": True,
        "seed": config.seed,
        "pretrained": config.pretrained,
        "optimizer": "Adam",
        "lr0": config.lr,
        "weight_decay": config.weight_decay,
        "workers": config.num_workers,
    }
    if config.device:
        kwargs["device"] = config.device

    results = model.train(**kwargs)

    # Copy/link best weights to output_dir/best.pt for a stable path
    run_best = config.output_dir / "weights" / "best.pt"
    stable = config.output_dir / "best.pt"
    if run_best.exists() and not stable.exists():
        try:
            stable.symlink_to(run_best.resolve())
        except OSError:
            import shutil

            shutil.copy2(run_best, stable)

    # Build class_to_idx from folder names
    train_dir = config.data_dir / "train"
    class_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    (config.output_dir / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2) + "\n", encoding="utf-8"
    )

    # Evaluate with YOLO val if possible
    metrics_summary = {"architecture": "yolo26x-cls"}
    try:
        val_res = model.val(data=str(config.data_dir), imgsz=config.imgsz)
        # Ultralytics classification metrics vary by version; best-effort extract
        top1 = float(getattr(val_res, "top1", 0.0) or 0.0)
        metrics_summary.update(
            {
                "accuracy": top1,
                "precision_macro": top1,
                "recall_macro": top1,
                "f1_macro": top1,
                "note": "YOLO-cls reports top1; macro P/R/F1 approximated by top1 in summary",
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("YOLO-cls val metrics unavailable: %s", exc)
        metrics_summary["error"] = str(exc)

    # Optional detailed sklearn metrics via torchvision-style reload is heavy;
    # compute with a simple pass over val folder using YOLO predict batches.
    try:
        from PIL import Image

        y_true: list[int] = []
        y_pred: list[int] = []
        val_root = config.data_dir / "val"
        for class_name in class_names:
            class_dir = val_root / class_name
            if not class_dir.is_dir():
                continue
            true_id = class_to_idx[class_name]
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                pred = model.predict(str(img_path), imgsz=config.imgsz, verbose=False)
                if not pred:
                    continue
                probs = pred[0].probs
                if probs is None:
                    continue
                y_true.append(true_id)
                y_pred.append(int(probs.top1))
        if y_true:
            m = compute_classification_metrics(y_true, y_pred, class_names)
            metrics_summary.update(
                {
                    "accuracy": m.accuracy,
                    "precision_macro": m.precision,
                    "recall_macro": m.recall,
                    "f1_macro": m.f1,
                }
            )
            save_confusion_heatmap(
                m,
                config.output_dir / "confusion_matrix.png",
                title="yolo26x-cls val confusion",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Detailed YOLO-cls eval failed: %s", exc)

    (config.output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2) + "\n", encoding="utf-8"
    )
    (config.output_dir / "architecture.json").write_text(
        json.dumps({"architecture": "yolo26x-cls", "imgsz": config.imgsz}, indent=2) + "\n",
        encoding="utf-8",
    )
    _ = results
    return {
        "architecture": "yolo26x-cls",
        "best_f1_macro": metrics_summary.get("f1_macro", 0.0),
        "best_path": str(stable if stable.exists() else run_best),
        **{k: v for k, v in metrics_summary.items() if k != "architecture"},
    }


def run_train_classifier(config: ClassifierTrainConfig) -> dict:
    if config.architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"architecture must be one of {SUPPORTED_ARCHITECTURES}, got {config.architecture}"
        )
    if config.architecture == "yolo26x-cls":
        return _train_yolo_cls(config)
    return _train_torchvision(config)


def run_compare_classifiers(
    configs: list[ClassifierTrainConfig],
    comparison_out: Path,
) -> dict:
    """Train each architecture and write a comparison table (best by F1 macro)."""
    rows = []
    for cfg in configs:
        logger.info("Training classifier %s → %s", cfg.architecture, cfg.output_dir)
        result = run_train_classifier(cfg)
        metrics_path = cfg.output_dir / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "architecture": cfg.architecture,
                "accuracy": metrics.get("accuracy", result.get("accuracy")),
                "precision_macro": metrics.get(
                    "precision_macro", result.get("precision_macro")
                ),
                "recall_macro": metrics.get("recall_macro", result.get("recall_macro")),
                "f1_macro": metrics.get("f1_macro", result.get("best_f1_macro")),
                "best_path": result.get("best_path"),
            }
        )
    rows_sorted = sorted(rows, key=lambda r: float(r.get("f1_macro") or 0.0), reverse=True)
    payload = {
        "models": rows_sorted,
        "best_architecture": rows_sorted[0]["architecture"] if rows_sorted else None,
        "selection_metric": "f1_macro",
    }
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    comparison_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Comparison written to %s (best=%s)", comparison_out, payload["best_architecture"])
    return payload
