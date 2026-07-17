"""CLI for YOLO26 multi-class training."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.shared.paths import ProjectPaths
from pipelines.train_model.application import run_train
from pipelines.train_model.domain import TrainConfig


def _parse_cache(value: str) -> bool | str:
    lowered = value.lower()
    if lowered in {"false", "0", "no", "none"}:
        return False
    if lowered in {"true", "1", "yes", "ram"}:
        return "ram"
    if lowered == "disk":
        return "disk"
    raise argparse.ArgumentTypeError(
        f"invalid cache value {value!r}; use false, ram, or disk"
    )


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(description="Train YOLO26 multi-class bird detector")
    p.add_argument("--data", type=str, default=str(paths.data_yaml))
    p.add_argument("--out", type=str, default=str(paths.train_run))
    p.add_argument("--model", type=str, default="yolo26x.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--batch",
        type=int,
        default=512,
        help="Batch size (use -1 for Ultralytics auto-batch)",
    )
    p.add_argument("--device", type=str, default="")
    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader workers (lower uses less host RAM)",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatic Mixed Precision (fp16); default on",
    )
    p.add_argument(
        "--cache",
        type=_parse_cache,
        default=False,
        help="Dataset cache: false (default), ram, or disk — avoid ram on large sets",
    )
    p.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic augmentation probability (0 disables; lowers VRAM/RAM)",
    )
    p.add_argument(
        "--close-mosaic",
        type=int,
        default=5,
        help="Disable mosaic for the last N epochs",
    )
    p.add_argument(
        "--skip-cls-eval",
        action="store_true",
        help="Skip post-train classification eval on val set",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = TrainConfig(
        data_yaml=Path(args.data),
        output_dir=Path(args.out),
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer=args.optimizer,
        seed=args.seed,
        conf=args.conf,
        iou_match=args.iou_match,
        workers=args.workers,
        amp=args.amp,
        cache=args.cache,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        skip_cls_eval=args.skip_cls_eval,
    )
    run_train(config)


if __name__ == "__main__":
    main()
