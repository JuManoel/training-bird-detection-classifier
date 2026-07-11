"""CLI for YOLO26 multi-class training."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.shared.paths import ProjectPaths
from pipelines.train_model.application import run_train
from pipelines.train_model.domain import TrainConfig


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(description="Train YOLO26 multi-class bird detector")
    p.add_argument("--data", type=str, default=str(paths.data_yaml))
    p.add_argument("--out", type=str, default=str(paths.train_run))
    p.add_argument("--model", type=str, default="yolo26n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--optimizer", type=str, default="AdamW")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-match", type=float, default=0.5)
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
    )
    run_train(config)


if __name__ == "__main__":
    main()
