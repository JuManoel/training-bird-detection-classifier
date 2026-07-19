"""CLI for species classifier training (ResNet18 / VGG16 / YOLO26x-cls)."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.shared.paths import ProjectPaths
from pipelines.train_classifier.application import (
    run_compare_classifiers,
    run_train_classifier,
)
from pipelines.train_classifier.domain import (
    SUPPORTED_ARCHITECTURES,
    ClassifierTrainConfig,
)


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(
        description="Train bird-species classifiers on 256×256 YOLO crops"
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(paths.dataset_classify),
        help="ImageFolder root with train/ and val/ species folders",
    )
    p.add_argument(
        "--architecture",
        type=str,
        choices=[*SUPPORTED_ARCHITECTURES, "all"],
        default="resnet18",
        help="Model to train, or 'all' to train and compare the three",
    )
    p.add_argument("--out", type=str, default=None, help="Run directory (single model)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--yolo-model", type=str, default="yolo26x-cls.pt")
    return p


def main(argv: list[str] | None = None) -> None:
    paths = ProjectPaths.from_root()
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data)

    if args.architecture == "all":
        configs = [
            ClassifierTrainConfig(
                data_dir=data_dir,
                output_dir=paths.classifier_run(arch.replace("-", "_")),
                architecture=arch,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed,
                num_workers=args.workers,
                pretrained=not args.no_pretrained,
                yolo_model=args.yolo_model,
            )
            for arch in SUPPORTED_ARCHITECTURES
        ]
        run_compare_classifiers(configs, paths.train_cls_run / "comparison.json")
        return

    out = Path(args.out) if args.out else paths.classifier_run(args.architecture.replace("-", "_"))
    config = ClassifierTrainConfig(
        data_dir=data_dir,
        output_dir=out,
        architecture=args.architecture,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.workers,
        pretrained=not args.no_pretrained,
        yolo_model=args.yolo_model,
    )
    run_train_classifier(config)


if __name__ == "__main__":
    main()
