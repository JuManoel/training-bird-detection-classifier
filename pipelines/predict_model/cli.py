"""CLI for two-stage bird detection + species classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.predict_model.application import run_predict
from pipelines.predict_model.domain import PredictConfig
from pipelines.shared.paths import ProjectPaths
from pipelines.train_classifier.domain import SUPPORTED_ARCHITECTURES


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(
        description="Detect birds with YOLO then classify species with a specialized model"
    )
    p.add_argument(
        "--detector",
        type=str,
        default="yolo26x.pt",
        help="YOLO bird detector weights (COCO or fine-tuned single-class)",
    )
    p.add_argument(
        "--classifier",
        type=str,
        default=None,
        help="Classifier checkpoint (default: artifacts/runs/train_cls/<arch>/best.pt)",
    )
    p.add_argument(
        "--architecture",
        type=str,
        choices=list(SUPPORTED_ARCHITECTURES),
        default="resnet18",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run all three classifiers on the same detections (no ensemble)",
    )
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument("--out", type=str, default=str(paths.predict_out))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--pad-ratio", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--device", type=str, default="")
    return p


def main(argv: list[str] | None = None) -> None:
    paths = ProjectPaths.from_root()
    args = build_parser().parse_args(argv)

    weights_map: tuple[tuple[str, Path], ...] = ()
    classifier_weights: Path | None = None
    if args.compare:
        pairs = []
        for arch in SUPPORTED_ARCHITECTURES:
            path = paths.classifier_best(arch.replace("-", "_"))
            if path.exists():
                pairs.append((arch, path))
        if not pairs:
            raise SystemExit(
                "No classifier checkpoints found under artifacts/runs/train_cls/"
            )
        weights_map = tuple(pairs)
    else:
        if args.classifier:
            classifier_weights = Path(args.classifier)
        else:
            classifier_weights = paths.classifier_best(args.architecture.replace("-", "_"))
        if not classifier_weights.exists():
            raise SystemExit(f"Classifier weights not found: {classifier_weights}")

    config = PredictConfig(
        detector_weights=Path(args.detector),
        source=Path(args.source),
        output_dir=Path(args.out),
        classifier_weights=classifier_weights,
        classifier_architecture=args.architecture,
        classifier_weights_map=weights_map,
        compare=args.compare,
        conf=args.conf,
        imgsz=args.imgsz,
        crop_size=args.crop_size,
        pad_ratio=args.pad_ratio,
        top_k=args.top_k,
        device=args.device,
    )
    run_predict(config)


if __name__ == "__main__":
    main()
