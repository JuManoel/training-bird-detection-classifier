"""CLI for YOLO bird detection + 256×256 classification crop dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.extract_bb.application import run_extract
from pipelines.extract_bb.domain import ExtractConfig
from pipelines.shared.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(
        description=(
            "Detect birds with YOLO26, build a 1-class detect dataset, "
            "and write 256×256 classification crops"
        )
    )
    p.add_argument("--manifest", type=str, default=str(paths.manifest))
    p.add_argument("--species", type=str, default=str(paths.species_file))
    p.add_argument("--catalog", type=str, default=str(paths.species_catalog))
    p.add_argument("--detect-dataset", type=str, default=str(paths.dataset_detect))
    p.add_argument("--classify-dataset", type=str, default=str(paths.dataset_classify))
    p.add_argument("--coverage-json", type=str, default=str(paths.coverage_json))
    p.add_argument(
        "--model",
        type=str,
        default="yolo26x.pt",
        help="Ultralytics COCO detector weights (default: yolo26x.pt)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device, e.g. 0, cuda:0 or cpu",
    )
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=640, help="Max side for detect full-frames")
    p.add_argument("--crop-size", type=int, default=256, help="Classifier crop size")
    p.add_argument("--pad-ratio", type=float, default=0.1)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-images",
        type=int,
        default=125,
        help="Drop species with fewer than this many unique crops",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ExtractConfig(
        manifest_path=Path(args.manifest),
        species_path=Path(args.species),
        catalog_path=Path(args.catalog),
        detect_root=Path(args.detect_dataset),
        classify_root=Path(args.classify_dataset),
        coverage_json=Path(args.coverage_json),
        model=args.model,
        device=args.device,
        threshold=args.threshold,
        imgsz=args.imgsz,
        crop_size=args.crop_size,
        pad_ratio=args.pad_ratio,
        jpeg_quality=args.jpeg_quality,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_images=args.min_images,
    )
    run_extract(config)


if __name__ == "__main__":
    main()
