"""CLI for dfine-cpp bird box extraction → YOLO dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.extract_bb.application import run_extract
from pipelines.extract_bb.domain import ExtractConfig
from pipelines.shared.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(
        description="Extract bird boxes with dfine-cpp and build a YOLO dataset"
    )
    p.add_argument("--manifest", type=str, default=str(paths.manifest))
    p.add_argument("--species", type=str, default=str(paths.species_file))
    p.add_argument("--dataset", type=str, default=str(paths.dataset))
    p.add_argument(
        "--engine",
        type=str,
        required=True,
        help="Path to dfine TensorRT engine (e.g. dfine_m_slim.engine)",
    )
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument(
        "--keep-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep all bird boxes (default: true). Use --no-keep-all for highest-score only.",
    )
    p.add_argument(
        "--highest-only",
        action="store_true",
        help="Alias for --no-keep-all: keep only the highest-scoring bird box",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Max image side written to the dataset (downscale Full HD to model size)",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality for resized dataset images (1-95)",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ExtractConfig(
        manifest_path=Path(args.manifest),
        species_path=Path(args.species),
        dataset_root=Path(args.dataset),
        engine_path=Path(args.engine),
        threshold=args.threshold,
        keep_all=args.keep_all and not args.highest_only,
        imgsz=args.imgsz,
        jpeg_quality=args.jpeg_quality,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    run_extract(config)


if __name__ == "__main__":
    main()
