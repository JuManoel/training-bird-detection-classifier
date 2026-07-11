"""CLI for YOLO26 prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.predict_model.application import run_predict
from pipelines.predict_model.domain import PredictConfig
from pipelines.shared.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(description="Predict with the best YOLO26 bird detector")
    p.add_argument("--weights", type=str, default=str(paths.best_checkpoint))
    p.add_argument("--source", type=str, required=True, help="Image file or directory")
    p.add_argument("--out", type=str, default=str(paths.predict_out))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = PredictConfig(
        weights=Path(args.weights),
        source=Path(args.source),
        output_dir=Path(args.out),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    run_predict(config)


if __name__ == "__main__":
    main()
