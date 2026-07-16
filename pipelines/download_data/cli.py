"""CLI entrypoint for the download pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipelines.download_data.application import run_download
from pipelines.download_data.domain import DownloadConfig
from pipelines.shared.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    paths = ProjectPaths.from_root()
    p = argparse.ArgumentParser(
        description="Download bird photos (Macaulay export and/or aves_descarga CSV)"
    )
    p.add_argument(
        "--csv",
        type=str,
        action="append",
        default=None,
        help=(
            "Media CSV path (repeatable). "
            "Default: Macaulay export + data/aves_descarga_v2.csv"
        ),
    )
    p.add_argument(
        "--species",
        type=str,
        default=str(paths.species_file),
        help="Species list (common,scientific)",
    )
    p.add_argument("--out", type=str, default=str(paths.images_raw), help="Raw images directory")
    p.add_argument(
        "--manifest",
        type=str,
        default=str(paths.manifest),
        help="Output manifest CSV path",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--retries", type=int, default=3)
    return p


def main(argv: list[str] | None = None) -> None:
    paths = ProjectPaths.from_root()
    args = build_parser().parse_args(argv)
    csv_paths = (
        tuple(Path(p) for p in args.csv)
        if args.csv
        else (paths.default_csv, paths.default_download_csv)
    )
    config = DownloadConfig(
        csv_paths=csv_paths,
        species_path=Path(args.species),
        output_dir=Path(args.out),
        manifest_path=Path(args.manifest),
        max_workers=args.workers,
        skip_existing=not args.no_skip_existing,
        timeout_s=args.timeout,
        max_retries=args.retries,
    )
    run_download(config)


if __name__ == "__main__":
    main()
