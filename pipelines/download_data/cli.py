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
        description=(
            "Download bird photos for Colombian catalog species "
            "(Macaulay / iNaturalist CSVs + iNat/GBIF APIs by default)"
        )
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
        help="Species list (common,scientific) used to bootstrap the catalog",
    )
    p.add_argument(
        "--catalog",
        type=str,
        default=str(paths.species_catalog),
        help="Canonical Colombian species catalog JSON",
    )
    p.add_argument("--out", type=str, default=str(paths.images_raw), help="Raw images directory")
    p.add_argument(
        "--manifest",
        type=str,
        default=str(paths.manifest),
        help="Output manifest CSV path",
    )
    p.add_argument("--coverage-json", type=str, default=str(paths.coverage_json))
    p.add_argument("--coverage-csv", type=str, default=str(paths.coverage_csv))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument(
        "--max-per-species",
        type=int,
        default=2000,
        help="Max candidate downloads per species before detection (default: 2000)",
    )
    p.add_argument(
        "--min-images",
        type=int,
        default=125,
        help="Coverage advisory minimum (final gate is post-detection crops)",
    )
    p.add_argument(
        "--target-images",
        type=int,
        default=500,
        help="Coverage target for deciding which species need API top-up",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-fetch-inat",
        action="store_true",
        help="Disable iNaturalist API fetching (enabled by default)",
    )
    p.add_argument(
        "--no-fetch-gbif",
        action="store_true",
        help="Disable GBIF API fetching (enabled by default)",
    )
    p.add_argument(
        "--fetch-all-species",
        action="store_true",
        help="API-fetch every catalog species (default: only those below target)",
    )
    p.add_argument(
        "--gbif-country",
        type=str,
        default=None,
        help="Optional GBIF country filter (e.g. CO). Default: no geographic filter",
    )
    p.add_argument(
        "--perceptual-max-hamming",
        type=int,
        default=5,
        help="Max aHash Hamming distance to treat two downloads as near-duplicates",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    paths = ProjectPaths.from_root()
    args = build_parser().parse_args(argv)
    csv_paths = (
        tuple(Path(p) for p in args.csv)
        if args.csv
        else (paths.default_csv, paths.default_download_csv)
    )
    # Auto-include Icesi exports when present
    icesi = sorted(paths.data.glob("icesi*.csv"))
    if not args.csv and icesi:
        csv_paths = csv_paths + tuple(icesi)

    config = DownloadConfig(
        csv_paths=csv_paths,
        species_path=Path(args.species),
        catalog_path=Path(args.catalog),
        output_dir=Path(args.out),
        manifest_path=Path(args.manifest),
        coverage_json=Path(args.coverage_json),
        coverage_csv=Path(args.coverage_csv),
        max_workers=args.workers,
        skip_existing=not args.no_skip_existing,
        timeout_s=args.timeout,
        max_retries=args.retries,
        max_per_species=args.max_per_species,
        min_images=args.min_images,
        target_images=args.target_images,
        seed=args.seed,
        fetch_inat=not args.no_fetch_inat,
        fetch_gbif=not args.no_fetch_gbif,
        fetch_only_below_target=not args.fetch_all_species,
        gbif_country=args.gbif_country,
        perceptual_max_hamming=args.perceptual_max_hamming,
    )
    run_download(config)


if __name__ == "__main__":
    main()
