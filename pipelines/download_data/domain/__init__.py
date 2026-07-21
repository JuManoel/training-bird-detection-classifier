"""Domain types for the download pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipelines.shared.csv_manifest import MediaRecord


@dataclass(frozen=True)
class DownloadConfig:
    csv_paths: tuple[Path, ...]
    species_path: Path
    catalog_path: Path
    output_dir: Path
    manifest_path: Path
    coverage_json: Path
    coverage_csv: Path
    max_workers: int = 8
    skip_existing: bool = True
    timeout_s: float = 60.0
    max_retries: int = 8
    # Candidate budget before detection (oversample so post-crop caps can fill).
    max_per_species: int = 2000
    min_images: int = 125
    target_images: int = 500
    seed: int = 42
    fetch_inat: bool = True
    fetch_gbif: bool = True
    fetch_only_below_target: bool = True
    gbif_country: str | None = None
    api_checkpoint_path: Path | None = None
    fresh_api_fetch: bool = False


@dataclass(frozen=True)
class DownloadResult:
    record: MediaRecord
    image_path: Path | None
    status: str
    error: str | None = None
    media_hash: str | None = None
