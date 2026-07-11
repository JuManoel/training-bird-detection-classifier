"""Domain types for the download pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pipelines.shared.csv_manifest import MediaRecord


@dataclass(frozen=True)
class DownloadConfig:
    csv_path: Path
    species_path: Path
    output_dir: Path
    manifest_path: Path
    max_workers: int = 8
    skip_existing: bool = True
    timeout_s: float = 60.0
    max_retries: int = 3


@dataclass(frozen=True)
class DownloadResult:
    record: MediaRecord
    image_path: Path | None
    status: str
    error: str | None = None
