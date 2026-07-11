"""Macaulay CSV parsing and download manifest I/O."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaRecord:
    catalog_id: str
    scientific_name: str
    common_name: str
    format: str = "Photo"


@dataclass
class ManifestEntry:
    catalog_id: str
    scientific_name: str
    common_name: str
    image_path: str
    status: str = "ok"


def _normalize_header(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def load_macaulay_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
    photos_only: bool = True,
) -> list[MediaRecord]:
    """Parse Macaulay export CSV and optionally filter by species list."""
    records: list[MediaRecord] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldmap = {_normalize_header(k): k for k in reader.fieldnames}

        def col(row: dict[str, str], logical: str) -> str:
            key = fieldmap.get(logical)
            if key is None:
                raise KeyError(f"Missing column {logical!r} in {path}")
            return (row.get(key) or "").strip()

        for row in reader:
            fmt = col(row, "Format")
            if photos_only and fmt.lower() != "photo":
                continue
            scientific = col(row, "Scientific Name")
            if allowed_scientific is not None and scientific not in allowed_scientific:
                continue
            catalog_id = col(row, "ML Catalog Number")
            if not catalog_id or not scientific:
                continue
            records.append(
                MediaRecord(
                    catalog_id=catalog_id,
                    scientific_name=scientific,
                    common_name=col(row, "Common Name"),
                    format=fmt or "Photo",
                )
            )
    return records


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "catalog_id",
                "scientific_name",
                "common_name",
                "image_path",
                "status",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def read_manifest(path: Path) -> list[ManifestEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    entries: list[ManifestEntry] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            entries.append(
                ManifestEntry(
                    catalog_id=row["catalog_id"],
                    scientific_name=row["scientific_name"],
                    common_name=row["common_name"],
                    image_path=row["image_path"],
                    status=row.get("status", "ok"),
                )
            )
    return entries
