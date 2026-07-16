"""Media CSV parsing and download manifest I/O."""

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
    url: str | None = None
    fuente: str = "macaulay"


@dataclass
class ManifestEntry:
    catalog_id: str
    scientific_name: str
    common_name: str
    image_path: str
    status: str = "ok"


def _normalize_header(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def _fieldmap(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {_normalize_header(k): k for k in fieldnames}


def load_macaulay_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
    photos_only: bool = True,
) -> list[MediaRecord]:
    """Parse Macaulay export CSV and optionally filter by species list."""
    records: list[MediaRecord] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldmap = _fieldmap(reader.fieldnames)
        if "ML Catalog Number" not in fieldmap:
            raise ValueError(f"Not a Macaulay export CSV (missing ML Catalog Number): {path}")

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
                    url=None,
                    fuente="macaulay",
                )
            )
    return records


def load_aves_descarga_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
) -> list[MediaRecord]:
    """Parse aves_descarga_v2-style CSV (asset_id / url / fuente)."""
    records: list[MediaRecord] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldmap = _fieldmap(reader.fieldnames)
        required = ("asset_id", "nombre_cientifico", "nombre_comun")
        missing = [c for c in required if c not in fieldmap]
        if missing:
            raise ValueError(f"Not an aves_descarga CSV (missing {missing}): {path}")

        def col(row: dict[str, str], logical: str) -> str:
            key = fieldmap.get(logical)
            if key is None:
                return ""
            return (row.get(key) or "").strip()

        for row in reader:
            scientific = col(row, "nombre_cientifico")
            if allowed_scientific is not None and scientific not in allowed_scientific:
                continue
            catalog_id = col(row, "asset_id")
            if not catalog_id or not scientific:
                continue
            url = col(row, "url") or None
            fuente = col(row, "fuente") or ("inaturalist" if url else "macaulay")
            records.append(
                MediaRecord(
                    catalog_id=catalog_id,
                    scientific_name=scientific,
                    common_name=col(row, "nombre_comun"),
                    format="Photo",
                    url=url,
                    fuente=fuente,
                )
            )
    return records


def detect_csv_format(path: Path) -> str:
    """Return ``macaulay``, ``aves_descarga``, or raise if unknown."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = {_normalize_header(h) for h in (reader.fieldnames or [])}
    if "ML Catalog Number" in headers and "Scientific Name" in headers:
        return "macaulay"
    if "asset_id" in headers and "nombre_cientifico" in headers:
        return "aves_descarga"
    raise ValueError(f"Unrecognized media CSV schema: {path}")


def load_media_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
) -> list[MediaRecord]:
    """Load a media CSV, auto-detecting Macaulay vs aves_descarga schema."""
    kind = detect_csv_format(path)
    if kind == "macaulay":
        return load_macaulay_csv(path, allowed_scientific=allowed_scientific)
    return load_aves_descarga_csv(path, allowed_scientific=allowed_scientific)


def merge_media_records(records: list[MediaRecord]) -> list[MediaRecord]:
    """Dedupe by catalog_id; first occurrence wins."""
    seen: set[str] = set()
    merged: list[MediaRecord] = []
    for record in records:
        if record.catalog_id in seen:
            continue
        seen.add(record.catalog_id)
        merged.append(record)
    return merged


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
