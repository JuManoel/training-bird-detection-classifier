"""Media CSV parsing and download manifest I/O."""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from pipelines.shared.licenses import is_license_allowed
from pipelines.shared.taxonomy import normalize_scientific_name


@dataclass(frozen=True)
class MediaRecord:
    catalog_id: str
    scientific_name: str
    common_name: str
    format: str = "Photo"
    url: str | None = None
    fuente: str = "macaulay"
    observation_id: str | None = None
    license: str | None = None
    author: str | None = None
    taxon_id: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    event_date: str | None = None
    media_hash: str | None = None


@dataclass
class ManifestEntry:
    catalog_id: str
    scientific_name: str
    common_name: str
    image_path: str
    status: str = "ok"
    fuente: str = "macaulay"
    observation_id: str | None = None
    license: str | None = None
    author: str | None = None
    taxon_id: str | None = None
    latitude: str | None = None
    longitude: str | None = None
    event_date: str | None = None
    media_hash: str | None = None
    url: str | None = None
    crop_path: str | None = None
    box_w: str | None = None
    box_h: str | None = None
    reject_reason: str | None = None


MANIFEST_FIELDS = [f.name for f in fields(ManifestEntry)]


def _normalize_header(name: str) -> str:
    return name.lstrip("\ufeff").strip()


def _fieldmap(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        return {}
    return {_normalize_header(k): k for k in fieldnames}


def _maybe_normalize(
    scientific: str,
    *,
    normalize: bool,
) -> str:
    return normalize_scientific_name(scientific) if normalize else scientific.strip()


def load_macaulay_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
    photos_only: bool = True,
    *,
    normalize: bool = True,
    resolve_allowed: Callable[[str], str | None] | None = None,
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
                return ""
            return (row.get(key) or "").strip()

        for row in reader:
            fmt = col(row, "Format")
            if photos_only and fmt.lower() != "photo":
                continue
            raw_scientific = col(row, "Scientific Name")
            scientific = _maybe_normalize(raw_scientific, normalize=normalize)
            if resolve_allowed is not None:
                resolved = resolve_allowed(raw_scientific)
                if resolved is None:
                    continue
                scientific = resolved
            elif allowed_scientific is not None and scientific not in allowed_scientific:
                continue
            catalog_id = col(row, "ML Catalog Number")
            if not catalog_id or not scientific:
                continue
            license_raw = col(row, "License") or None
            if license_raw and not is_license_allowed(license_raw):
                continue
            records.append(
                MediaRecord(
                    catalog_id=catalog_id,
                    scientific_name=scientific,
                    common_name=col(row, "Common Name"),
                    format=fmt or "Photo",
                    url=None,
                    fuente="macaulay",
                    observation_id=col(row, "eBird Checklist ID") or None,
                    license=license_raw,
                    author=col(row, "Recordist") or col(row, "Observer") or None,
                    taxon_id=col(row, "eBird Species Code") or None,
                    latitude=col(row, "Latitude") or None,
                    longitude=col(row, "Longitude") or None,
                    event_date=col(row, "Date") or None,
                )
            )
    return records


def load_aves_descarga_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
    *,
    normalize: bool = True,
    resolve_allowed: Callable[[str], str | None] | None = None,
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
            raw_scientific = col(row, "nombre_cientifico")
            scientific = _maybe_normalize(raw_scientific, normalize=normalize)
            if resolve_allowed is not None:
                resolved = resolve_allowed(raw_scientific)
                if resolved is None:
                    continue
                scientific = resolved
            elif allowed_scientific is not None and scientific not in allowed_scientific:
                continue
            catalog_id = col(row, "asset_id")
            if not catalog_id or not scientific:
                continue
            url = col(row, "url") or None
            fuente = col(row, "fuente") or ("inaturalist" if url else "macaulay")
            license_raw = col(row, "license") or col(row, "licencia") or None
            if license_raw and not is_license_allowed(license_raw):
                continue
            records.append(
                MediaRecord(
                    catalog_id=catalog_id,
                    scientific_name=scientific,
                    common_name=col(row, "nombre_comun"),
                    format="Photo",
                    url=url,
                    fuente=fuente,
                    observation_id=col(row, "observation_id") or None,
                    license=license_raw,
                    author=col(row, "author") or col(row, "autor") or None,
                    taxon_id=col(row, "species_code") or col(row, "taxon_id") or None,
                    latitude=col(row, "latitude") or col(row, "lat") or None,
                    longitude=col(row, "longitude") or col(row, "lon") or None,
                    event_date=col(row, "date") or col(row, "event_date") or None,
                )
            )
    return records


def load_generic_media_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
    *,
    normalize: bool = True,
    resolve_allowed: Callable[[str], str | None] | None = None,
    default_fuente: str = "icesi",
) -> list[MediaRecord]:
    """Parse a flexible CSV used for Icesi / GBIF exports.

    Expected columns (any subset beyond the required ones):
    ``catalog_id|asset_id|gbifID``, ``scientific_name|nombre_cientifico``,
    ``common_name|nombre_comun``, ``url|identifier``, ``license``, ``fuente``.
    """
    records: list[MediaRecord] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldmap = _fieldmap(reader.fieldnames)

        def col(row: dict[str, str], *logicals: str) -> str:
            for logical in logicals:
                key = fieldmap.get(logical)
                if key is not None:
                    return (row.get(key) or "").strip()
            return ""

        for row in reader:
            raw_scientific = col(row, "scientific_name", "nombre_cientifico", "species")
            scientific = _maybe_normalize(raw_scientific, normalize=normalize)
            if resolve_allowed is not None:
                resolved = resolve_allowed(raw_scientific)
                if resolved is None:
                    continue
                scientific = resolved
            elif allowed_scientific is not None and scientific not in allowed_scientific:
                continue
            catalog_id = col(row, "catalog_id", "asset_id", "gbifID", "id", "media_id")
            url = col(row, "url", "identifier", "media_url") or None
            if not catalog_id or not scientific:
                continue
            if not url and not catalog_id:
                continue
            license_raw = col(row, "license", "licencia", "licenseRights") or None
            if license_raw and not is_license_allowed(license_raw):
                continue
            fuente = col(row, "fuente", "source", "datasetName") or default_fuente
            records.append(
                MediaRecord(
                    catalog_id=catalog_id,
                    scientific_name=scientific,
                    common_name=col(row, "common_name", "nombre_comun", "vernacularName"),
                    format="Photo",
                    url=url,
                    fuente=fuente.lower(),
                    observation_id=col(row, "observation_id", "occurrenceID") or None,
                    license=license_raw,
                    author=col(row, "author", "rightsHolder", "recordedBy") or None,
                    taxon_id=col(row, "taxon_id", "taxonKey", "speciesKey") or None,
                    latitude=col(row, "latitude", "decimalLatitude", "lat") or None,
                    longitude=col(row, "longitude", "decimalLongitude", "lon") or None,
                    event_date=col(row, "event_date", "eventDate", "date") or None,
                )
            )
    return records


def detect_csv_format(path: Path) -> str:
    """Return ``macaulay``, ``aves_descarga``, ``generic``, or raise if unknown."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = {_normalize_header(h) for h in (reader.fieldnames or [])}
    if "ML Catalog Number" in headers and "Scientific Name" in headers:
        return "macaulay"
    if "asset_id" in headers and "nombre_cientifico" in headers:
        return "aves_descarga"
    sci = {"scientific_name", "nombre_cientifico", "species"} & headers
    ids = {"catalog_id", "asset_id", "gbifID", "id", "media_id"} & headers
    if sci and ids:
        return "generic"
    raise ValueError(f"Unrecognized media CSV schema: {path}")


def load_media_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
    *,
    normalize: bool = True,
    resolve_allowed: Callable[[str], str | None] | None = None,
) -> list[MediaRecord]:
    """Load a media CSV, auto-detecting schema."""
    kind = detect_csv_format(path)
    if kind == "macaulay":
        return load_macaulay_csv(
            path,
            allowed_scientific=allowed_scientific,
            normalize=normalize,
            resolve_allowed=resolve_allowed,
        )
    if kind == "aves_descarga":
        return load_aves_descarga_csv(
            path,
            allowed_scientific=allowed_scientific,
            normalize=normalize,
            resolve_allowed=resolve_allowed,
        )
    return load_generic_media_csv(
        path,
        allowed_scientific=allowed_scientific,
        normalize=normalize,
        resolve_allowed=resolve_allowed,
    )


def merge_media_records(records: list[MediaRecord]) -> list[MediaRecord]:
    """Dedupe by catalog_id; first occurrence wins (legacy helper)."""
    seen: set[str] = set()
    merged: list[MediaRecord] = []
    for record in records:
        key = f"{record.fuente}:{record.catalog_id}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)
    return merged


MEDIA_RECORD_FIELDS = [f.name for f in fields(MediaRecord)]


def write_media_csv(path: Path, records: list[MediaRecord]) -> None:
    """Persist MediaRecords in the generic schema ``load_media_csv`` can reload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MEDIA_RECORD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow(asdict(entry))


def read_manifest(path: Path) -> list[ManifestEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    entries: list[ManifestEntry] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            kwargs = {name: row.get(name) or None for name in MANIFEST_FIELDS}
            # required fields
            kwargs["catalog_id"] = row["catalog_id"]
            kwargs["scientific_name"] = row["scientific_name"]
            kwargs["common_name"] = row.get("common_name") or ""
            kwargs["image_path"] = row["image_path"]
            kwargs["status"] = row.get("status") or "ok"
            kwargs["fuente"] = row.get("fuente") or "macaulay"
            entries.append(ManifestEntry(**kwargs))
    return entries
