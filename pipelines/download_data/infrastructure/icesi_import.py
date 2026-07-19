"""Optional Icesi / local institutional CSV importer."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pipelines.shared.csv_manifest import MediaRecord, load_generic_media_csv


def load_icesi_csv(
    path: Path,
    allowed_scientific: set[str] | None = None,
    *,
    resolve_allowed: Callable[[str], str | None] | None = None,
) -> list[MediaRecord]:
    """Load an Icesi-provided media CSV (generic schema).

    Place exports under ``data/icesi_*.csv`` with columns such as
    ``catalog_id,scientific_name,common_name,url,license``.
    """
    return load_generic_media_csv(
        path,
        allowed_scientific=allowed_scientific,
        resolve_allowed=resolve_allowed,
        default_fuente="icesi",
    )
