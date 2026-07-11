"""Species list loading and class-id mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeciesRecord:
    common_name: str
    scientific_name: str

    @property
    def folder_name(self) -> str:
        return self.scientific_name.replace(" ", "_")


def load_species_list(path: Path) -> list[SpeciesRecord]:
    """Load ``common,scientific`` lines from ``spicies.txt``."""
    records: list[SpeciesRecord] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",", maxsplit=1)]
        if len(parts) != 2 or not parts[1]:
            raise ValueError(f"Invalid species line in {path}: {raw!r}")
        records.append(SpeciesRecord(common_name=parts[0], scientific_name=parts[1]))
    if not records:
        raise ValueError(f"No species found in {path}")
    return records


def scientific_to_id(records: list[SpeciesRecord]) -> dict[str, int]:
    """Stable class id by list order (0..N-1)."""
    return {r.scientific_name: i for i, r in enumerate(records)}


def id_to_scientific(records: list[SpeciesRecord]) -> dict[int, str]:
    return {i: r.scientific_name for i, r in enumerate(records)}
