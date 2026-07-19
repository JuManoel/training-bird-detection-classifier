"""Colombian species catalog (canonical names + synonym map)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipelines.shared.species import SpeciesRecord, load_species_list
from pipelines.shared.taxonomy import normalize_scientific_name


@dataclass
class CatalogEntry:
    scientific_name: str
    common_name: str
    synonyms: list[str] = field(default_factory=list)
    gbif_taxon_id: str | None = None
    inat_taxon_id: str | None = None
    ebird_code: str | None = None
    colombia_listed: bool = True


@dataclass
class SpeciesCatalog:
    """Canonical Colombian species list with synonym resolution."""

    entries: list[CatalogEntry]
    _alias_to_canonical: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        index: dict[str, str] = {}
        for entry in self.entries:
            canonical = normalize_scientific_name(entry.scientific_name)
            index[canonical.lower()] = canonical
            index[entry.scientific_name.lower()] = canonical
            for syn in entry.synonyms:
                index[normalize_scientific_name(syn).lower()] = canonical
                index[syn.lower()] = canonical
        self._alias_to_canonical = index

    def resolve(self, scientific_name: str) -> str | None:
        """Map any alias/subspecies string to the catalog canonical binomial."""
        raw = (scientific_name or "").strip()
        if not raw:
            return None
        key = raw.lower()
        if key in self._alias_to_canonical:
            return self._alias_to_canonical[key]
        norm = normalize_scientific_name(raw).lower()
        return self._alias_to_canonical.get(norm)

    def allowed_scientific(self) -> set[str]:
        return {normalize_scientific_name(e.scientific_name) for e in self.entries}

    def contains(self, scientific_name: str) -> bool:
        return self.resolve(scientific_name) is not None

    def ordered_canonical(self) -> list[str]:
        return [normalize_scientific_name(e.scientific_name) for e in self.entries]

    def to_json(self) -> dict:
        return {"species": [asdict(e) for e in self.entries]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> SpeciesCatalog:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = [CatalogEntry(**row) for row in payload.get("species", [])]
        if not entries:
            raise ValueError(f"Empty catalog: {path}")
        return cls(entries=entries)

    @classmethod
    def from_species_txt(cls, path: Path) -> SpeciesCatalog:
        """Build a catalog from ``common,scientific`` lines, collapsing duplicates."""
        records = load_species_list(path)
        by_canon: dict[str, CatalogEntry] = {}
        for rec in records:
            canon = normalize_scientific_name(rec.scientific_name)
            if not canon:
                continue
            if canon not in by_canon:
                by_canon[canon] = CatalogEntry(
                    scientific_name=canon,
                    common_name=rec.common_name,
                    synonyms=[],
                    colombia_listed=True,
                )
            else:
                entry = by_canon[canon]
                raw = rec.scientific_name.strip()
                if raw != canon and raw not in entry.synonyms:
                    entry.synonyms.append(raw)
                if not entry.common_name and rec.common_name:
                    entry.common_name = rec.common_name
        return cls(entries=list(by_canon.values()))


def load_or_build_catalog(catalog_path: Path, species_txt: Path) -> SpeciesCatalog:
    """Load JSON catalog if present, otherwise build from species.txt and write it."""
    if catalog_path.exists():
        return SpeciesCatalog.from_json(catalog_path)
    catalog = SpeciesCatalog.from_species_txt(species_txt)
    catalog.save(catalog_path)
    return catalog


def catalog_as_species_records(catalog: SpeciesCatalog) -> list[SpeciesRecord]:
    return [
        SpeciesRecord(common_name=e.common_name, scientific_name=normalize_scientific_name(e.scientific_name))
        for e in catalog.entries
    ]
