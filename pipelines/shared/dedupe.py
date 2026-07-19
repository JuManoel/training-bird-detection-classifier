"""Deduplicate media records across sources."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path

from pipelines.shared.csv_manifest import MediaRecord
from pipelines.shared.taxonomy import normalize_scientific_name


def record_identity_keys(record: MediaRecord) -> list[str]:
    """Stable identity keys for cross-source dedupe (first match wins)."""
    keys: list[str] = []
    src = (record.fuente or "unknown").lower()
    keys.append(f"{src}:{record.catalog_id}")
    if record.observation_id:
        keys.append(f"obs:{src}:{record.observation_id}")
    if record.url:
        keys.append(f"url:{record.url.strip().lower()}")
    if record.media_hash:
        keys.append(f"hash:{record.media_hash}")
    return keys


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dedupe_media_records(records: list[MediaRecord]) -> list[MediaRecord]:
    """Drop duplicates by catalog/url/observation/hash; first occurrence wins."""
    seen: set[str] = set()
    merged: list[MediaRecord] = []
    for record in records:
        keys = record_identity_keys(record)
        if any(k in seen for k in keys):
            continue
        for k in keys:
            seen.add(k)
        merged.append(record)
    return merged


def cap_per_species(
    records: list[MediaRecord],
    *,
    max_per_species: int = 500,
    seed: int = 42,
) -> list[MediaRecord]:
    """Keep at most ``max_per_species`` records per normalized species.

    Prefers diversity across sources via round-robin sampling when capping.
    """
    by_species: dict[str, list[MediaRecord]] = defaultdict(list)
    for record in records:
        key = normalize_scientific_name(record.scientific_name)
        by_species[key].append(record)

    rng = random.Random(seed)
    kept: list[MediaRecord] = []
    for species, group in by_species.items():
        if len(group) <= max_per_species:
            kept.extend(group)
            continue
        by_source: dict[str, list[MediaRecord]] = defaultdict(list)
        for r in group:
            by_source[r.fuente or "unknown"].append(r)
        for src_group in by_source.values():
            rng.shuffle(src_group)
        sources = sorted(by_source.keys())
        selected: list[MediaRecord] = []
        idx = 0
        while len(selected) < max_per_species and any(by_source[s] for s in sources):
            src = sources[idx % len(sources)]
            idx += 1
            if by_source[src]:
                selected.append(by_source[src].pop())
        kept.extend(selected)
    return kept
