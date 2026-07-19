"""Deduplicate media records across sources (IDs, hashes, perceptual)."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

from PIL import Image

from pipelines.shared.csv_manifest import MediaRecord
from pipelines.shared.taxonomy import normalize_scientific_name

T = TypeVar("T")


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


@dataclass(frozen=True)
class PerceptualFingerprint:
    """Structural aHash plus mean RGB so flat colors stay distinguishable."""

    ahash: int
    mean_r: int
    mean_g: int
    mean_b: int

    @property
    def bucket(self) -> int:
        return self.ahash >> 48


def _pixels(im: Image.Image) -> list[int]:
    if hasattr(im, "get_flattened_data"):
        return list(im.get_flattened_data())
    return list(im.getdata())


def average_hash(path: Path, hash_size: int = 8) -> int:
    """64-bit grayscale average hash (aHash)."""
    with Image.open(path) as im:
        im = im.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = _pixels(im)
    avg = sum(pixels) / float(len(pixels))
    bits = 0
    for i, value in enumerate(pixels):
        if value >= avg:
            bits |= 1 << i
    return bits


def perceptual_fingerprint(path: Path, hash_size: int = 8) -> PerceptualFingerprint:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        gray = im.convert("L")
        pixels = _pixels(gray)
        r, g, b = im.split()
        n = float(hash_size * hash_size)
        mean_r = int(sum(_pixels(r)) / n)
        mean_g = int(sum(_pixels(g)) / n)
        mean_b = int(sum(_pixels(b)) / n)
    avg = sum(pixels) / float(len(pixels))
    bits = 0
    for i, value in enumerate(pixels):
        if value >= avg:
            bits |= 1 << i
    return PerceptualFingerprint(
        ahash=bits, mean_r=mean_r, mean_g=mean_g, mean_b=mean_b
    )


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def fingerprints_near_duplicate(
    a: PerceptualFingerprint,
    b: PerceptualFingerprint,
    *,
    max_hamming: int = 5,
    max_mean_delta: int = 12,
) -> bool:
    if hamming_distance(a.ahash, b.ahash) > max_hamming:
        return False
    return (
        abs(a.mean_r - b.mean_r) <= max_mean_delta
        and abs(a.mean_g - b.mean_g) <= max_mean_delta
        and abs(a.mean_b - b.mean_b) <= max_mean_delta
    )


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
    max_per_species: int = 2000,
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
    for _species, group in by_species.items():
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


@dataclass(frozen=True)
class DedupeDrop(Generic[T]):
    dropped: T
    kept: T
    reason: str
    distance: int | None = None


@dataclass
class DedupeResult(Generic[T]):
    kept: list[T]
    dropped: list[DedupeDrop[T]]
    exact_duplicates: int = 0
    perceptual_duplicates: int = 0


def dedupe_by_content(
    items: list[T],
    *,
    path_of: Callable[[T], Path],
    group_key: Callable[[T], str] | None = None,
    max_hamming: int = 5,
    max_mean_delta: int = 12,
    hash_size: int = 8,
) -> DedupeResult[T]:
    """Drop exact (SHA-256) and near-duplicate (aHash + mean RGB) images.

    Items that share the same ``group_key`` (e.g. crops from the same source
    photo) are never treated as duplicates of each other.
    """
    kept: list[T] = []
    dropped: list[DedupeDrop[T]] = []
    exact = 0
    perceptual = 0

    seen_sha: dict[str, T] = {}
    # bucket -> list of (fingerprint, item, group)
    buckets: dict[int, list[tuple[PerceptualFingerprint, T, str]]] = defaultdict(list)

    for item in items:
        path = path_of(item)
        if not path.exists():
            kept.append(item)
            continue
        group = group_key(item) if group_key is not None else ""
        try:
            digest = file_sha256(path)
        except OSError:
            kept.append(item)
            continue

        prior_exact = seen_sha.get(digest)
        if prior_exact is not None:
            prior_group = ""
            if group_key is not None:
                prior_group = group_key(prior_exact)
            if not (group and group == prior_group):
                exact += 1
                dropped.append(
                    DedupeDrop(
                        dropped=item,
                        kept=prior_exact,
                        reason="exact_sha256",
                        distance=0,
                    )
                )
                continue

        try:
            fp = perceptual_fingerprint(path, hash_size=hash_size)
        except OSError:
            seen_sha[digest] = item
            kept.append(item)
            continue

        match: tuple[T, int] | None = None
        candidates = list(buckets.get(fp.bucket, []))
        for neighbor in (fp.bucket - 1, fp.bucket + 1):
            candidates.extend(buckets.get(neighbor, []))
        for prior_fp, prior_item, prior_group in candidates:
            if group and group == prior_group:
                continue
            if fingerprints_near_duplicate(
                fp,
                prior_fp,
                max_hamming=max_hamming,
                max_mean_delta=max_mean_delta,
            ):
                match = (prior_item, hamming_distance(fp.ahash, prior_fp.ahash))
                break

        if match is not None:
            perceptual += 1
            dropped.append(
                DedupeDrop(
                    dropped=item,
                    kept=match[0],
                    reason="perceptual_ahash",
                    distance=match[1],
                )
            )
            continue

        seen_sha[digest] = item
        buckets[fp.bucket].append((fp, item, group))
        kept.append(item)

    return DedupeResult(
        kept=kept,
        dropped=dropped,
        exact_duplicates=exact,
        perceptual_duplicates=perceptual,
    )


def cap_items_per_species(
    items: list[T],
    *,
    species_of: Callable[[T], str],
    fuente_of: Callable[[T], str],
    max_per_species: int,
    seed: int = 42,
) -> list[T]:
    """Round-robin cap for arbitrary staged samples (crops)."""
    by_species: dict[str, list[T]] = defaultdict(list)
    for item in items:
        by_species[normalize_scientific_name(species_of(item))].append(item)

    rng = random.Random(seed)
    kept: list[T] = []
    for _species, group in by_species.items():
        if len(group) <= max_per_species:
            kept.extend(group)
            continue
        by_source: dict[str, list[T]] = defaultdict(list)
        for item in group:
            by_source[fuente_of(item) or "unknown"].append(item)
        for src_group in by_source.values():
            rng.shuffle(src_group)
        sources = sorted(by_source.keys())
        selected: list[T] = []
        idx = 0
        while len(selected) < max_per_species and any(by_source[s] for s in sources):
            src = sources[idx % len(sources)]
            idx += 1
            if by_source[src]:
                selected.append(by_source[src].pop())
        kept.extend(selected)
    return kept
