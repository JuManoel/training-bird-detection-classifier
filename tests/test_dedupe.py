"""Dedup, perceptual hash, and per-species cap tests."""

from pathlib import Path

from PIL import Image

from pipelines.shared.csv_manifest import MediaRecord
from pipelines.shared.dedupe import (
    average_hash,
    cap_items_per_species,
    cap_per_species,
    dedupe_by_content,
    dedupe_media_records,
    file_sha256,
    hamming_distance,
)


def _rec(cid: str, name: str, fuente: str = "macaulay", url: str | None = None) -> MediaRecord:
    return MediaRecord(
        catalog_id=cid,
        scientific_name=name,
        common_name=name,
        url=url,
        fuente=fuente,
    )


def test_dedupe_by_catalog_and_url():
    records = [
        _rec("1", "Turdus fuscater", url="http://a"),
        _rec("1", "Turdus fuscater", fuente="macaulay"),
        _rec("2", "Turdus fuscater", url="http://a"),
        _rec("3", "Turdus fuscater", url="http://b"),
    ]
    merged = dedupe_media_records(records)
    assert len(merged) == 2
    assert {r.catalog_id for r in merged} == {"1", "3"}


def test_cap_per_species():
    records = [_rec(str(i), "Turdus fuscater", fuente="a" if i % 2 == 0 else "b") for i in range(20)]
    capped = cap_per_species(records, max_per_species=5, seed=0)
    assert len(capped) == 5


def _solid_jpeg(path: Path, color: tuple[int, int, int], size: int = 64) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color).save(path, format="JPEG", quality=95)
    return path


def test_exact_and_perceptual_dedupe(tmp_path: Path):
    a = _solid_jpeg(tmp_path / "a.jpg", (200, 10, 10))
    b = _solid_jpeg(tmp_path / "b.jpg", (200, 10, 10))  # identical content
    c = _solid_jpeg(tmp_path / "c.jpg", (10, 200, 10))  # different

    class Item:
        def __init__(self, path: Path, cid: str):
            self.path = path
            self.cid = cid

    items = [Item(a, "a"), Item(b, "b"), Item(c, "c")]
    result = dedupe_by_content(items, path_of=lambda x: x.path, max_hamming=5)
    assert len(result.kept) == 2
    assert result.exact_duplicates + result.perceptual_duplicates >= 1
    assert {i.cid for i in result.kept} == {"a", "c"} or {i.cid for i in result.kept} == {
        "b",
        "c",
    }


def test_perceptual_group_key_protects_siblings(tmp_path: Path):
    """Crops from the same source photo must not dedupe against each other."""
    a = _solid_jpeg(tmp_path / "a.jpg", (120, 120, 120))
    b = _solid_jpeg(tmp_path / "b.jpg", (120, 120, 120))

    class Item:
        def __init__(self, path: Path, group: str, cid: str):
            self.path = path
            self.group = group
            self.cid = cid

    items = [Item(a, "photo1", "a"), Item(b, "photo1", "b")]
    result = dedupe_by_content(
        items,
        path_of=lambda x: x.path,
        group_key=lambda x: x.group,
        max_hamming=5,
    )
    assert len(result.kept) == 2


def test_average_hash_distance():
    # Same solid color => identical aHash
    assert hamming_distance(0b1010, 0b1000) == 1


def test_cap_items_per_species():
    class Item:
        def __init__(self, species: str, fuente: str, i: int):
            self.species = species
            self.fuente = fuente
            self.i = i

    items = [
        Item("Turdus fuscater", "a" if i % 2 == 0 else "b", i) for i in range(30)
    ]
    capped = cap_items_per_species(
        items,
        species_of=lambda x: x.species,
        fuente_of=lambda x: x.fuente,
        max_per_species=10,
        seed=1,
    )
    assert len(capped) == 10


def test_file_sha256_stable(tmp_path: Path):
    path = _solid_jpeg(tmp_path / "x.jpg", (1, 2, 3))
    assert file_sha256(path) == file_sha256(path)
    assert average_hash(path) == average_hash(path)
