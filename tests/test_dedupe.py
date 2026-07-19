"""Dedup and per-species cap tests."""

from pipelines.shared.csv_manifest import MediaRecord
from pipelines.shared.dedupe import cap_per_species, dedupe_media_records


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
