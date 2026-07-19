"""Coverage and min-image threshold tests."""

from pipelines.shared.coverage import coverage_from_records, filter_included_names
from pipelines.shared.csv_manifest import MediaRecord


def test_coverage_min_threshold():
    records = [
        MediaRecord(
            catalog_id=str(i),
            scientific_name="Turdus fuscater",
            common_name="Mirla",
            fuente="macaulay",
        )
        for i in range(130)
    ] + [
        MediaRecord(
            catalog_id=f"x{i}",
            scientific_name="Habia cristata",
            common_name="Habia",
            fuente="inaturalist",
        )
        for i in range(10)
    ]
    rows = coverage_from_records(records, min_images=125, target_images=500)
    by_name = {r.scientific_name: r for r in rows}
    assert by_name["Turdus fuscater"].included
    assert not by_name["Habia cristata"].included
    assert filter_included_names(rows) == {"Turdus fuscater"}
