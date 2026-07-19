"""Catalog and CSV loader tests."""

from pathlib import Path

from pipelines.shared.catalog import CatalogEntry, SpeciesCatalog
from pipelines.shared.csv_manifest import (
    ManifestEntry,
    load_media_csv,
    read_manifest,
    write_manifest,
)


def test_catalog_from_species_txt(tmp_path: Path):
    path = tmp_path / "spicies.txt"
    path.write_text(
        "Mirla,Turdus fuscater\n"
        "Garza,Ardea alba modesta\n"
        "Garza grande,Ardea alba\n",
        encoding="utf-8",
    )
    catalog = SpeciesCatalog.from_species_txt(path)
    names = catalog.allowed_scientific()
    assert "Turdus fuscater" in names
    assert "Ardea alba" in names
    assert catalog.resolve("Ardea alba modesta") == "Ardea alba"


def test_load_aves_descarga_and_normalize(tmp_path: Path):
    csv_path = tmp_path / "media.csv"
    csv_path.write_text(
        "asset_id,nombre_cientifico,nombre_comun,species_code,url,fuente\n"
        "1,Turdus fuscater quindio,Mirla,x,http://example.com/a.jpg,inaturalist\n"
        "2,Unknown bird,X,,http://example.com/b.jpg,inaturalist\n",
        encoding="utf-8",
    )
    catalog = SpeciesCatalog(
        entries=[CatalogEntry(scientific_name="Turdus fuscater", common_name="Mirla")]
    )
    records = load_media_csv(csv_path, resolve_allowed=catalog.resolve)
    assert len(records) == 1
    assert records[0].scientific_name == "Turdus fuscater"


def test_manifest_roundtrip(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    entries = [
        ManifestEntry(
            catalog_id="1",
            scientific_name="Turdus fuscater",
            common_name="Mirla",
            image_path="/tmp/a.jpg",
            status="ok",
            fuente="inaturalist",
            license="CC-BY",
            crop_path="/tmp/crop.jpg",
            box_w="40.5",
            box_h="50.0",
        )
    ]
    write_manifest(path, entries)
    loaded = read_manifest(path)
    assert loaded[0].catalog_id == "1"
    assert loaded[0].crop_path == "/tmp/crop.jpg"
    assert loaded[0].fuente == "inaturalist"
