"""Unit tests for taxonomy normalization."""

from pipelines.shared.taxonomy import normalize_scientific_name, species_folder_name


def test_normalize_subspecies():
    assert normalize_scientific_name("Ardea alba modesta") == "Ardea alba"


def test_normalize_group():
    assert (
        normalize_scientific_name("Camptostoma obsoletum [obsoletum Group]")
        == "Camptostoma obsoletum"
    )


def test_normalize_domestic():
    assert (
        normalize_scientific_name("Anas platyrhynchos (Domestic type)")
        == "Anas platyrhynchos"
    )


def test_folder_name():
    assert species_folder_name("Turdus fuscater") == "Turdus_fuscater"
