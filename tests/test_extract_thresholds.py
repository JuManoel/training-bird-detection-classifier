"""Post-detection crop threshold helpers."""

from collections import defaultdict


def filter_species_by_crop_count(
    crops_by_species: dict[str, list[str]],
    *,
    min_images: int = 125,
    max_per_species: int = 1000,
) -> tuple[dict[str, list[str]], list[str]]:
    """Keep species with enough crops; callers cap before calling this."""
    kept: dict[str, list[str]] = {}
    dropped: list[str] = []
    for species, crops in crops_by_species.items():
        if len(crops) < min_images:
            dropped.append(species)
            continue
        kept[species] = crops[:max_per_species]
    return kept, dropped


def test_exclude_below_min_and_cap():
    by_species = {
        "Turdus fuscater": [f"a{i}" for i in range(130)],
        "Habia cristata": [f"b{i}" for i in range(40)],
        "Diglossa cyanea": [f"c{i}" for i in range(1500)],
    }
    kept, dropped = filter_species_by_crop_count(
        by_species, min_images=125, max_per_species=1000
    )
    assert dropped == ["Habia cristata"]
    assert set(kept) == {"Turdus fuscater", "Diglossa cyanea"}
    assert len(kept["Diglossa cyanea"]) == 1000
    assert all(len(v) >= 125 for v in kept.values())


def test_five_birds_in_one_photo_count_as_five():
    """Simulates one source photo contributing five detection crops."""
    crops = [f"photo1_b{i}" for i in range(5)]
    by_species: dict[str, list[str]] = defaultdict(list)
    by_species["Turdus fuscater"].extend(crops)
    assert len(by_species["Turdus fuscater"]) == 5
