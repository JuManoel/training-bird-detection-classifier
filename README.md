# avesia-yolo

Two-stage bird pipeline for Colombian avifauna:

1. **download** — photos for species in the Colombian catalog (Macaulay / iNaturalist CSVs + **iNat & GBIF APIs on by default**; optional Icesi CSV)
2. **extract** — YOLO detects every `bird` box → 1-class detection dataset + **256×256** species crops for classification
3. **train** — optional fine-tune of the single-class bird detector
4. **train-cls** — train/compare **ResNet18**, **VGG16**, and **YOLO26x-cls** on crops
5. **predict** — detect birds, crop, classify species (optionally compare all three classifiers)

## Quickstart

```bash
uv sync --extra dev

# Build/update catalog + download (CSVs + iNat/GBIF APIs)
uv run avesia-download
# Disable APIs if needed:
# uv run avesia-download --no-fetch-inat --no-fetch-gbif

uv run avesia-extract
# optional detector fine-tune:
uv run avesia-train --epochs 50

# Train one classifier or all three
uv run avesia-train-cls --architecture resnet18 --epochs 50
uv run avesia-train-cls --architecture all --epochs 50

uv run avesia-predict --source path/to/image.jpg --architecture resnet18
uv run avesia-predict --source path/to/image.jpg --compare
```

## Dataset policy

- Classes = species listed for Colombia (canonical binomials; subspecies collapsed).
- Photos may come from any country.
- **Thresholds apply after detection**, on unique crops (exact + perceptual dedupe):
  - **minimum 125** crops/species to include a class
  - **target ≈ 500** crops/species
  - **hard cap 1000** crops/species
- Each YOLO bird box becomes one crop: 5 birds in one photo → +5 toward that species.
- Classification trains **only** on YOLO bird crops resized to **256×256**.
- Species that never reach 125 unique crops are excluded and reported.

See [docs/DATA.md](docs/DATA.md) and [docs/MODELS.md](docs/MODELS.md).

## Tests

```bash
uv run pytest -q
```
