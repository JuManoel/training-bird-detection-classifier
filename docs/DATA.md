# Data

## Catalog

- **Species list (bootstrap):** [`data/spicies.txt`](../data/spicies.txt) — `common_name,scientific_name`
- **Canonical catalog:** [`data/colombia_species_catalog.json`](../data/colombia_species_catalog.json) — binomials for Colombian-listed species, synonyms, optional GBIF/iNat/eBird IDs
- Names are normalized to species rank (subspecies, `[Group]`, `(Domestic type)` collapsed)

Photos of catalog species may come from **any country**. Xeno-canto is not used (audio only).

## Sources

| Source | How it is used |
|---|---|
| Macaulay / eBird exports | CSV (`ML__*_photo_CO-CAL.csv` and additional ML exports) |
| iNaturalist | Rows in `aves_descarga_v2.csv` + **iNat API (on by default)** |
| GBIF / SiB Colombia | **GBIF StillImage API (on by default)**; SiB records appear in GBIF |
| Icesi | Optional `data/icesi*.csv` generic media CSV (auto-included when present) |
| eBird API | Not used for photos; use Macaulay exports instead |

Disable APIs with `--no-fetch-inat` / `--no-fetch-gbif`.

Respect [Macaulay media guidelines](https://support.ebird.org/en/support/solutions/articles/48001064551-using-and-requesting-media) and Creative Commons licenses on iNat/GBIF. Incompatible licenses are filtered when present; legacy Macaulay rows without a license column are kept under the project's academic-use policy.

## Thresholds (post-detection crops)

All of **125 / 500 / 1000** are measured on **unique classification crops after YOLO**, not on raw downloads:

| Threshold | Meaning |
|---|---|
| **125** | Minimum unique crops to keep a species for training |
| **500** | Coverage target (reported in extract stats) |
| **1000** | Hard cap per species after crop dedupe |

Rules:

- Each detected bird box → one 256×256 crop. A photo with 5 birds contributes **5** crops.
- Exact (SHA-256) and perceptual (aHash) duplicates are removed so they do not inflate counts.
- Crops from the **same source photo** are never perceptual-deduped against each other (distinct birds in one frame).
- Species below 125 unique crops are **excluded** and listed in `rejected_manifest.csv` / `extract_stats.json`.
- Download still oversamples candidates (default up to **2000** images/species) so detection can fill the crop caps.

Coverage / dedupe reports:

- `artifacts/coverage.json` + `artifacts/coverage.csv` (download-time advisory)
- `artifacts/dataset/classify/crop_duplicates.csv` (post-extract crop dedupe)
- `artifacts/dataset/classify/extract_stats.json`

## Artifacts layout

```
artifacts/
  images/raw/{Scientific_Name}/{catalog_id}.jpg
  manifest.csv                 # rich metadata (source, license, hash, …)
  coverage.json
  dataset/
    detect/                    # single-class bird YOLO dataset
      images/{train,val}/
      labels/{train,val}/      # all bird boxes per image
      classes.txt              # bird
      data.yaml
      rejected/
    classify/                  # ImageFolder crops for classifiers
      train/{Scientific_Name}/*.jpg   # always 256×256; stem_b{i}.jpg per box
      val/{Scientific_Name}/*.jpg
      classes.txt
      crops_manifest.csv
      crop_duplicates.csv
      extract_stats.json
  runs/
    train/                     # optional detector fine-tune
    train_cls/{resnet18,vgg16,yolo26x_cls}/
    predict/
```

## Pipeline

```bash
uv run avesia-download
# uv run avesia-download --no-fetch-inat --no-fetch-gbif

uv run avesia-extract
# --crop-size 256 --min-images 125 --target-images 500 --max-per-species 1000
```

`avesia-extract` keeps **every** bird box above the score threshold (multi-bird scenes produce multiple crops). Pseudo-labels come from the image's species folder/manifest — review rejected/multi-bird cases when adding noisy sources.
