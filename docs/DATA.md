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
| iNaturalist | Rows in `aves_descarga_v2.csv` + optional `--fetch-inat` API |
| GBIF / SiB Colombia | Optional `--fetch-gbif` StillImage API (SiB records appear in GBIF) |
| Icesi | Optional `data/icesi*.csv` generic media CSV |
| eBird API | Not used for photos; use Macaulay exports instead |

Respect [Macaulay media guidelines](https://support.ebird.org/en/support/solutions/articles/48001064551-using-and-requesting-media) and Creative Commons licenses on iNat/GBIF. Incompatible licenses are filtered when present; legacy Macaulay rows without a license column are kept under the project's academic-use policy.

## Thresholds

- **Target:** 500 images per species (cap while sampling across sources)
- **Minimum to train:** 125 unique crops after detection (80/20 → ≥100 train)
- Coverage report: `artifacts/coverage.json` + `artifacts/coverage.csv`

## Artifacts layout

```
artifacts/
  images/raw/{Scientific_Name}/{catalog_id}.jpg
  manifest.csv                 # rich metadata (source, license, hash, …)
  coverage.json
  dataset/
    detect/                    # single-class bird YOLO dataset
      images/{train,val}/
      labels/{train,val}/
      classes.txt              # bird
      data.yaml
      rejected/
    classify/                  # ImageFolder crops for classifiers
      train/{Scientific_Name}/*.jpg   # always 256×256
      val/{Scientific_Name}/*.jpg
      classes.txt
      crops_manifest.csv
  runs/
    train/                     # optional detector fine-tune
    train_cls/{resnet18,vgg16,yolo26x_cls}/
    predict/
```

## Pipeline

```bash
uv run avesia-download
# optional APIs for species below the 500 target:
uv run avesia-download --fetch-inat --fetch-gbif

uv run avesia-extract
# --crop-size 256 --min-images 125 --pad-ratio 0.1
```

`avesia-extract` keeps the **highest-scoring** bird box per image (multi-bird scenes are recorded but only the top box becomes a crop). Pseudo-labels come from the image's species folder/manifest — review rejected/multi-bird cases when adding noisy sources.
