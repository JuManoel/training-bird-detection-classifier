# Data

## Sources

- **Species list:** [`data/spicies.txt`](../data/spicies.txt) — `common_name,scientific_name` (100 target species).
- **Macaulay export:** [`data/ML__*_photo_CO-CAL.csv`](../data/) — photo metadata for Caldas, Colombia (`regionCode=CO-CAL`). Export from [Macaulay Library search](https://search.macaulaylibrary.org/catalog?regionCode=CO-CAL&view=list).
- **Download CDN:** `https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{ML_Catalog_Number}/`

Respect [Macaulay Library media use guidelines](https://support.ebird.org/en/support/solutions/articles/48001064551-using-and-requesting-media) for academic/non-commercial research.

## Artifacts layout

```
artifacts/
  images/raw/{Scientific_Name}/{catalog_id}.jpg
  manifest.csv
  dataset/
    images/{train,val}/
    labels/{train,val}/
    rejected/
    classes.txt
    data.yaml
  runs/train/          # Ultralytics run + best.pt + plots/
  runs/predict/
```

## Pipeline

```bash
uv run avesia-download
# filters CSV by spicies.txt and writes artifacts/manifest.csv
```

Of the ~10k CSV rows, expect ~3.2k photos overlapping the species list (~87 of 100 species present in the export).
