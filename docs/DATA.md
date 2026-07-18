# Data

## Sources

- **Species list:** [`data/spicies.txt`](../data/spicies.txt) — `common_name,scientific_name` (~938 target species: original 100 plus species from the download CSV).
- **Macaulay export:** [`data/ML__*_photo_CO-CAL.csv`](../data/) — photo metadata for Caldas, Colombia (`regionCode=CO-CAL`). Export from [Macaulay Library search](https://search.macaulaylibrary.org/catalog?regionCode=CO-CAL&view=list).
- **Unified download CSV:** [`data/aves_descarga_v2.csv`](../data/aves_descarga_v2.csv) — `asset_id,nombre_cientifico,nombre_comun,species_code,url,fuente` (Macaulay + iNaturalist; iNaturalist rows include a direct `url`).
- **Download CDN (Macaulay):** `https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{ML_Catalog_Number}/`

`avesia-download` loads both CSVs by default, filters by `spicies.txt`, and dedupes by asset/catalog id. When a species appears in both datasets, images from both sources are kept.

Respect [Macaulay Library media use guidelines](https://support.ebird.org/en/support/solutions/articles/48001064551-using-and-requesting-media) for academic/non-commercial research.

## Artifacts layout

```
artifacts/
  images/raw/{Scientific_Name}/{catalog_id}.jpg   # originals (Full HD / source resolution)
  manifest.csv
  dataset/
    images/{train,val}/   # full frames downscaled so max(side) <= imgsz (default 640)
    labels/{train,val}/
    rejected/
    classes.txt
    data.yaml
  runs/train/          # Ultralytics run + best.pt + plots/
  runs/predict/
```

Raw downloads stay at source resolution. `avesia-extract` detects boxes on the original, then writes **resized** JPEG full frames into the YOLO dataset (proportional downscale, no upscale) so training matches production cameras (Full HD → model input). By default **all** COCO `bird` boxes are kept (multi-bird scenes); use `--highest-only` for a single box.

## Pipeline

```bash
uv run avesia-download
# filters default CSVs by spicies.txt and writes artifacts/manifest.csv
# override sources: uv run avesia-download --csv path/a.csv --csv path/b.csv

uv run avesia-extract
# optional: --model yolo26m.pt --device 0 --imgsz 640 --highest-only
```
