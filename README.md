# avesia-yolo

Multi-class bird detection pipelines for Caldas (Colombia) media from Macaulay Library and iNaturalist.

1. **download** — photos for species in `data/spicies.txt` (ML export + `aves_descarga_v2.csv`)
2. **extract** — COCO `bird` boxes via YOLO26x → species labels
3. **train** — YOLO26x (AdamW, early stop 5% epochs, best by val loss)
4. **predict** — run `best.pt`

## Quickstart

```bash
uv sync
uv run avesia-download
uv run avesia-extract
uv run avesia-train --epochs 100
uv run avesia-predict --source path/to/image.jpg
```

See [docs/DATA.md](docs/DATA.md) and [docs/MODELS.md](docs/MODELS.md).
