# avesia-yolo

Multi-class bird detection pipelines for Caldas (Colombia) media from Macaulay Library and iNaturalist.

1. **download** — photos for species in `data/spicies.txt` (ML export + `aves_descarga_v2.csv`)
2. **extract** — bird boxes via dfine-cpp → YOLO labels
3. **train** — YOLO26x (AdamW, early stop 5% epochs, best by val loss)
4. **predict** — run `best.pt`

## Quickstart

```bash
uv sync
uv run avesia-download
# see docs/MODELS.md for dfine-cpp + TensorRT engine setup
uv run avesia-extract --engine artifacts/dfine_m_slim.engine
uv run avesia-train --epochs 100
uv run avesia-predict --source path/to/image.jpg
```

See [docs/DATA.md](docs/DATA.md) and [docs/MODELS.md](docs/MODELS.md).
