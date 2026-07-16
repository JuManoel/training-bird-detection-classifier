# Models

## Training / prediction — YOLO26

- Hub: [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26)
- Framework: PyTorch via `ultralytics`
- Task: multi-class object detection (one class per scientific species)
- Defaults: `yolo26x.pt`, batch **8** (`-1` = auto-batch), workers **2**, AMP **on**, cache **off**, mosaic **1.0** / close_mosaic **5**, optimizer **AdamW**, train/val **80/20** stratified, early stopping patience = **5% of epochs**, checkpoint **`artifacts/runs/train/best.pt`** by lowest validation loss
- Detection loss: Ultralytics YOLO composite (box + cls + DFL)
- Extra metrics: accuracy, macro precision/recall/F1 + confusion heatmap under `artifacts/runs/train/plots/` (skip with `--skip-cls-eval`)
- Memory: training model is freed before classification eval; prefer lower `--batch` / `--workers` or a smaller `--model` (`yolo26m.pt`, `yolo26s.pt`) if CUDA/host RAM OOMs

```bash
uv run avesia-train --epochs 100 --optimizer AdamW
# Anti-OOM recipe:
uv run avesia-train --batch 8 --workers 2 --model yolo26m.pt
# or: --batch -1
uv run avesia-predict --source path/to/image.jpg
```

## Auto-labeling — dfine-cpp

- Repo: [PogChamper/dfine-cpp](https://github.com/PogChamper/dfine-cpp)
- Used **only for inference** (COCO `bird` boxes). Does not train.
- Requires NVIDIA GPU + TensorRT 10.x.

### Setup

```bash
uv sync
# Install the platform wheel from GitHub Releases, e.g.:
uv pip install "tensorrt-cu12==10.13.*" --extra-index-url https://pypi.nvidia.com
uv pip install "dfine @ https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine-0.3.3-py3-none-linux_x86_64.whl"

# Download ONNX + build a local engine (one-time):
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine_m_slim.onnx
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine_m_slim.json
uv run dfine build --model m --onnx dfine_m_slim.onnx --output artifacts/dfine_m_slim.engine

uv run avesia-extract --engine artifacts/dfine_m_slim.engine
# optional: --imgsz 640 --jpeg-quality 90 --highest-only
```

Species class ids come from the CSV/scientific name. Box geometry comes from all COCO `bird` detections by default (`--highest-only` keeps only the top score). Dataset images are full frames downscaled so `max(side) <= --imgsz` (default 640), matching Full HD camera → model input without requiring zoom.
