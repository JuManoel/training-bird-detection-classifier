# Models

## Training / prediction — YOLO26

- Hub: [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26)
- Framework: PyTorch via `ultralytics`
- Task: multi-class object detection (one class per scientific species)
- Defaults: `yolo26n.pt`, optimizer **AdamW**, train/val **80/20** stratified, early stopping patience = **5% of epochs**, checkpoint **`artifacts/runs/train/best.pt`** by lowest validation loss
- Detection loss: Ultralytics YOLO composite (box + cls + DFL)
- Extra metrics: accuracy, macro precision/recall/F1 + confusion heatmap under `artifacts/runs/train/plots/`

```bash
uv run avesia-train --epochs 100 --optimizer AdamW
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
dfine build --model m --onnx dfine_m_slim.onnx --output artifacts/dfine_m_slim.engine

uv run avesia-extract --engine artifacts/dfine_m_slim.engine
```

Species class ids come from the CSV/scientific name; box geometry comes from the highest-scoring COCO `bird` detection (unless `--keep-all`).
