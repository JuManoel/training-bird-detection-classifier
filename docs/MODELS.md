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

## Auto-labeling — YOLO26x

- Hub: [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26)
- Uses the COCO-pretrained `yolo26x.pt` detector and keeps only the `bird` class.
- The detected boxes are relabeled with each image's scientific species from the manifest.
- Uses PyTorch/Ultralytics, so TensorRT and dfine-cpp are not required.

### Setup

```bash
uv sync
uv run avesia-extract
# optional: --model yolo26m.pt --device 0 --imgsz 640 --highest-only
```

`yolo26x.pt` is downloaded automatically on first use. Species class ids come from the CSV/scientific name. Box geometry comes from all COCO `bird` detections by default (`--highest-only` keeps only the top score). Dataset images are full frames downscaled so `max(side) <= --imgsz` (default 640), matching Full HD camera → model input without requiring zoom. Use a smaller COCO model such as `yolo26m.pt` if GPU memory is limited.
