# Models

## Stage 1 — Bird detector (YOLO26)

- Hub: [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26)
- Default weights: COCO-pretrained `yolo26x.pt` (class `bird`)
- Optional fine-tune on `artifacts/dataset/detect` (single class):

```bash
uv run avesia-train --epochs 50 --data artifacts/dataset/detect/data.yaml
```

## Stage 2 — Species classifiers

Trained on **256×256** bird crops under `artifacts/dataset/classify/{train,val}`.

| Architecture | Backend | CLI |
|---|---|---|
| ResNet18 | torchvision | `--architecture resnet18` |
| VGG16 | torchvision | `--architecture vgg16` |
| YOLO26x-cls | Ultralytics | `--architecture yolo26x-cls` |

```bash
uv run avesia-train-cls --architecture resnet18 --epochs 50 --imgsz 256
uv run avesia-train-cls --architecture vgg16 --epochs 50
uv run avesia-train-cls --architecture yolo26x-cls --epochs 50
# train all and write comparison.json (best by macro F1)
uv run avesia-train-cls --architecture all --epochs 50
```

Shared protocol: seed 42, input 256, Adam + cosine schedule (torchvision), ImageNet normalization, metrics = accuracy + macro precision/recall/F1. Checkpoints land in `artifacts/runs/train_cls/<arch>/best.pt` with `class_to_idx.json` and confusion matrix.

## Inference (detector → classifier)

```bash
uv run avesia-predict --source path/to/image.jpg --architecture resnet18
uv run avesia-predict --source path/to/dir --compare
```

Each detection is cropped with the same padding/resize as training, then classified. JSON fields include `xyxy`, `det_conf`, `species`, `cls_conf`, `classifier`, and `top_k`. `--compare` runs all available classifiers without ensembling.
