"""Detection classification metrics and confusion-matrix heatmap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    class_names: list[str]


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> ClassificationMetrics:
    if not y_true:
        empty = np.zeros((len(class_names), len(class_names)), dtype=int)
        return ClassificationMetrics(0.0, 0.0, 0.0, 0.0, empty, class_names)

    labels = list(range(len(class_names)))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        recall=float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
        class_names=class_names,
    )


def save_confusion_heatmap(
    metrics: ClassificationMetrics,
    out_path: Path,
    title: str = "Confusion matrix",
    max_labels: int = 40,
) -> Path:
    """Save a seaborn heatmap; truncates tick labels if there are many classes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics.confusion
    names = metrics.class_names
    fig_w = max(8, min(0.35 * len(names), 28))
    fig_h = max(6, min(0.35 * len(names), 24))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(cm, ax=ax, cmap="Blues", square=False, cbar=True)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if len(names) <= max_labels:
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticklabels(names, rotation=0, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def match_detections_to_gt(
    gt_boxes: list[tuple[int, float, float, float, float]],
    pred_boxes: list[tuple[int, float, float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Greedy IoU match. Returns (y_true_class, y_pred_class) for matched pairs.

    gt_boxes: (class_id, x1, y1, x2, y2)
    pred_boxes: (class_id, conf, x1, y1, x2, y2) — already sorted by conf desc preferred.
    Unmatched GT counted as false negative via background class is NOT added; callers
    that need FN/FP rates should use detection metrics. For species confusion we only
    score matched pairs; unmatched GT are appended with pred = -1 skipped, and
    unmatched preds ignored for confusion (still reflected in recall via match rate).
    """
    y_true: list[int] = []
    y_pred: list[int] = []
    used_pred: set[int] = set()

    def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    preds = sorted(pred_boxes, key=lambda p: p[1], reverse=True)
    for gt_cls, *gt_xyxy in gt_boxes:
        best_j = -1
        best_iou = iou_threshold
        gt_box = (gt_xyxy[0], gt_xyxy[1], gt_xyxy[2], gt_xyxy[3])
        for j, (p_cls, _conf, *p_xyxy) in enumerate(preds):
            if j in used_pred:
                continue
            score = iou(gt_box, (p_xyxy[0], p_xyxy[1], p_xyxy[2], p_xyxy[3]))
            if score >= best_iou:
                best_iou = score
                best_j = j
        if best_j >= 0:
            used_pred.add(best_j)
            y_true.append(gt_cls)
            y_pred.append(preds[best_j][0])
    return y_true, y_pred
