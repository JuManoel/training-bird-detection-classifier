"""Build YOLO dataset from dfine bird boxes + species labels."""

from __future__ import annotations

import shutil
from pathlib import Path

from tqdm import tqdm

from pipelines.extract_bb.domain import ExtractConfig
from pipelines.extract_bb.infrastructure import DfineBirdDetector
from pipelines.shared.csv_manifest import read_manifest
from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.species import load_species_list
from pipelines.shared.split import stratified_split
from pipelines.shared.yolo_io import (
    write_classes_txt,
    write_data_yaml,
    write_label_file,
    xyxy_to_yolo,
)

logger = setup_logging(name="avesia.extract")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def run_extract(config: ExtractConfig) -> dict[str, int]:
    species = load_species_list(config.species_path)
    allowed = {s.scientific_name for s in species}
    # Preserve spicies.txt order for stable class ids among present species.
    ordered_allowed = [s.scientific_name for s in species]

    entries = [
        e
        for e in read_manifest(config.manifest_path)
        if e.status in {"ok", "skipped"} and e.scientific_name in allowed
    ]
    logger.info("Manifest entries usable: %d", len(entries))

    images_root = config.dataset_root / "images"
    labels_root = config.dataset_root / "labels"
    rejected_root = config.dataset_root / "rejected"
    for split in ("train", "val"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)
    rejected_root.mkdir(parents=True, exist_ok=True)

    # First pass: detect boxes; assign provisional species labels (scientific name).
    staged: list[tuple[ManifestEntry, Path, list]] = []
    rejected = 0

    with DfineBirdDetector(
        engine_path=config.engine_path,
        threshold=config.threshold,
        bird_class_names=config.bird_class_names,
    ) as detector:
        for entry in tqdm(entries, desc="extract_bb"):
            src = Path(entry.image_path)
            if not src.exists():
                logger.warning("Missing image: %s", src)
                rejected += 1
                continue
            boxes, (w, h) = detector.detect(src)
            if not boxes:
                _link_or_copy(src, rejected_root / src.name)
                rejected += 1
                continue
            if not config.keep_all:
                boxes = [max(boxes, key=lambda b: b.score)]

            stem = entry.catalog_id
            stage_img = (
                config.dataset_root
                / "_stage"
                / "images"
                / f"{stem}{src.suffix.lower() or '.jpg'}"
            )
            _link_or_copy(src, stage_img)
            staged.append((entry, stage_img, [(b, w, h) for b in boxes]))

    if not staged:
        raise RuntimeError("No images produced bird boxes; cannot build dataset")

    present = {e.scientific_name for e, _, _ in staged}
    class_names = [name for name in ordered_allowed if name in present]
    class_map = {name: i for i, name in enumerate(class_names)}

    labeled_paths: list[Path] = []
    labeled_species: list[str] = []
    for entry, stage_img, box_dims in staged:
        class_id = class_map[entry.scientific_name]
        yolo_boxes = [
            xyxy_to_yolo(b.x1, b.y1, b.x2, b.y2, w, h, class_id) for b, w, h in box_dims
        ]
        stage_lbl = config.dataset_root / "_stage" / "labels" / f"{stage_img.stem}.txt"
        write_label_file(stage_lbl, yolo_boxes)
        labeled_paths.append(stage_img)
        labeled_species.append(entry.scientific_name)

    split = stratified_split(
        labeled_paths,
        labeled_species,
        train_ratio=config.train_ratio,
        seed=config.seed,
    )

    def place(split_name: str, paths: list[Path]) -> None:
        for img_path in paths:
            stem = img_path.stem
            lbl = config.dataset_root / "_stage" / "labels" / f"{stem}.txt"
            _link_or_copy(img_path, images_root / split_name / img_path.name)
            shutil.copy2(lbl, labels_root / split_name / f"{stem}.txt")

    place("train", split.train)
    place("val", split.val)

    stage = config.dataset_root / "_stage"
    if stage.exists():
        shutil.rmtree(stage)

    write_classes_txt(config.dataset_root / "classes.txt", class_names)
    write_data_yaml(config.dataset_root / "data.yaml", config.dataset_root, class_names)

    stats = {
        "labeled": len(labeled_paths),
        "train": len(split.train),
        "val": len(split.val),
        "rejected": rejected,
        "classes": len(class_names),
    }
    logger.info("Extract done: %s", stats)
    return stats
