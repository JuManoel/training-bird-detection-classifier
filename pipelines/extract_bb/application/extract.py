"""Build bird-detection + 256×256 classification crop datasets from YOLO boxes."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pipelines.extract_bb.domain import ExtractConfig
from pipelines.extract_bb.infrastructure import YoloBirdDetector
from pipelines.shared.catalog import load_or_build_catalog
from pipelines.shared.crop import crop_bird_to_square
from pipelines.shared.csv_manifest import ManifestEntry, read_manifest, write_manifest
from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.split import stratified_split
from pipelines.shared.taxonomy import normalize_scientific_name, species_folder_name
from pipelines.shared.yolo_io import (
    write_classes_txt,
    write_data_yaml,
    write_label_file,
    xyxy_to_yolo,
)

logger = setup_logging(name="avesia.extract")


@dataclass
class _StagedSample:
    entry: ManifestEntry
    detect_img: Path
    crop_path: Path
    box_w: float
    box_h: float
    observation_key: str


def _write_resized(src: Path, dst: Path, imgsz: int, quality: int) -> None:
    """Write a full-frame image scaled so max(side) <= imgsz (no upscale)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > imgsz:
            scale = imgsz / float(longest)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        im.save(dst, format="JPEG", quality=quality, optimize=True)


def _observation_key(entry: ManifestEntry) -> str:
    if entry.observation_id:
        return f"{entry.fuente}:{entry.observation_id}"
    if entry.media_hash:
        return f"hash:{entry.media_hash}"
    return f"{entry.fuente}:{entry.catalog_id}"


def _load_included_species(coverage_json: Path, min_images: int) -> set[str] | None:
    if not coverage_json.exists():
        return None
    payload = json.loads(coverage_json.read_text(encoding="utf-8"))
    included = set()
    for row in payload.get("species", []):
        if row.get("included") or int(row.get("total", 0)) >= min_images:
            included.add(normalize_scientific_name(row["scientific_name"]))
    return included


def run_extract(config: ExtractConfig) -> dict[str, int]:
    catalog = load_or_build_catalog(config.catalog_path, config.species_path)
    resolve = catalog.resolve

    included = _load_included_species(config.coverage_json, config.min_images)

    entries: list[ManifestEntry] = []
    for e in read_manifest(config.manifest_path):
        if e.status not in {"ok", "skipped"}:
            continue
        canon = resolve(e.scientific_name)
        if canon is None:
            continue
        if included is not None and canon not in included:
            continue
        e.scientific_name = canon
        entries.append(e)
    logger.info("Manifest entries usable: %d", len(entries))

    detect_images = config.detect_root / "images"
    detect_labels = config.detect_root / "labels"
    rejected_root = config.detect_root / "rejected"
    for split in ("train", "val"):
        (detect_images / split).mkdir(parents=True, exist_ok=True)
        (detect_labels / split).mkdir(parents=True, exist_ok=True)
        (config.classify_root / split).mkdir(parents=True, exist_ok=True)
    rejected_root.mkdir(parents=True, exist_ok=True)

    stage_detect = config.detect_root / "_stage"
    stage_crops = config.classify_root / "_stage"
    if stage_detect.exists():
        shutil.rmtree(stage_detect)
    if stage_crops.exists():
        shutil.rmtree(stage_crops)

    staged: list[_StagedSample] = []
    rejected_rows: list[dict[str, str]] = []
    rejected = 0
    multi_bird = 0

    with YoloBirdDetector(
        model=config.model,
        threshold=config.threshold,
        imgsz=config.imgsz,
        device=config.device,
        bird_class_names=config.bird_class_names,
    ) as detector:
        for entry in tqdm(entries, desc="extract_bb"):
            src = Path(entry.image_path)
            if not src.exists():
                logger.warning("Missing image: %s", src)
                rejected += 1
                rejected_rows.append(
                    {
                        "catalog_id": entry.catalog_id,
                        "scientific_name": entry.scientific_name,
                        "reason": "missing_image",
                        "image_path": entry.image_path,
                    }
                )
                continue
            boxes, (w, h) = detector.detect(src)
            if not boxes:
                link = rejected_root / src.name
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists():
                    try:
                        link.symlink_to(src.resolve())
                    except OSError:
                        shutil.copy2(src, link)
                rejected += 1
                rejected_rows.append(
                    {
                        "catalog_id": entry.catalog_id,
                        "scientific_name": entry.scientific_name,
                        "reason": "no_bird",
                        "image_path": entry.image_path,
                    }
                )
                continue
            if len(boxes) > 1:
                multi_bird += 1
            box = max(boxes, key=lambda b: b.score)

            stem = entry.catalog_id.replace("/", "_").replace(":", "_")
            detect_img = stage_detect / "images" / f"{stem}.jpg"
            _write_resized(src, detect_img, config.imgsz, config.jpeg_quality)
            # Labels written after we know final class (always bird=0)
            yolo_box = xyxy_to_yolo(box.x1, box.y1, box.x2, box.y2, w, h, class_id=0)
            write_label_file(stage_detect / "labels" / f"{stem}.txt", [yolo_box])

            species_dir = species_folder_name(entry.scientific_name)
            crop_path = stage_crops / species_dir / f"{stem}.jpg"
            crop = crop_bird_to_square(
                src,
                crop_path,
                box.x1,
                box.y1,
                box.x2,
                box.y2,
                size=config.crop_size,
                pad_ratio=config.pad_ratio,
                jpeg_quality=config.jpeg_quality,
            )
            staged.append(
                _StagedSample(
                    entry=entry,
                    detect_img=detect_img,
                    crop_path=crop.path,
                    box_w=crop.box_w,
                    box_h=crop.box_h,
                    observation_key=_observation_key(entry),
                )
            )

    if not staged:
        raise RuntimeError("No images produced bird boxes; cannot build dataset")

    # Count crops per species; drop below min_images
    by_species: dict[str, list[_StagedSample]] = defaultdict(list)
    for sample in staged:
        by_species[sample.entry.scientific_name].append(sample)

    kept: list[_StagedSample] = []
    dropped_species = 0
    for species, samples in by_species.items():
        # Observation-level collapse before counting
        by_obs: dict[str, _StagedSample] = {}
        for s in samples:
            by_obs.setdefault(s.observation_key, s)
        unique = list(by_obs.values())
        if len(unique) < config.min_images:
            dropped_species += 1
            for s in unique:
                rejected_rows.append(
                    {
                        "catalog_id": s.entry.catalog_id,
                        "scientific_name": species,
                        "reason": f"below_min_{config.min_images}",
                        "image_path": s.entry.image_path,
                    }
                )
            continue
        kept.extend(unique)

    if not kept:
        raise RuntimeError(
            f"No species reached min_images={config.min_images} after cropping"
        )

    # Grouped stratified split: split by observation_key within species
    items = kept
    labels = [s.entry.scientific_name for s in items]
    # Use observation keys as split units via proxy list
    split = stratified_split(items, labels, train_ratio=config.train_ratio, seed=config.seed)

    def place_detect(split_name: str, samples: list[_StagedSample]) -> None:
        for sample in samples:
            stem = sample.detect_img.stem
            dst_img = detect_images / split_name / sample.detect_img.name
            dst_lbl = detect_labels / split_name / f"{stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.detect_img, dst_img)
            shutil.copy2(stage_detect / "labels" / f"{stem}.txt", dst_lbl)

    def place_classify(split_name: str, samples: list[_StagedSample]) -> None:
        for sample in samples:
            species = species_folder_name(sample.entry.scientific_name)
            dst = config.classify_root / split_name / species / sample.crop_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.crop_path, dst)

    place_detect("train", split.train)
    place_detect("val", split.val)
    place_classify("train", split.train)
    place_classify("val", split.val)

    class_names_detect = ["bird"]
    write_classes_txt(config.detect_root / "classes.txt", class_names_detect)
    write_data_yaml(config.detect_root / "data.yaml", config.detect_root, class_names_detect)

    present = sorted({s.entry.scientific_name for s in kept})
    write_classes_txt(config.classify_root / "classes.txt", present)

    # Crops manifest
    crop_entries: list[ManifestEntry] = []
    train_stems = {s.detect_img.stem for s in split.train}
    for sample in kept:
        split_name = "train" if sample.detect_img.stem in train_stems else "val"
        species = species_folder_name(sample.entry.scientific_name)
        final_crop = config.classify_root / split_name / species / sample.crop_path.name
        crop_entries.append(
            ManifestEntry(
                catalog_id=sample.entry.catalog_id,
                scientific_name=sample.entry.scientific_name,
                common_name=sample.entry.common_name,
                image_path=sample.entry.image_path,
                status=split_name,
                fuente=sample.entry.fuente,
                observation_id=sample.entry.observation_id,
                license=sample.entry.license,
                author=sample.entry.author,
                taxon_id=sample.entry.taxon_id,
                latitude=sample.entry.latitude,
                longitude=sample.entry.longitude,
                event_date=sample.entry.event_date,
                media_hash=sample.entry.media_hash,
                url=sample.entry.url,
                crop_path=str(final_crop.resolve()),
                box_w=f"{sample.box_w:.2f}",
                box_h=f"{sample.box_h:.2f}",
            )
        )
    write_manifest(config.classify_root / "crops_manifest.csv", crop_entries)

    reject_path = config.detect_root / "rejected_manifest.csv"
    with reject_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["catalog_id", "scientific_name", "reason", "image_path"]
        )
        writer.writeheader()
        writer.writerows(rejected_rows)

    if stage_detect.exists():
        shutil.rmtree(stage_detect)
    if stage_crops.exists():
        shutil.rmtree(stage_crops)

    stats = {
        "labeled": len(kept),
        "train": len(split.train),
        "val": len(split.val),
        "rejected": rejected,
        "multi_bird_kept_top": multi_bird,
        "classes": len(present),
        "dropped_species": dropped_species,
        "crop_size": config.crop_size,
    }
    (config.classify_root / "extract_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Extract done: %s", stats)
    return stats
