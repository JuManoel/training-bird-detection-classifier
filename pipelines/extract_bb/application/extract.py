"""Build bird-detection + 256×256 classification crop datasets from YOLO boxes."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from pipelines.extract_bb.domain import ExtractConfig
from pipelines.extract_bb.infrastructure import YoloBirdDetector
from pipelines.shared.catalog import load_or_build_catalog
from pipelines.shared.crop import crop_bird_to_square
from pipelines.shared.csv_manifest import ManifestEntry, read_manifest, write_manifest
from pipelines.shared.dedupe import (
    cap_items_per_species,
    dedupe_by_content,
)
from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.split import stratified_group_split
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
    detect_label: Path
    crop_path: Path
    box_w: float
    box_h: float
    box_index: int
    source_key: str

    @property
    def crop_id(self) -> str:
        return f"{self.entry.catalog_id}:b{self.box_index}"


def _write_resized(src: Path, dst: Path, imgsz: int, quality: int) -> None:
    """Write a full-frame image scaled so max(side) <= imgsz (no upscale)."""
    from PIL import Image

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


def _source_key(entry: ManifestEntry) -> str:
    """Stable group id for all crops from one downloaded photo."""
    if entry.observation_id:
        return f"{entry.fuente}:{entry.observation_id}:{entry.catalog_id}"
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


def _reset_split_dirs(root: Path, splits: tuple[str, ...] = ("train", "val")) -> None:
    for split in splits:
        path = root / split
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _write_duplicate_report(path: Path, drops) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dropped_crop_id",
                "kept_crop_id",
                "scientific_name",
                "reason",
                "distance",
                "dropped_path",
                "kept_path",
            ],
        )
        writer.writeheader()
        for drop in drops:
            writer.writerow(
                {
                    "dropped_crop_id": drop.dropped.crop_id,
                    "kept_crop_id": drop.kept.crop_id,
                    "scientific_name": drop.dropped.entry.scientific_name,
                    "reason": drop.reason,
                    "distance": drop.distance if drop.distance is not None else "",
                    "dropped_path": str(drop.dropped.crop_path),
                    "kept_path": str(drop.kept.crop_path),
                }
            )


def run_extract(config: ExtractConfig) -> dict:
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

    # Exact + perceptual dedupe on full-frame downloads before YOLO.
    frame_dedupe = dedupe_by_content(
        entries,
        path_of=lambda e: Path(e.image_path),
        max_hamming=config.perceptual_max_hamming,
    )
    entries = frame_dedupe.kept
    logger.info(
        "Frame dedupe: kept=%d exact=%d perceptual=%d",
        len(entries),
        frame_dedupe.exact_duplicates,
        frame_dedupe.perceptual_duplicates,
    )

    detect_images = config.detect_root / "images"
    detect_labels = config.detect_root / "labels"
    rejected_root = config.detect_root / "rejected"
    _reset_split_dirs(detect_images)
    _reset_split_dirs(detect_labels)
    _reset_split_dirs(config.classify_root)
    if rejected_root.exists():
        shutil.rmtree(rejected_root)
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
    multi_bird_images = 0
    total_boxes = 0

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
                multi_bird_images += 1
            total_boxes += len(boxes)

            stem = entry.catalog_id.replace("/", "_").replace(":", "_")
            detect_img = stage_detect / "images" / f"{stem}.jpg"
            detect_label = stage_detect / "labels" / f"{stem}.txt"
            _write_resized(src, detect_img, config.imgsz, config.jpeg_quality)
            yolo_boxes = [
                xyxy_to_yolo(b.x1, b.y1, b.x2, b.y2, w, h, class_id=0) for b in boxes
            ]
            write_label_file(detect_label, yolo_boxes)

            species_dir = species_folder_name(entry.scientific_name)
            source_key = _source_key(entry)
            # Sort by score so box indices are stable/reproducible.
            ordered = sorted(
                enumerate(boxes), key=lambda ib: (-ib[1].score, ib[0])
            )
            for box_index, (_orig_i, box) in enumerate(ordered):
                crop_name = f"{stem}_b{box_index}.jpg"
                crop_path = stage_crops / species_dir / crop_name
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
                        detect_label=detect_label,
                        crop_path=crop.path,
                        box_w=crop.box_w,
                        box_h=crop.box_h,
                        box_index=box_index,
                        source_key=source_key,
                    )
                )

    if not staged:
        raise RuntimeError("No images produced bird boxes; cannot build dataset")

    # Crop-level exact + perceptual dedupe. Crops from the same source photo
    # are never collapsed against each other (distinct birds in one frame).
    crop_dedupe = dedupe_by_content(
        staged,
        path_of=lambda s: s.crop_path,
        group_key=lambda s: s.source_key,
        max_hamming=config.perceptual_max_hamming,
    )
    staged = crop_dedupe.kept
    _write_duplicate_report(
        config.classify_root / "crop_duplicates.csv", crop_dedupe.dropped
    )

    staged = cap_items_per_species(
        staged,
        species_of=lambda s: s.entry.scientific_name,
        fuente_of=lambda s: s.entry.fuente or "unknown",
        max_per_species=config.max_per_species,
        seed=config.seed,
    )

    by_species: dict[str, list[_StagedSample]] = defaultdict(list)
    for sample in staged:
        by_species[sample.entry.scientific_name].append(sample)

    kept: list[_StagedSample] = []
    dropped_species = 0
    species_stats: list[dict] = []
    for species, samples in sorted(by_species.items()):
        total = len(samples)
        if total < config.min_images:
            dropped_species += 1
            species_stats.append(
                {
                    "scientific_name": species,
                    "total": total,
                    "included": False,
                    "reason": f"below_min_{config.min_images}",
                }
            )
            for s in samples:
                rejected_rows.append(
                    {
                        "catalog_id": s.entry.catalog_id,
                        "scientific_name": species,
                        "reason": f"below_min_{config.min_images}",
                        "image_path": s.entry.image_path,
                    }
                )
            continue
        if total < config.target_images:
            reason = f"below_target_{config.target_images}"
        elif total >= config.max_per_species:
            reason = "at_cap"
        else:
            reason = "ok"
        species_stats.append(
            {
                "scientific_name": species,
                "total": total,
                "included": True,
                "reason": reason,
            }
        )
        kept.extend(samples)

    if not kept:
        raise RuntimeError(
            f"No species reached min_images={config.min_images} after cropping"
        )

    items = kept
    labels = [s.entry.scientific_name for s in items]
    groups = [s.source_key for s in items]
    split = stratified_group_split(
        items, labels, groups, train_ratio=config.train_ratio, seed=config.seed
    )

    def place_detect(split_name: str, samples: list[_StagedSample]) -> None:
        seen_stems: set[str] = set()
        for sample in samples:
            stem = sample.detect_img.stem
            if stem in seen_stems:
                continue
            seen_stems.add(stem)
            dst_img = detect_images / split_name / sample.detect_img.name
            dst_lbl = detect_labels / split_name / f"{stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.detect_img, dst_img)
            shutil.copy2(sample.detect_label, dst_lbl)

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

    # Final gate: every included species must satisfy 125 <= n <= 1000.
    final_counts: Counter[str] = Counter()
    for sample in split.train + split.val:
        final_counts[sample.entry.scientific_name] += 1
    violations = {
        name: n
        for name, n in final_counts.items()
        if n < config.min_images or n > config.max_per_species
    }
    if violations:
        raise RuntimeError(
            "Post-split crop counts violate thresholds "
            f"(min={config.min_images}, max={config.max_per_species}): {violations}"
        )

    class_names_detect = ["bird"]
    write_classes_txt(config.detect_root / "classes.txt", class_names_detect)
    write_data_yaml(config.detect_root / "data.yaml", config.detect_root, class_names_detect)

    present = sorted(final_counts.keys())
    write_classes_txt(config.classify_root / "classes.txt", present)

    crop_entries: list[ManifestEntry] = []
    train_crop_names = {s.crop_path.name for s in split.train}
    for sample in kept:
        split_name = "train" if sample.crop_path.name in train_crop_names else "val"
        species = species_folder_name(sample.entry.scientific_name)
        final_crop = config.classify_root / split_name / species / sample.crop_path.name
        crop_entries.append(
            ManifestEntry(
                catalog_id=f"{sample.entry.catalog_id}_b{sample.box_index}",
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
        "multi_bird_images": multi_bird_images,
        "total_boxes": total_boxes,
        "classes": len(present),
        "dropped_species": dropped_species,
        "crop_size": config.crop_size,
        "min_images": config.min_images,
        "target_images": config.target_images,
        "max_per_species": config.max_per_species,
        "frame_exact_duplicates": frame_dedupe.exact_duplicates,
        "frame_perceptual_duplicates": frame_dedupe.perceptual_duplicates,
        "crop_exact_duplicates": crop_dedupe.exact_duplicates,
        "crop_perceptual_duplicates": crop_dedupe.perceptual_duplicates,
        "species": species_stats,
        "per_species_final": dict(final_counts),
    }
    (config.classify_root / "extract_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Extract done: labeled=%d train=%d val=%d classes=%d dropped_species=%d "
        "boxes=%d multi_bird_images=%d",
        stats["labeled"],
        stats["train"],
        stats["val"],
        stats["classes"],
        stats["dropped_species"],
        stats["total_boxes"],
        stats["multi_bird_images"],
    )
    return stats
