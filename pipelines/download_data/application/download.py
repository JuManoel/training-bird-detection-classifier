"""Application use-case: download filtered media from one or more CSVs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from pipelines.download_data.domain import DownloadConfig, DownloadResult
from pipelines.download_data.infrastructure import MacaulayDownloader
from pipelines.shared.csv_manifest import (
    ManifestEntry,
    MediaRecord,
    load_media_csv,
    merge_media_records,
    write_manifest,
)
from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.species import load_species_list

logger = setup_logging(name="avesia.download")


def _dest_path(output_dir, scientific_name: str, catalog_id: str):
    folder = scientific_name.replace(" ", "_")
    return output_dir / folder / f"{catalog_id}.jpg"


def _download_one(
    downloader: MacaulayDownloader,
    record: MediaRecord,
    dest,
    skip_existing: bool,
) -> DownloadResult:
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return DownloadResult(record=record, image_path=dest, status="skipped")
    try:
        path = downloader.download(
            record.catalog_id,
            dest,
            url=record.url,
            fuente=record.fuente,
        )
        return DownloadResult(record=record, image_path=path, status="ok")
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(record=record, image_path=None, status="error", error=str(exc))


def run_download(config: DownloadConfig) -> list[ManifestEntry]:
    species = load_species_list(config.species_path)
    allowed = {s.scientific_name for s in species}

    media_chunks: list[MediaRecord] = []
    for csv_path in config.csv_paths:
        chunk = load_media_csv(csv_path, allowed_scientific=allowed)
        logger.info("Loaded %d media rows from %s", len(chunk), csv_path)
        media_chunks.extend(chunk)
    media = merge_media_records(media_chunks)
    logger.info(
        "Found %d unique media rows matching %d target species (%d csv files)",
        len(media),
        len(allowed),
        len(config.csv_paths),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    downloader = MacaulayDownloader(
        timeout_s=config.timeout_s,
        max_retries=config.max_retries,
    )

    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = [
            pool.submit(
                _download_one,
                downloader,
                m,
                _dest_path(config.output_dir, m.scientific_name, m.catalog_id),
                config.skip_existing,
            )
            for m in media
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="download"):
            results.append(fut.result())

    entries: list[ManifestEntry] = []
    ok = skipped = errors = 0
    for r in results:
        if r.status == "ok":
            ok += 1
        elif r.status == "skipped":
            skipped += 1
        else:
            errors += 1
            logger.warning("Download failed %s: %s", r.record.catalog_id, r.error)
        if r.image_path is not None:
            entries.append(
                ManifestEntry(
                    catalog_id=r.record.catalog_id,
                    scientific_name=r.record.scientific_name,
                    common_name=r.record.common_name,
                    image_path=str(r.image_path.resolve()),
                    status=r.status,
                )
            )

    write_manifest(config.manifest_path, entries)
    logger.info(
        "Download done: ok=%d skipped=%d errors=%d manifest=%s",
        ok,
        skipped,
        errors,
        config.manifest_path,
    )
    return entries
