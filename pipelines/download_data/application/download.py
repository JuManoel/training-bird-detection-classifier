"""Application use-case: download filtered media from CSVs and APIs."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from pipelines.download_data.domain import DownloadConfig, DownloadResult
from pipelines.download_data.infrastructure import MacaulayDownloader
from pipelines.download_data.infrastructure.gbif_client import GbifClient
from pipelines.download_data.infrastructure.icesi_import import load_icesi_csv
from pipelines.download_data.infrastructure.inaturalist_client import INaturalistClient
from pipelines.shared.catalog import load_or_build_catalog
from pipelines.shared.coverage import coverage_from_records, write_coverage_report
from pipelines.shared.csv_manifest import (
    ManifestEntry,
    MediaRecord,
    detect_csv_format,
    load_media_csv,
    write_manifest,
    write_media_csv,
)
from pipelines.shared.dedupe import (
    cap_per_species,
    dedupe_media_records,
    file_sha256,
)
from pipelines.shared.logging_utils import setup_logging
from pipelines.shared.taxonomy import species_folder_name

logger = setup_logging(name="avesia.download")


def _api_done_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + ".done.json")


def _load_api_checkpoint(path: Path) -> tuple[list[MediaRecord], set[str]]:
    media: list[MediaRecord] = []
    if path.exists():
        media = load_media_csv(path)
    done: set[str] = set()
    done_path = _api_done_path(path)
    if done_path.exists():
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        done = {str(name) for name in payload.get("completed", [])}
    return media, done


def _save_api_checkpoint(
    path: Path, media: list[MediaRecord], completed: set[str]
) -> None:
    write_media_csv(path, media)
    done_path = _api_done_path(path)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(
        json.dumps({"completed": sorted(completed)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _dest_path(output_dir: Path, scientific_name: str, catalog_id: str) -> Path:
    folder = species_folder_name(scientific_name)
    safe_id = catalog_id.replace("/", "_").replace(":", "_")
    return output_dir / folder / f"{safe_id}.jpg"


def _download_one(
    downloader: MacaulayDownloader,
    record: MediaRecord,
    dest: Path,
    skip_existing: bool,
) -> DownloadResult:
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        try:
            digest = file_sha256(dest)
        except OSError:
            digest = None
        return DownloadResult(
            record=record, image_path=dest, status="skipped", media_hash=digest
        )
    try:
        path = downloader.download(
            record.catalog_id,
            dest,
            url=record.url,
            fuente=record.fuente,
        )
        digest = file_sha256(path)
        return DownloadResult(
            record=record, image_path=path, status="ok", media_hash=digest
        )
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(
            record=record, image_path=None, status="error", error=str(exc)
        )


def _load_csv_records(config: DownloadConfig, resolve) -> list[MediaRecord]:
    media_chunks: list[MediaRecord] = []
    for csv_path in config.csv_paths:
        kind = detect_csv_format(csv_path)
        if kind == "generic" and "icesi" in csv_path.name.lower():
            chunk = load_icesi_csv(csv_path, resolve_allowed=resolve)
        else:
            chunk = load_media_csv(csv_path, resolve_allowed=resolve)
        logger.info("Loaded %d media rows from %s (%s)", len(chunk), csv_path, kind)
        media_chunks.extend(chunk)
    return media_chunks


def _split_api_quotas(need: int, fetch_inat: bool, fetch_gbif: bool) -> tuple[int, int]:
    """Split the remaining deficit across enabled APIs (both get work when both on)."""
    if need <= 0:
        return 0, 0
    if fetch_inat and fetch_gbif:
        inat_quota = (need + 1) // 2
        gbif_quota = need - inat_quota
        return inat_quota, gbif_quota
    if fetch_inat:
        return need, 0
    if fetch_gbif:
        return 0, need
    return 0, 0


def _maybe_fetch_apis(
    config: DownloadConfig,
    catalog_names: list[str],
    existing: list[MediaRecord],
) -> list[MediaRecord]:
    extra: list[MediaRecord] = []
    completed: set[str] = set()
    if not (config.fetch_inat or config.fetch_gbif):
        return extra

    checkpoint = config.api_checkpoint_path
    if checkpoint is not None and not config.fresh_api_fetch:
        extra, completed = _load_api_checkpoint(checkpoint)
        if extra or completed:
            logger.info(
                "Resuming API checkpoint: %d media rows, %d species done (%s)",
                len(extra),
                len(completed),
                checkpoint,
            )
    elif checkpoint is not None and config.fresh_api_fetch:
        logger.info("Fresh API fetch requested; ignoring checkpoint at %s", checkpoint)

    coverage = coverage_from_records(
        existing + extra,
        min_images=config.min_images,
        target_images=config.target_images,
    )
    by_name = {row.scientific_name: row.total for row in coverage}
    if config.fetch_only_below_target:
        targets = [
            name
            for name in catalog_names
            if by_name.get(name, 0) < config.target_images
        ]
    else:
        targets = list(catalog_names)

    pending = [name for name in targets if name not in completed]
    logger.info(
        "API fetch targets: %d species (%d pending, %d already done)",
        len(targets),
        len(pending),
        len(targets) - len(pending),
    )
    inat_client = (
        INaturalistClient(timeout_s=config.timeout_s, max_retries=config.max_retries)
        if config.fetch_inat
        else None
    )
    gbif_client = (
        GbifClient(
            timeout_s=config.timeout_s,
            max_retries=config.max_retries,
            country=config.gbif_country,
        )
        if config.fetch_gbif
        else None
    )

    failures = 0
    for name in tqdm(pending, desc="api_fetch"):
        need = max(0, config.max_per_species - by_name.get(name, 0))
        if need <= 0:
            completed.add(name)
            if checkpoint is not None:
                _save_api_checkpoint(checkpoint, extra, completed)
            continue

        inat_got = 0
        gbif_got = 0
        try:
            inat_quota, gbif_quota = _split_api_quotas(
                need, config.fetch_inat, config.fetch_gbif
            )

            if inat_client is not None and inat_quota > 0:
                fetched = inat_client.fetch_species_photos(name, max_records=inat_quota)
                extra.extend(fetched)
                inat_got = len(fetched)

            # GBIF gets its share plus any iNat shortfall.
            gbif_need = gbif_quota + max(0, inat_quota - inat_got)
            if gbif_client is not None and gbif_need > 0:
                fetched = gbif_client.fetch_species_photos(name, max_records=gbif_need)
                extra.extend(fetched)
                gbif_got = len(fetched)

            # If GBIF fell short and iNat still has budget room, top up iNat.
            remaining = need - inat_got - gbif_got
            if inat_client is not None and remaining > 0 and inat_got < need:
                fetched = inat_client.fetch_species_photos(name, max_records=remaining)
                extra.extend(fetched)
                inat_got += len(fetched)

            by_name[name] = by_name.get(name, 0) + inat_got + gbif_got
            completed.add(name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            by_name[name] = by_name.get(name, 0) + inat_got + gbif_got
            logger.warning(
                "API fetch failed for %s (will retry next run): %s", name, exc
            )
        finally:
            if checkpoint is not None:
                _save_api_checkpoint(checkpoint, extra, completed)

    if failures:
        logger.warning(
            "API fetch finished with %d species failures; re-run to resume them",
            failures,
        )
    elif checkpoint is not None:
        logger.info("API checkpoint saved: %s (%d media)", checkpoint, len(extra))

    return extra


def run_download(config: DownloadConfig) -> list[ManifestEntry]:
    catalog = load_or_build_catalog(config.catalog_path, config.species_path)
    resolve = catalog.resolve
    catalog_names = catalog.ordered_canonical()

    media = _load_csv_records(config, resolve)
    media = dedupe_media_records(media)
    media.extend(_maybe_fetch_apis(config, catalog_names, media))
    media = dedupe_media_records(media)
    media = cap_per_species(
        media, max_per_species=config.max_per_species, seed=config.seed
    )
    logger.info(
        "Prepared %d unique media rows for %d catalog species",
        len(media),
        len(catalog_names),
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
                    fuente=r.record.fuente,
                    observation_id=r.record.observation_id,
                    license=r.record.license,
                    author=r.record.author,
                    taxon_id=r.record.taxon_id,
                    latitude=r.record.latitude,
                    longitude=r.record.longitude,
                    event_date=r.record.event_date,
                    media_hash=r.media_hash or r.record.media_hash,
                    url=r.record.url,
                )
            )

    # Content/perceptual image dedupe is intentionally skipped in extract (too heavy).
    write_manifest(config.manifest_path, entries)
    cov = coverage_from_records(
        entries,
        min_images=config.min_images,
        target_images=config.target_images,
    )
    write_coverage_report(cov, config.coverage_json, config.coverage_csv)
    included = sum(1 for row in cov if row.included)
    logger.info(
        "Download done: ok=%d skipped=%d errors=%d kept=%d "
        "species_included=%d/%d manifest=%s",
        ok,
        skipped,
        errors,
        len(entries),
        included,
        len(cov),
        config.manifest_path,
    )
    return entries
