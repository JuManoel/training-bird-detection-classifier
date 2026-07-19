"""iNaturalist Observations API client (photos of Colombian catalog species)."""

from __future__ import annotations

import time
from typing import Iterable

from pipelines.shared.csv_manifest import MediaRecord
from pipelines.shared.http_retry import get_json
from pipelines.shared.licenses import is_license_allowed
from pipelines.shared.taxonomy import normalize_scientific_name

INAT_API = "https://api.inaturalist.org/v1"
DEFAULT_HEADERS = {
    "User-Agent": "avesia-yolo/0.1 (+research; academic bird classifier)",
    "Accept": "application/json",
}


class INaturalistClient:
    """Fetch research-grade photo observations for a set of species names."""

    def __init__(
        self,
        timeout_s: float = 60.0,
        max_retries: int = 8,
        sleep_s: float = 0.5,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.sleep_s = sleep_s

    def _get(self, path: str, params: dict) -> dict:
        return get_json(
            f"{INAT_API}{path}",
            params=params,
            headers=DEFAULT_HEADERS,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
            label="iNaturalist",
        )

    def fetch_species_photos(
        self,
        scientific_name: str,
        *,
        per_page: int = 200,
        max_records: int = 2000,
        quality_grade: str = "research",
    ) -> list[MediaRecord]:
        """Return photo MediaRecords for one species (any geography)."""
        records: list[MediaRecord] = []
        page = 1
        canon = normalize_scientific_name(scientific_name)
        while len(records) < max_records:
            payload = self._get(
                "/observations",
                {
                    "taxon_name": canon,
                    "photos": "true",
                    "quality_grade": quality_grade,
                    "per_page": min(per_page, max_records - len(records)),
                    "page": page,
                    "order_by": "votes",
                },
            )
            results = payload.get("results") or []
            if not results:
                break
            for obs in results:
                obs_id = str(obs.get("id", ""))
                taxon = obs.get("taxon") or {}
                name = normalize_scientific_name(
                    taxon.get("name") or obs.get("species_guess") or canon
                )
                common = ""
                if isinstance(taxon.get("preferred_common_name"), str):
                    common = taxon["preferred_common_name"]
                photos = obs.get("photos") or []
                for photo in photos:
                    license_code = photo.get("license_code") or obs.get("license_code")
                    if not is_license_allowed(license_code, allow_unknown=False):
                        continue
                    url = (
                        photo.get("url")
                        or photo.get("original_url")
                        or photo.get("large_url")
                    )
                    if url and "square" in url:
                        url = url.replace("square", "original")
                    if not url:
                        continue
                    photo_id = str(photo.get("id") or f"{obs_id}_{len(records)}")
                    records.append(
                        MediaRecord(
                            catalog_id=f"inat_{photo_id}",
                            scientific_name=name or canon,
                            common_name=common,
                            format="Photo",
                            url=url,
                            fuente="inaturalist",
                            observation_id=obs_id or None,
                            license=str(license_code) if license_code else None,
                            author=(photo.get("attribution") or None),
                            taxon_id=str(taxon.get("id") or "") or None,
                            latitude=str((obs.get("geojson") or {}).get("coordinates", [None, None])[1] or "")
                            or None,
                            longitude=str((obs.get("geojson") or {}).get("coordinates", [None, None])[0] or "")
                            or None,
                            event_date=(obs.get("observed_on") or obs.get("time_observed_at") or None),
                        )
                    )
                    if len(records) >= max_records:
                        break
                if len(records) >= max_records:
                    break
            page += 1
            time.sleep(self.sleep_s)
            if page > int(payload.get("total_pages") or page):
                break
        return records

    def fetch_many(
        self,
        scientific_names: Iterable[str],
        *,
        max_per_species: int = 2000,
    ) -> list[MediaRecord]:
        out: list[MediaRecord] = []
        for name in scientific_names:
            out.extend(self.fetch_species_photos(name, max_records=max_per_species))
            time.sleep(self.sleep_s)
        return out
