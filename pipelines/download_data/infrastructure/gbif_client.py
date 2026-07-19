"""GBIF / SiB Colombia StillImage occurrence client."""

from __future__ import annotations

import time
from typing import Iterable

import httpx

from pipelines.shared.csv_manifest import MediaRecord
from pipelines.shared.licenses import is_license_allowed
from pipelines.shared.taxonomy import normalize_scientific_name

GBIF_API = "https://api.gbif.org/v1"
# SiB Colombia publishing organization (GBIF); used as soft filter when available.
SIB_COLOMBIA_COUNTRY = "CO"
DEFAULT_HEADERS = {
    "User-Agent": "avesia-yolo/0.1 (+research; academic bird classifier)",
    "Accept": "application/json",
}


class GbifClient:
    """Fetch StillImage media for species via the GBIF Occurrence API."""

    def __init__(
        self,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        sleep_s: float = 0.25,
        country: str | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.sleep_s = sleep_s
        # Default: no country filter (photos of Colombian-listed species worldwide).
        self.country = country

    def _get(self, path: str, params: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(
                    headers=DEFAULT_HEADERS,
                    timeout=self.timeout_s,
                    follow_redirects=True,
                ) as client:
                    response = client.get(f"{GBIF_API}{path}", params=params)
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"GBIF request failed: {last_error}")

    def match_species_key(self, scientific_name: str) -> str | None:
        payload = self._get("/species/match", {"name": scientific_name, "strict": "false"})
        if payload.get("matchType") in {"NONE", None} and not payload.get("usageKey"):
            return None
        key = payload.get("speciesKey") or payload.get("usageKey")
        return str(key) if key else None

    def fetch_species_photos(
        self,
        scientific_name: str,
        *,
        max_records: int = 2000,
        limit: int = 100,
    ) -> list[MediaRecord]:
        canon = normalize_scientific_name(scientific_name)
        taxon_key = self.match_species_key(canon)
        if not taxon_key:
            return []
        records: list[MediaRecord] = []
        offset = 0
        while len(records) < max_records:
            params: dict[str, object] = {
                "taxonKey": taxon_key,
                "mediaType": "StillImage",
                "limit": min(limit, max_records - len(records)),
                "offset": offset,
            }
            if self.country:
                params["country"] = self.country
            payload = self._get("/occurrence/search", params)
            results = payload.get("results") or []
            if not results:
                break
            for occ in results:
                media_list = occ.get("media") or []
                for media in media_list:
                    if (media.get("type") or "").lower() not in {"", "stillimage", "still_image"}:
                        # GBIF often omits type when mediaType filter already applied
                        if media.get("type") and "image" not in str(media.get("type")).lower():
                            continue
                    url = media.get("identifier") or media.get("references")
                    if not url:
                        continue
                    license_raw = media.get("license") or occ.get("license")
                    if license_raw and not is_license_allowed(license_raw):
                        continue
                    gbif_id = str(occ.get("key") or occ.get("gbifID") or "")
                    media_id = str(media.get("identifier") or url)
                    catalog_id = f"gbif_{gbif_id}_{abs(hash(media_id)) % 10_000_000}"
                    records.append(
                        MediaRecord(
                            catalog_id=catalog_id,
                            scientific_name=normalize_scientific_name(
                                occ.get("species") or occ.get("scientificName") or canon
                            )
                            or canon,
                            common_name="",
                            format="Photo",
                            url=url,
                            fuente="gbif",
                            observation_id=gbif_id or None,
                            license=str(license_raw) if license_raw else None,
                            author=media.get("rightsHolder") or occ.get("rightsHolder"),
                            taxon_id=taxon_key,
                            latitude=str(occ.get("decimalLatitude") or "") or None,
                            longitude=str(occ.get("decimalLongitude") or "") or None,
                            event_date=occ.get("eventDate"),
                        )
                    )
                    if len(records) >= max_records:
                        break
                if len(records) >= max_records:
                    break
            offset += len(results)
            time.sleep(self.sleep_s)
            if offset >= int(payload.get("count") or 0):
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
