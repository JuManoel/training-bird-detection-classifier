"""HTTP client for Macaulay CDN and direct media URLs (e.g. iNaturalist)."""

from __future__ import annotations

import time
from pathlib import Path

import httpx

MACAULAY_ASSET_URL = "https://cdn.download.ams.birds.cornell.edu/api/v1/asset/{catalog_id}/"

DEFAULT_HEADERS = {
    "User-Agent": (
        "avesia-yolo/0.1 (+research; media download for academic use)"
    ),
    "Accept": "image/*,*/*",
}

MACAULAY_HEADERS = {
    **DEFAULT_HEADERS,
    "Referer": "https://macaulaylibrary.org/",
}


class MacaulayDownloader:
    """Download photo assets by ML catalog number or absolute URL."""

    def __init__(
        self,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.headers = headers or DEFAULT_HEADERS

    def asset_url(self, catalog_id: str) -> str:
        return MACAULAY_ASSET_URL.format(catalog_id=catalog_id)

    def download(
        self,
        catalog_id: str,
        dest: Path,
        *,
        url: str | None = None,
        fuente: str = "macaulay",
    ) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if url:
            fetch_url = url
            headers = self.headers
        else:
            fetch_url = self.asset_url(catalog_id)
            headers = MACAULAY_HEADERS if fuente == "macaulay" else self.headers

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(
                    headers=headers,
                    timeout=self.timeout_s,
                    follow_redirects=True,
                ) as client:
                    with client.stream("GET", fetch_url) as response:
                        response.raise_for_status()
                        tmp = dest.with_suffix(dest.suffix + ".part")
                        with tmp.open("wb") as fh:
                            for chunk in response.iter_bytes(chunk_size=1024 * 64):
                                fh.write(chunk)
                        tmp.replace(dest)
                return dest
            except Exception as exc:  # noqa: BLE001 — retry then surface
                last_error = exc
                time.sleep(min(2**attempt, 8))
        label = url or f"ML{catalog_id}"
        raise RuntimeError(
            f"Failed to download {label} after {self.max_retries} tries: {last_error}"
        )
