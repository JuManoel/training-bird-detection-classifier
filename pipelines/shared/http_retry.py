"""HTTP GET with retries, including longer backoff on 429 rate limits."""

from __future__ import annotations

import time
from typing import Any

import httpx


def _retry_after_seconds(response: httpx.Response, attempt: int) -> float:
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return max(float(raw), 1.0)
        except ValueError:
            pass
    # Cap grows with attempt: 8, 16, 32, 64, 120...
    return min(2 ** (attempt + 2), 120)


def get_json(
    url: str,
    *,
    params: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
    max_retries: int,
    label: str,
) -> dict:
    """GET JSON with retries. Rate limits (429) use longer backoff."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(
                headers=headers,
                timeout=timeout_s,
                follow_redirects=True,
            ) as client:
                response = client.get(url, params=params)
                if response.status_code == 429:
                    wait = _retry_after_seconds(response, attempt)
                    last_error = httpx.HTTPStatusError(
                        f"429 Too Many Requests (attempt {attempt}/{max_retries})",
                        request=response.request,
                        response=response,
                    )
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response is not None and exc.response.status_code == 429:
                time.sleep(_retry_after_seconds(exc.response, attempt))
            else:
                time.sleep(min(2**attempt, 8))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"{label} request failed: {last_error}")
