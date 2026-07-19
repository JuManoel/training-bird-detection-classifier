"""API checkpoint resume and HTTP retry helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from pipelines.download_data.application.download import (
    _load_api_checkpoint,
    _maybe_fetch_apis,
    _save_api_checkpoint,
)
from pipelines.download_data.domain import DownloadConfig
from pipelines.shared.csv_manifest import MediaRecord
from pipelines.shared.http_retry import get_json


def _rec(name: str, cid: str) -> MediaRecord:
    return MediaRecord(
        catalog_id=cid,
        scientific_name=name,
        common_name="",
        url=f"https://example.com/{cid}.jpg",
        fuente="gbif",
    )


def _config(tmp_path: Path, **kwargs) -> DownloadConfig:
    defaults = dict(
        csv_paths=(),
        species_path=tmp_path / "species.txt",
        catalog_path=tmp_path / "catalog.json",
        output_dir=tmp_path / "out",
        manifest_path=tmp_path / "manifest.csv",
        coverage_json=tmp_path / "coverage.json",
        coverage_csv=tmp_path / "coverage.csv",
        max_per_species=10,
        min_images=1,
        target_images=5,
        fetch_inat=True,
        fetch_gbif=True,
        api_checkpoint_path=tmp_path / "api_fetch_checkpoint.csv",
        fresh_api_fetch=False,
    )
    defaults.update(kwargs)
    return DownloadConfig(**defaults)


def test_api_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "api_fetch_checkpoint.csv"
    media = [_rec("Aves one", "gbif_1"), _rec("Aves two", "gbif_2")]
    completed = {"Aves one"}
    _save_api_checkpoint(path, media, completed)
    loaded_media, loaded_done = _load_api_checkpoint(path)
    assert len(loaded_media) == 2
    assert loaded_media[0].catalog_id == "gbif_1"
    assert loaded_media[0].url.endswith("gbif_1.jpg")
    assert loaded_done == {"Aves one"}


def test_maybe_fetch_apis_resumes_completed(tmp_path: Path):
    checkpoint = tmp_path / "api_fetch_checkpoint.csv"
    existing_api = [_rec("Done bird", "gbif_done")]
    _save_api_checkpoint(checkpoint, existing_api, {"Done bird"})

    config = _config(tmp_path, api_checkpoint_path=checkpoint)
    catalog = ["Done bird", "New bird"]

    inat = MagicMock()
    gbif = MagicMock()
    gbif.fetch_species_photos.return_value = [_rec("New bird", "gbif_new")]
    inat.fetch_species_photos.return_value = []

    with (
        patch(
            "pipelines.download_data.application.download.INaturalistClient",
            return_value=inat,
        ),
        patch(
            "pipelines.download_data.application.download.GbifClient",
            return_value=gbif,
        ),
    ):
        extra = _maybe_fetch_apis(config, catalog, existing=[])

    assert any(r.catalog_id == "gbif_done" for r in extra)
    assert any(r.catalog_id == "gbif_new" for r in extra)
    # Completed species must not be re-queried.
    for call in inat.fetch_species_photos.call_args_list:
        assert call.args[0] != "Done bird"
    for call in gbif.fetch_species_photos.call_args_list:
        assert call.args[0] != "Done bird"

    _, done = _load_api_checkpoint(checkpoint)
    assert "Done bird" in done
    assert "New bird" in done


def test_maybe_fetch_apis_continues_on_species_error(tmp_path: Path):
    config = _config(tmp_path)
    catalog = ["Fail bird", "Ok bird"]

    inat = MagicMock()
    gbif = MagicMock()

    def gbif_fetch(name, max_records=0):
        if name == "Fail bird":
            raise RuntimeError("GBIF request failed: 429 Too Many Requests")
        return [_rec(name, f"gbif_{name}")]

    gbif.fetch_species_photos.side_effect = gbif_fetch
    inat.fetch_species_photos.return_value = []

    with (
        patch(
            "pipelines.download_data.application.download.INaturalistClient",
            return_value=inat,
        ),
        patch(
            "pipelines.download_data.application.download.GbifClient",
            return_value=gbif,
        ),
    ):
        extra = _maybe_fetch_apis(config, catalog, existing=[])

    assert any(r.scientific_name == "Ok bird" for r in extra)
    _, done = _load_api_checkpoint(config.api_checkpoint_path)
    assert "Fail bird" not in done
    assert "Ok bird" in done


def test_get_json_retries_429(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "pipelines.shared.http_retry.time.sleep", lambda s: sleeps.append(s)
    )

    request = httpx.Request("GET", "https://example.com/x")
    too_many = httpx.Response(429, headers={"Retry-After": "2"}, request=request)
    ok = httpx.Response(200, json={"ok": True}, request=request)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None):
            if len(sleeps) == 0:
                return too_many
            return ok

    monkeypatch.setattr("pipelines.shared.http_retry.httpx.Client", FakeClient)
    payload = get_json(
        "https://example.com/x",
        params={},
        headers={},
        timeout_s=5.0,
        max_retries=3,
        label="test",
    )
    assert payload == {"ok": True}
    assert sleeps == [2.0]


def test_download_parser_api_checkpoint_flags():
    from pipelines.download_data.cli import build_parser

    p = build_parser()
    args = p.parse_args(["--fresh-api-fetch", "--retries", "5"])
    assert args.fresh_api_fetch is True
    assert args.retries == 5
    assert args.api_checkpoint.endswith("api_fetch_checkpoint.csv")
