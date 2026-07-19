"""Per-species coverage report for multi-source media."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from pipelines.shared.csv_manifest import ManifestEntry, MediaRecord
from pipelines.shared.taxonomy import normalize_scientific_name


@dataclass
class SpeciesCoverage:
    scientific_name: str
    total: int
    by_source: dict[str, int]
    included: bool
    reason: str


def coverage_from_records(
    records: list[MediaRecord] | list[ManifestEntry],
    *,
    min_images: int = 125,
    target_images: int = 500,
) -> list[SpeciesCoverage]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        name = normalize_scientific_name(rec.scientific_name)
        fuente = getattr(rec, "fuente", None) or "unknown"
        counts[name][fuente] += 1

    rows: list[SpeciesCoverage] = []
    for name in sorted(counts):
        by_source = dict(counts[name])
        total = sum(by_source.values())
        if total < min_images:
            included = False
            reason = f"below_min_{min_images}"
        elif total < target_images:
            included = True
            reason = f"below_target_{target_images}"
        else:
            included = True
            reason = "ok"
        rows.append(
            SpeciesCoverage(
                scientific_name=name,
                total=total,
                by_source=by_source,
                included=included,
                reason=reason,
            )
        )
    return rows


def write_coverage_report(
    rows: list[SpeciesCoverage],
    out_json: Path,
    out_csv: Path | None = None,
) -> Path:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_species": len(rows),
        "included": sum(1 for r in rows if r.included),
        "excluded": sum(1 for r in rows if not r.included),
        "species": [asdict(r) for r in rows],
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["scientific_name", "total", "by_source", "included", "reason"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "scientific_name": row.scientific_name,
                        "total": row.total,
                        "by_source": json.dumps(row.by_source, ensure_ascii=False),
                        "included": row.included,
                        "reason": row.reason,
                    }
                )
    return out_json


def filter_included_names(rows: list[SpeciesCoverage]) -> set[str]:
    return {r.scientific_name for r in rows if r.included}
