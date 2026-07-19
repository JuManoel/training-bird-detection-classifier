"""License allow/deny helpers for multi-source media."""

from __future__ import annotations

# Research-friendly licenses. Empty / unknown is kept for legacy Macaulay CSVs
# that do not ship a license column (academic use assumed by project policy).
ALLOWED_LICENSE_TOKENS = {
    "cc0",
    "cc-0",
    "public domain",
    "pd",
    "cc-by",
    "cc by",
    "cc-by-sa",
    "cc by-sa",
    "cc-by-nc",
    "cc by-nc",
    "cc-by-nc-sa",
    "cc by-nc-sa",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by/4.0/",
    "http://creativecommons.org/licenses/by-sa/4.0/",
    "https://creativecommons.org/licenses/by-sa/4.0/",
    "http://creativecommons.org/licenses/by-nc/4.0/",
    "https://creativecommons.org/licenses/by-nc/4.0/",
    "http://creativecommons.org/licenses/by-nc-sa/4.0/",
    "https://creativecommons.org/licenses/by-nc-sa/4.0/",
}

DENIED_LICENSE_TOKENS = {
    "all rights reserved",
    "© all rights reserved",
    "no rights reserved",  # ambiguous; treat carefully but allow if cc0 elsewhere
    "copyright",
}


def normalize_license(license_raw: str | None) -> str:
    return (license_raw or "").strip().lower()


def is_license_allowed(license_raw: str | None, *, allow_unknown: bool = True) -> bool:
    """Return True when the media may be used for academic training."""
    lic = normalize_license(license_raw)
    if not lic:
        return allow_unknown
    if any(tok in lic for tok in DENIED_LICENSE_TOKENS if tok == "all rights reserved"):
        if "all rights reserved" in lic and "creativecommons" not in lic:
            return False
    for allowed in ALLOWED_LICENSE_TOKENS:
        if allowed in lic:
            return True
    # Common short codes
    if lic.startswith("cc0") or lic.startswith("cc-by"):
        return True
    return False
