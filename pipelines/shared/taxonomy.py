"""Scientific-name normalization to species rank."""

from __future__ import annotations

import re

_GROUP_RE = re.compile(r"\s*\[.*?\]\s*")
_PAREN_RE = re.compile(r"\s*\(.*?\)\s*")
_SPACES_RE = re.compile(r"\s+")


def normalize_scientific_name(name: str) -> str:
    """Collapse subspecies / groups / domestic types to binomial species rank.

    Examples
    --------
    >>> normalize_scientific_name("Ardea alba modesta")
    'Ardea alba'
    >>> normalize_scientific_name("Camptostoma obsoletum [obsoletum Group]")
    'Camptostoma obsoletum'
    >>> normalize_scientific_name("Anas platyrhynchos (Domestic type)")
    'Anas platyrhynchos'
    """
    text = (name or "").strip()
    if not text:
        return ""
    text = _GROUP_RE.sub(" ", text)
    text = _PAREN_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    # Drop hybrid markers and keep left parent when "A x B"
    if " x " in text.lower():
        text = re.split(r"\s+[xX]\s+", text, maxsplit=1)[0].strip()
    parts = text.split()
    if len(parts) >= 2:
        genus, epithet = parts[0], parts[1]
        # Skip "sp.", "spp.", "cf." as epithets
        if epithet.lower().rstrip(".") in {"sp", "spp", "cf", "aff"}:
            return genus
        return f"{genus} {epithet}"
    return text


def species_folder_name(scientific_name: str) -> str:
    """Filesystem-safe folder from a (possibly raw) scientific name."""
    return normalize_scientific_name(scientific_name).replace(" ", "_")
