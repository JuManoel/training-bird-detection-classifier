"""Consistent logging setup for all pipelines."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.INFO, name: str | None = None) -> logging.Logger:
    """Configure root logging once and return a named logger."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        root.addHandler(handler)
        root.setLevel(level)
    else:
        root.setLevel(level)
    return logging.getLogger(name or "avesia")
