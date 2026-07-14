"""Logging setup for the Fashion Retrieval System."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure the root logger. Call once at application startup.

    Args:
        level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.
        fmt:   "text" for coloured dev output, "json" for structured prod logs.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Suppress noisy third-party loggers
    for lib in ("transformers", "qdrant_client", "httpx", "PIL", "torch"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use __name__ as the name."""
    return logging.getLogger(name)
