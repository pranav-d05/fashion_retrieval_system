"""General-purpose utilities for the Fashion Retrieval System."""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterator

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def generate_image_id(image_path: Path) -> str:
    """Return a stable ID for an image based on its resolved path."""
    digest = hashlib.sha256(str(image_path.resolve()).encode()).hexdigest()[:16]
    return f"IMG_{digest}"


def iter_images(directory: Path, extensions: list[str] | None = None) -> Iterator[Path]:
    """Recursively yield image paths from a directory, sorted for reproducibility."""
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    allowed = (
        frozenset(extension.lower() for extension in extensions)
        if extensions
        else SUPPORTED_EXTENSIONS
    )

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def chunk_list(items: list, chunk_size: int) -> Generator[list, None, None]:
    """Yield successive chunks of size `chunk_size` from `items`."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


@contextmanager
def timer(operation: str) -> Generator[None, None, None]:
    """Log elapsed time for a code block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = round(time.perf_counter() - start, 3)
        logger.info(f"[{operation}] took {elapsed}s")
