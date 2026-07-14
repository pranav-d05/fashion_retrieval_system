"""
Image loader utility for the indexing pipeline.

Centralises image loading so that format normalisation
(RGB conversion, EXIF orientation) is applied consistently.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def load_image(path: Path) -> Image.Image:
    """Load a PIL Image from disk, applying EXIF orientation and converting to RGB.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        An RGB PIL ``Image`` object.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be opened as an image.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # Fix EXIF rotation
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
