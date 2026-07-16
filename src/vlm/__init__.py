"""VLM sub-package for the Fashion Retrieval System.

Imports stay lazy so using a small text-only utility does not eagerly load
vision-language dependencies.
"""

from importlib import import_module
from typing import Any

__all__ = ["CaptionGenerator", "MetadataExtractor"]


def __getattr__(name: str) -> Any:
    if name == "CaptionGenerator":
        return import_module("src.vlm.caption_generator").CaptionGenerator
    if name == "MetadataExtractor":
        return import_module("src.vlm.metadata_extractor").MetadataExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
