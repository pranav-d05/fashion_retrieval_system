"""Embedding sub-package for the Fashion Retrieval System."""

from importlib import import_module
from typing import Any

__all__ = ["FashionCLIPEmbedder", "TextEmbedder"]

_MODULES = {
    "FashionCLIPEmbedder": "src.embeddings.fashionclip_embedder",
    "TextEmbedder": "src.embeddings.text_embedder",
}


def __getattr__(name: str) -> Any:
    if module_name := _MODULES.get(name):
        return getattr(import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
