"""
Indexing sub-package for the Fashion Retrieval System.

Lazy import so a lightweight caller doesn't eagerly pull in the VLM/embedding
stack just to touch this package.
"""

from importlib import import_module
from typing import Any

__all__ = ["StagedIndexer"]

_MODULES = {
    "StagedIndexer": "src.indexing.staged_indexer",
}


def __getattr__(name: str) -> Any:
    if module_name := _MODULES.get(name):
        return getattr(import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
