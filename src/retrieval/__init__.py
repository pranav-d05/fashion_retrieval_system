"""Retrieval sub-package for the Fashion Retrieval System."""

from importlib import import_module
from typing import Any

__all__ = ["QueryParser", "Retriever", "Reranker"]

_MODULES = {
    "QueryParser": "src.retrieval.query_parser",
    "Retriever": "src.retrieval.retriever",
    "Reranker": "src.retrieval.reranker",
}


def __getattr__(name: str) -> Any:
    if module_name := _MODULES.get(name):
        return getattr(import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
