"""
Embeddings sub-package for the Fashion Retrieval System.
"""

from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
from src.embeddings.text_embedder import TextEmbedder

__all__ = ["FashionCLIPEmbedder", "TextEmbedder"]
