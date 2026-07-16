"""
BGE text encoder (BAAI/bge-base-en-v1.5).

Wraps ``sentence-transformers`` to produce L2-normalised float32
embeddings of dimension 768 from natural language text.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.config_loader import TextEmbeddingConfig

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class TextEmbedder:
    """Semantic text embedder backed by BAAI/bge-base-en-v1.5.

    Args:
        config: ``TextEmbeddingConfig`` loaded from ``models.yaml``.
    """

    def __init__(self, config: TextEmbeddingConfig) -> None:
        device = _resolve_device(config.device)
        logger.info(
            "Loading TextEmbedder '%s' (device=%s)…",
            config.model_name,
            device,
        )
        self._model = SentenceTransformer(
            config.model_name,
            device=device,
        )
        self._normalize = config.normalize_embeddings
        logger.info("TextEmbedder loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts into BGE embeddings.

        Args:
            texts:      List of strings to encode.
            batch_size: Internal batch size forwarded to sentence-transformers.

        Returns:
            Float32 numpy array of shape ``(N, 768)``.
            Rows are L2-normalised when ``normalize_embeddings=True`` (default).
        """
        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)
