"""
Cross-Encoder Re-ranker.

Scores ``(query, caption)`` pairs jointly using a sentence-transformers
CrossEncoder. The reranked results form the final output of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.qdrant_store import QdrantStore
from src.schemas import RetrievalResult
from src.utils.config_loader import AppSettings, CrossEncoderConfig

logger = logging.getLogger(__name__)

# Kept module-level for straightforward dependency injection in tests, while
# avoiding an eager sentence-transformers import for callers that never build
# a reranker (and for lightweight CLI/help operations).
CrossEncoder: Any | None = None


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _get_cross_encoder() -> Any:
    """Load the optional heavyweight CrossEncoder dependency on demand."""
    global CrossEncoder
    if CrossEncoder is None:
        from sentence_transformers import CrossEncoder as SentenceCrossEncoder

        CrossEncoder = SentenceCrossEncoder
    return CrossEncoder


class Reranker:
    """Re-ranks retrieval candidates using a Cross-Encoder.

    Args:
        config:   ``CrossEncoderConfig`` from ``models.yaml``.
        settings: Application settings (for ``rerank_top_k``).
    """

    def __init__(self, config: CrossEncoderConfig, settings: AppSettings) -> None:
        device = _resolve_device(config.device)
        logger.info(
            "Loading CrossEncoder '%s' (device=%s)…",
            config.model_name,
            device,
        )
        model_kwargs = {}
        if device != "cpu" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16

        self._model = _get_cross_encoder()(
            config.model_name,
            max_length=config.max_length,
            device=device if device != "cpu" else None,  # sentence-transformers prefers None for CPU
            model_kwargs=model_kwargs,
        )
        self._top_k = settings.retrieval.rerank_top_k
        self._store: QdrantStore | None = None
        logger.info("CrossEncoder loaded successfully.")

    def attach_store(self, store: QdrantStore) -> None:
        """Attach the Qdrant store used for the final result lookup."""
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-rank candidate results by cross-encoder relevance score.

        Args:
            query:      The user's original search query.
            candidates: Retrieved candidates from the hybrid retriever.

        Returns:
            Top-K results sorted by descending cross-encoder score.
        """
        if not candidates:
            return []

        pairs = [(query, c.caption) for c in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()

        # Attach scores and sort descending
        for result, score in zip(candidates, scores):
            result.score = score

        ranked = sorted(candidates, key=lambda r: r.score, reverse=True)
        top = ranked[: self._top_k]

        if self._store is None:
            logger.warning("No Qdrant store attached; returning reranked candidates directly.")
            return top

        hydrated = self._store.lookup_by_image_ids([candidate.image_id for candidate in top])
        hydrated_map = {result.image_id: result for result in hydrated}
        final_results: list[RetrievalResult] = []
        for candidate in top:
            result = hydrated_map.get(candidate.image_id, candidate)
            result.score = candidate.score
            final_results.append(result)

        logger.info(
            "Reranking complete. Top score=%.4f, bottom score=%.4f (of %d).",
            final_results[0].score if final_results else float("nan"),
            final_results[-1].score if final_results else float("nan"),
            len(candidates),
        )
        return final_results
