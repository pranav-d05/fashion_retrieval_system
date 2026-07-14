"""
Cross-Encoder Re-ranker.

Scores ``(query, caption)`` pairs jointly using a sentence-transformers
CrossEncoder. The reranked results form the final output of the pipeline.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from src.schemas import RetrievalResult
from src.utils.config_loader import AppSettings, CrossEncoderConfig

logger = logging.getLogger(__name__)


class Reranker:
    """Re-ranks retrieval candidates using a Cross-Encoder.

    Args:
        config:   ``CrossEncoderConfig`` from ``models.yaml``.
        settings: Application settings (for ``rerank_top_k``).
    """

    def __init__(self, config: CrossEncoderConfig, settings: AppSettings) -> None:
        device = config.device if config.device != "auto" else None
        logger.info(
            "Loading CrossEncoder '%s' (device=%s)…",
            config.model_name,
            device or "auto",
        )
        self._model = CrossEncoder(
            config.model_name,
            max_length=config.max_length,
            device=device,
        )
        self._top_k = settings.retrieval.rerank_top_k
        logger.info("CrossEncoder loaded successfully.")

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

        logger.info(
            "Reranking complete. Top score=%.4f, bottom score=%.4f (of %d).",
            top[0].score if top else float("nan"),
            top[-1].score if top else float("nan"),
            len(candidates),
        )
        return top
