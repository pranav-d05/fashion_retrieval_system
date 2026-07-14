"""
Hybrid Retriever — combines FashionCLIP and BGE vector searches
using Reciprocal Rank Fusion (RRF) and optional metadata payload filtering.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
from src.embeddings.text_embedder import TextEmbedder
from src.qdrant_store import QdrantStore
from src.schemas import FashionMetadata, RetrievalResult
from src.utils.config_loader import AppSettings

logger = logging.getLogger(__name__)

# RRF smoothing constant (standard value from the original paper)
_RRF_K = 60


class Retriever:
    """Performs hybrid retrieval combining FashionCLIP + BGE vector search.

    Merges the two ranked lists from Qdrant using Reciprocal Rank Fusion (RRF),
    weighted by the configured ``fashionclip_weight`` and ``caption_weight``.

    Args:
        settings:       Application settings.
        clip_embedder:  FashionCLIP text encoder.
        text_embedder:  BGE text encoder.
        store:          Qdrant store instance.
    """

    def __init__(
        self,
        settings: AppSettings,
        clip_embedder: FashionCLIPEmbedder,
        text_embedder: TextEmbedder,
        store: QdrantStore,
    ) -> None:
        self._cfg = settings.retrieval
        self._vec_cfg = settings.vectors
        self._clip_embedder = clip_embedder
        self._text_embedder = text_embedder
        self._store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        parsed_metadata: FashionMetadata,
    ) -> list[RetrievalResult]:
        """Run hybrid retrieval for a user query.

        Args:
            query:           Raw user query string.
            parsed_metadata: Structured attributes extracted from the query.

        Returns:
            Up to ``retrieval_top_k`` candidates sorted by descending RRF score.
        """
        top_k = self._cfg.retrieval_top_k
        clip_w = self._cfg.fashionclip_weight
        caption_w = self._cfg.caption_weight

        # Build optional payload filter from structured metadata
        payload_filter = self._store.build_metadata_filter(parsed_metadata)
        if payload_filter:
            logger.debug("Applying metadata payload filter.")

        # --- FashionCLIP vector search ---
        clip_query = self._clip_embedder.encode_texts([query])[0].tolist()
        clip_hits = self._store.search_by_vector(
            vector_name=self._vec_cfg.fashionclip.name,
            query_vector=clip_query,
            top_k=top_k,
            payload_filter=payload_filter,
        )

        # --- BGE caption vector search ---
        bge_query = self._text_embedder.encode([query])[0].tolist()
        caption_hits = self._store.search_by_vector(
            vector_name=self._vec_cfg.caption.name,
            query_vector=bge_query,
            top_k=top_k,
            payload_filter=payload_filter,
        )

        logger.debug(
            "Vector search returned %d (clip) + %d (caption) hits.",
            len(clip_hits),
            len(caption_hits),
        )

        # --- RRF fusion ---
        fused = _reciprocal_rank_fusion(
            ranked_lists=[clip_hits, caption_hits],
            weights=[clip_w, caption_w],
        )

        # Convert to RetrievalResult, preserving fused score
        results: list[RetrievalResult] = []
        for point, rrf_score in fused[:top_k]:
            result = self._store.scored_point_to_result(point)
            result.score = rrf_score
            results.append(result)

        logger.info("Retrieved %d candidates after RRF fusion.", len(results))
        return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _reciprocal_rank_fusion(
    ranked_lists: list,
    weights: list[float],
    k: int = _RRF_K,
):
    """Merge multiple ranked lists into one using weighted RRF.

    Args:
        ranked_lists: Each element is a list of ``ScoredPoint`` objects,
                      ordered by descending score.
        weights:      Per-list weight multiplied into the RRF contribution.
        k:            Smoothing constant (default 60).

    Returns:
        List of ``(ScoredPoint, rrf_score)`` tuples sorted by descending
        ``rrf_score``.
    """
    # point_id -> (best ScoredPoint, accumulated RRF score)
    score_map: dict[int, list] = defaultdict(lambda: [None, 0.0])

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, point in enumerate(ranked_list, start=1):
            pid = point.id
            rrf_contribution = weight / (k + rank)
            if score_map[pid][0] is None:
                score_map[pid][0] = point
            score_map[pid][1] += rrf_contribution

    # Sort by descending RRF score
    sorted_items = sorted(score_map.values(), key=lambda x: x[1], reverse=True)
    return [(item[0], item[1]) for item in sorted_items]
