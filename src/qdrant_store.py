"""
Qdrant vector store wrapper for the Fashion Retrieval System.

Responsibilities:
  - Connect to a local or Cloud Qdrant instance.
  - Create / verify the named-vector collection.
  - Upsert batches of IndexedImage records.
  - Execute hybrid (multi-vector) retrieval with optional payload filters.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.schemas import FashionMetadata, IndexedImage, RetrievalResult
from src.utils.config_loader import AppSettings, VectorSpec

logger = logging.getLogger(__name__)


class QdrantStore:
    """Thin wrapper around ``QdrantClient`` providing collection management
    and domain-aware upsert / search helpers."""

    def __init__(self, settings: AppSettings) -> None:
        cfg = settings.qdrant
        self._collection = cfg.collection_name
        self._vec_cfg = settings.vectors

        connect_kwargs: dict[str, Any] = {"host": cfg.host, "port": cfg.port}
        if cfg.api_key:
            connect_kwargs["api_key"] = cfg.api_key

        self._client = QdrantClient(**connect_kwargs)
        logger.info(
            "Connected to Qdrant at %s:%s (collection=%s)",
            cfg.host,
            cfg.port,
            self._collection,
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self) -> bool:
        """Return True if the target collection already exists."""
        return self._client.collection_exists(self._collection)

    def ensure_collection(self) -> None:
        """Create the named-vector collection if it does not yet exist."""
        if self.collection_exists():
            logger.info("Collection '%s' already exists — skipping creation.", self._collection)
            return

        def _vec_params(spec: VectorSpec) -> qmodels.VectorParams:
            return qmodels.VectorParams(
                size=spec.size,
                distance=qmodels.Distance[spec.distance.upper()],
            )

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                self._vec_cfg.fashionclip.name: _vec_params(self._vec_cfg.fashionclip),
                self._vec_cfg.caption.name: _vec_params(self._vec_cfg.caption),
            },
        )
        logger.info("Created collection '%s'.", self._collection)

    def count(self) -> int:
        """Return the number of points in the collection."""
        result = self._client.count(collection_name=self._collection)
        return result.count

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_batch(self, records: list[IndexedImage]) -> None:
        """Upsert a batch of ``IndexedImage`` records into Qdrant.

        Each record is stored with:
          - two named vectors (fashionclip + caption)
          - a payload containing image_path, caption, and full metadata
        """
        points: list[qmodels.PointStruct] = []
        for rec in records:
            payload = {
                "image_id": rec.image_id,
                "image_path": rec.image_path,
                "caption": rec.caption,
                # Store metadata as a nested dict for payload filtering
                "metadata": rec.metadata.model_dump(),
            }
            points.append(
                qmodels.PointStruct(
                    id=_image_id_to_int(rec.image_id),
                    vector={
                        self._vec_cfg.fashionclip.name: rec.fashionclip_embedding,
                        self._vec_cfg.caption.name: rec.caption_embedding,
                    },
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self._collection, points=points)
        logger.debug("Upserted %d points.", len(points))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_by_vector(
        self,
        vector_name: str,
        query_vector: list[float],
        top_k: int,
        payload_filter: Optional[qmodels.Filter] = None,
    ) -> list[qmodels.ScoredPoint]:
        """Search a single named vector with optional payload filtering."""
        return self._client.search(
            collection_name=self._collection,
            query_vector=qmodels.NamedVector(name=vector_name, vector=query_vector),
            limit=top_k,
            query_filter=payload_filter,
            with_payload=True,
        )

    @staticmethod
    def build_metadata_filter(metadata: FashionMetadata) -> Optional[qmodels.Filter]:
        """Build a Qdrant payload filter from non-null metadata fields.

        Only the first garment's category is used to avoid over-filtering.
        Scene and person fields are also included when present.
        """
        conditions: list[qmodels.Condition] = []

        # Garment category — use NestedCondition to filter any garment in the array.
        # Qdrant does NOT support array-index notation like "garments[0].category".
        if metadata.garments:
            first = metadata.garments[0]
            if first.category:
                conditions.append(
                    qmodels.NestedCondition(
                        nested=qmodels.Nested(
                            key="metadata.garments",
                            filter=qmodels.Filter(
                                must=[
                                    qmodels.FieldCondition(
                                        key="category",
                                        match=qmodels.MatchValue(
                                            value=first.category.lower()
                                        ),
                                    )
                                ]
                            ),
                        )
                    )
                )

        # Scene environment
        if metadata.scene.environment:
            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.scene.environment",
                    match=qmodels.MatchValue(value=metadata.scene.environment.lower()),
                )
            )

        # Person gender
        if metadata.person.gender:
            conditions.append(
                qmodels.FieldCondition(
                    key="metadata.person.gender",
                    match=qmodels.MatchValue(value=metadata.person.gender.lower()),
                )
            )

        if not conditions:
            return None

        return qmodels.Filter(must=conditions)

    @staticmethod
    def scored_point_to_result(point: qmodels.ScoredPoint) -> RetrievalResult:
        """Convert a Qdrant ``ScoredPoint`` into a ``RetrievalResult``."""
        payload = point.payload or {}
        metadata_dict = payload.get("metadata", {})
        return RetrievalResult(
            image_id=payload.get("image_id", ""),
            image_path=payload.get("image_path", ""),
            caption=payload.get("caption", ""),
            metadata=FashionMetadata(**metadata_dict),
            score=point.score,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _image_id_to_int(image_id: str) -> int:
    """Convert an 'IMG_<hex>' identifier to a stable integer for Qdrant point IDs."""
    hex_part = image_id.replace("IMG_", "")
    return int(hex_part, 16) % (2**63)
