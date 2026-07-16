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

        if cfg.local_path:
            self._client = QdrantClient(path=cfg.local_path)
            logger.info(
                "Connected to local Qdrant storage at %s (collection=%s)",
                cfg.local_path,
                self._collection,
            )
        else:
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

    def existing_image_ids(self, image_ids: list[str]) -> set[str]:
        """Return the subset of ``image_ids`` already present in the collection.

        This is intentionally a single batched lookup so indexing can skip
        previously stored paths without loading any captioning or embedding
        models for them.
        """
        if not image_ids:
            return set()

        points = self._client.retrieve(
            collection_name=self._collection,
            ids=[_image_id_to_int(image_id) for image_id in image_ids],
            with_payload=True,
            with_vectors=False,
        )
        return {
            str(point.payload["image_id"])
            for point in points
            if point.payload and point.payload.get("image_id")
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_batch(self, records: list[IndexedImage]) -> None:
        """Upsert a batch of ``IndexedImage`` records into Qdrant.

        Each record is stored with:
          - two named vectors (fashionclip + caption)
          - a payload containing image_path, caption, and structured metadata
        """
        points: list[qmodels.PointStruct] = []
        for rec in records:
            metadata = rec.metadata.model_dump()
            payload = {
                "image_id": rec.image_id,
                "image_path": rec.image_path,
                "caption": rec.caption,
                "garments": metadata.get("garments", []),
                "accessories": metadata.get("accessories", []),
                "outfit": metadata.get("outfit", {}),
                "scene": metadata.get("scene", {}),
                "person": metadata.get("person", {}),
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
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using=vector_name,
            limit=top_k,
            query_filter=payload_filter,
            with_payload=True,
        )
        return response.points

    @staticmethod
    def build_metadata_filter(metadata: FashionMetadata) -> Optional[qmodels.Filter]:
        """Build a Qdrant payload filter from non-null/non-empty metadata fields.

        - String fields use ``MatchValue`` (exact match, lowercased).
        - List fields (colors, patterns, styles, occasions) use ``MatchAny``
          so that a query value matches if ANY stored list element equals it.
        - Each garment / accessory becomes its own ``NestedCondition`` so
          compound queries like 'red bag and white top' require the image to
          contain *both* items simultaneously.
        """
        conditions: list[qmodels.Condition] = []

        # --- Garments ---
        for garment in metadata.garments:
            inner: list[qmodels.Condition] = []

            if garment.category:
                inner.append(qmodels.FieldCondition(
                    key="category",
                    match=qmodels.MatchValue(value=garment.category.lower()),
                ))
            if garment.subcategory:
                inner.append(qmodels.FieldCondition(
                    key="subcategory",
                    match=qmodels.MatchText(text=garment.subcategory.lower()),
                ))
            if garment.material:
                inner.append(qmodels.FieldCondition(
                    key="material",
                    match=qmodels.MatchValue(value=garment.material.lower()),
                ))
            if garment.fit:
                inner.append(qmodels.FieldCondition(
                    key="fit",
                    match=qmodels.MatchValue(value=garment.fit.lower()),
                ))
            if garment.colors:
                inner.append(qmodels.FieldCondition(
                    key="colors",
                    match=qmodels.MatchAny(any=[c.lower() for c in garment.colors]),
                ))
            if garment.patterns:
                inner.append(qmodels.FieldCondition(
                    key="patterns",
                    match=qmodels.MatchAny(any=[p.lower() for p in garment.patterns]),
                ))

            if inner:
                conditions.append(qmodels.NestedCondition(
                    nested=qmodels.Nested(
                        key="garments",
                        filter=qmodels.Filter(must=inner),
                    )
                ))

        # --- Accessories ---
        for acc in metadata.accessories:
            inner = []

            if acc.category:
                inner.append(qmodels.FieldCondition(
                    key="category",
                    match=qmodels.MatchValue(value=acc.category.lower()),
                ))
            if acc.subcategory:
                inner.append(qmodels.FieldCondition(
                    key="subcategory",
                    match=qmodels.MatchText(text=acc.subcategory.lower()),
                ))
            if acc.colors:
                inner.append(qmodels.FieldCondition(
                    key="colors",
                    match=qmodels.MatchAny(any=[c.lower() for c in acc.colors]),
                ))

            if inner:
                conditions.append(qmodels.NestedCondition(
                    nested=qmodels.Nested(
                        key="accessories",
                        filter=qmodels.Filter(must=inner),
                    )
                ))

        # --- Outfit-level (styles / occasions) ---
        if metadata.outfit.styles:
            conditions.append(qmodels.FieldCondition(
                key="outfit.styles",
                match=qmodels.MatchAny(any=[s.lower() for s in metadata.outfit.styles]),
            ))
        if metadata.outfit.occasions:
            conditions.append(qmodels.FieldCondition(
                key="outfit.occasions",
                match=qmodels.MatchAny(any=[o.lower() for o in metadata.outfit.occasions]),
            ))

        # --- Scene ---
        if metadata.scene.location:
            conditions.append(qmodels.FieldCondition(
                key="scene.location",
                match=qmodels.MatchValue(value=metadata.scene.location.lower()),
            ))
        if metadata.scene.environment:
            conditions.append(qmodels.FieldCondition(
                key="scene.environment",
                match=qmodels.MatchText(text=metadata.scene.environment.lower()),
            ))
        if metadata.scene.activity:
            conditions.append(qmodels.FieldCondition(
                key="scene.activity",
                match=qmodels.MatchText(text=metadata.scene.activity.lower()),
            ))

        # --- Person ---
        if metadata.person.gender:
            conditions.append(qmodels.FieldCondition(
                key="person.gender",
                match=qmodels.MatchValue(value=metadata.person.gender.lower()),
            ))
        if metadata.person.num_people is not None:
            conditions.append(qmodels.FieldCondition(
                key="person.num_people",
                match=qmodels.MatchValue(value=metadata.person.num_people),
            ))

        return qmodels.Filter(must=conditions) if conditions else None

    def lookup_by_image_ids(self, image_ids: list[str]) -> list[RetrievalResult]:
        """Fetch full retrieval records for the provided image IDs.

        This is the final Qdrant lookup step after reranking so the online
        pipeline follows the architecture: candidate IDs -> rerank -> lookup.
        """
        if not image_ids:
            return []

        point_ids = [_image_id_to_int(image_id) for image_id in image_ids]
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        results = [self._point_to_result(point) for point in points]
        result_map = {result.image_id: result for result in results}
        return [result_map[image_id] for image_id in image_ids if image_id in result_map]

    @staticmethod
    def scored_point_to_result(point: qmodels.ScoredPoint) -> RetrievalResult:
        """Convert a Qdrant ``ScoredPoint`` into a ``RetrievalResult``."""
        payload = point.payload or {}
        result = QdrantStore._payload_to_result(payload)
        result.score = point.score
        return result

    @staticmethod
    def _point_to_result(point: Any) -> RetrievalResult:
        """Convert a Qdrant record or point with payload into ``RetrievalResult``."""
        payload = getattr(point, "payload", None) or {}
        return QdrantStore._payload_to_result(payload)

    @staticmethod
    def _payload_to_result(payload: dict[str, Any]) -> RetrievalResult:
        metadata = QdrantStore._payload_to_metadata(payload)
        return RetrievalResult(
            image_id=payload.get("image_id", ""),
            image_path=payload.get("image_path", ""),
            caption=payload.get("caption", ""),
            metadata=metadata,
        )

    @staticmethod
    def _payload_to_metadata(payload: dict[str, Any]) -> FashionMetadata:
        """Reconstruct ``FashionMetadata`` from a Qdrant payload dict."""
        # Legacy support: old payloads stored everything under 'metadata' key
        if "metadata" in payload and isinstance(payload["metadata"], dict):
            return FashionMetadata(**payload["metadata"])

        return FashionMetadata(
            garments=payload.get("garments", []),
            accessories=payload.get("accessories", []),
            outfit=payload.get("outfit", {}),
            scene=payload.get("scene", {}),
            person=payload.get("person", {}),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _image_id_to_int(image_id: str) -> int:
    """Convert an 'IMG_<hex>' identifier to a stable integer for Qdrant point IDs."""
    hex_part = image_id.replace("IMG_", "")
    return int(hex_part, 16) % (2**63)
