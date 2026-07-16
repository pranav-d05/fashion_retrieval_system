"""
Unit tests for the reranker final hydration step.

These tests keep the online pipeline aligned with the architecture:
candidate results are scored first, then hydrated from Qdrant by image_id
before being returned.
"""

from __future__ import annotations

import numpy as np

from src.retrieval.reranker import Reranker
from src.schemas import FashionMetadata, RetrievalResult
from src.utils.config_loader import (
    AppSettings,
    CrossEncoderConfig,
    RetrievalConfig,
    VectorSpec,
    VectorsConfig,
)


class _FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return np.array([0.2, 0.9], dtype=float)


class _FakeStore:
    def lookup_by_image_ids(self, image_ids):
        return [
            RetrievalResult(
                image_id="IMG_002",
                image_path="/data/img2.jpg",
                caption="Hydrated caption 2",
                metadata=FashionMetadata(),
            ),
            RetrievalResult(
                image_id="IMG_001",
                image_path="/data/img1.jpg",
                caption="Hydrated caption 1",
                metadata=FashionMetadata(),
            ),
        ]


class TestReranker:
    def test_rerank_hydrates_final_results_from_store(self, monkeypatch):
        monkeypatch.setattr("src.retrieval.reranker.CrossEncoder", _FakeCrossEncoder)

        settings = AppSettings(
            vectors=VectorsConfig(
                fashionclip=VectorSpec(name="fashionclip_embedding", size=512),
                caption=VectorSpec(name="caption_embedding", size=768),
            ),
            retrieval=RetrievalConfig(rerank_top_k=2),
        )
        reranker = Reranker(
            CrossEncoderConfig(model_name="fake/model", device="cpu", max_length=64),
            settings,
        )
        reranker.attach_store(_FakeStore())

        candidates = [
            RetrievalResult(
                image_id="IMG_001",
                image_path="/data/raw1.jpg",
                caption="Candidate caption 1",
                metadata=FashionMetadata(),
            ),
            RetrievalResult(
                image_id="IMG_002",
                image_path="/data/raw2.jpg",
                caption="Candidate caption 2",
                metadata=FashionMetadata(),
            ),
        ]

        results = reranker.rerank("red dress", candidates)

        assert [result.image_id for result in results] == ["IMG_002", "IMG_001"]
        assert [result.caption for result in results] == [
            "Hydrated caption 2",
            "Hydrated caption 1",
        ]
        assert results[0].score == 0.9
        assert results[1].score == 0.2
