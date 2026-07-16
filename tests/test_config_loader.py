"""
Unit tests for src/utils/config_loader.py

Tests cover:
  - get_app_settings(): loads and validates config.yaml
  - get_model_settings(): loads and validates models.yaml
  - Environment variable overrides for Qdrant
  - Pydantic field defaults and types
"""

from __future__ import annotations

import os

import pytest

from src.utils.config_loader import (
    AppSettings,
    FashionCLIPConfig,
    ModelSettings,
    QdrantConfig,
    RetrievalConfig,
    TextEmbeddingConfig,
    VectorSpec,
    VectorsConfig,
    get_app_settings,
    get_model_settings,
)


class TestGetAppSettings:
    def test_loads_successfully(self):
        # Clear cache to ensure fresh load
        get_app_settings.cache_clear()
        settings = get_app_settings()
        assert isinstance(settings, AppSettings)

    def test_app_name(self):
        get_app_settings.cache_clear()
        settings = get_app_settings()
        assert settings.app.name == "fashion-retrieval-system"

    def test_qdrant_defaults(self):
        get_app_settings.cache_clear()
        settings = get_app_settings()
        assert settings.qdrant.local_path == ".qdrant"
        assert settings.qdrant.host == "localhost"
        assert settings.qdrant.port == 6333
        assert settings.qdrant.collection_name == "fashion_images"

    def test_vectors_config(self):
        get_app_settings.cache_clear()
        settings = get_app_settings()
        assert settings.vectors.fashionclip.size == 512
        assert settings.vectors.caption.size == 768
        assert settings.vectors.fashionclip.distance == "Cosine"

    def test_indexing_defaults(self):
        get_app_settings.cache_clear()
        settings = get_app_settings()
        assert settings.indexing.batch_size == 16
        assert ".jpg" in settings.indexing.image_extensions

    def test_retrieval_defaults(self):
        get_app_settings.cache_clear()
        settings = get_app_settings()
        assert settings.retrieval.retrieval_top_k == 100
        assert settings.retrieval.rerank_top_k == 10
        assert settings.retrieval.fashionclip_weight + settings.retrieval.caption_weight == pytest.approx(1.0)

    def test_logging_defaults(self):
        get_app_settings.cache_clear()
        settings = get_app_settings()
        assert settings.logging.level == "INFO"

    def test_qdrant_host_env_override(self, monkeypatch):
        get_app_settings.cache_clear()
        monkeypatch.setenv("QDRANT_HOST", "qdrant-cloud.example.com")
        settings = get_app_settings()
        assert settings.qdrant.host == "qdrant-cloud.example.com"
        get_app_settings.cache_clear()  # Clean up

    def test_qdrant_local_path_env_override(self, monkeypatch):
        get_app_settings.cache_clear()
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "D:/tmp/qdrant_local")
        settings = get_app_settings()
        assert settings.qdrant.local_path == "D:/tmp/qdrant_local"
        get_app_settings.cache_clear()  # Clean up

    def test_qdrant_api_key_env_override(self, monkeypatch):
        get_app_settings.cache_clear()
        monkeypatch.setenv("QDRANT_API_KEY", "test-secret-key")
        settings = get_app_settings()
        assert settings.qdrant.api_key == "test-secret-key"
        get_app_settings.cache_clear()  # Clean up


class TestGetModelSettings:
    def test_loads_successfully(self):
        get_model_settings.cache_clear()
        settings = get_model_settings()
        assert isinstance(settings, ModelSettings)

    def test_vlm_config(self):
        get_model_settings.cache_clear()
        settings = get_model_settings()
        assert "Qwen" in settings.vision_language_model.model_name
        assert settings.vision_language_model.max_new_tokens == 400

    def test_fashionclip_config(self):
        get_model_settings.cache_clear()
        settings = get_model_settings()
        assert "fashion-clip" in settings.fashionclip.model_name

    def test_text_embedding_config(self):
        get_model_settings.cache_clear()
        settings = get_model_settings()
        assert "bge" in settings.text_embedding.model_name.lower()
        assert settings.text_embedding.normalize_embeddings is True

    def test_cross_encoder_config(self):
        get_model_settings.cache_clear()
        settings = get_model_settings()
        assert settings.cross_encoder.model_name == "BAAI/bge-reranker-v2-m3"
        assert settings.cross_encoder.max_length == 512

    def test_query_parser_config(self):
        get_model_settings.cache_clear()
        settings = get_model_settings()
        assert settings.query_parser.max_new_tokens == 512


class TestPydanticModels:
    def test_vector_spec_validation(self):
        v = VectorSpec(name="test_vec", size=256, distance="Cosine")
        assert v.size == 256

    def test_retrieval_config_defaults(self):
        r = RetrievalConfig()
        assert r.retrieval_top_k == 50
        assert r.rerank_top_k == 10

    def test_fashionclip_config(self):
        f = FashionCLIPConfig(model_name="test/model")
        assert f.device == "auto"

    def test_text_embedding_config_defaults(self):
        t = TextEmbeddingConfig(model_name="test/bge")
        assert t.normalize_embeddings is True
