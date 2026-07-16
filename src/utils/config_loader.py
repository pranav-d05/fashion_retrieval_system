"""
Configuration loader — loads and validates config.yaml and models.yaml.
Environment variable overrides are applied after YAML loading.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_CONFIGS = Path(__file__).parents[2] / "configs"


def _load(filename: str) -> dict:
    path = _CONFIGS / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# config.yaml models
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    name: str = "fashion-retrieval-system"
    version: str = "1.0.0"


class QdrantConfig(BaseModel):
    local_path: str | None = ".qdrant"
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "fashion_images"
    api_key: str | None = None


class VectorSpec(BaseModel):
    name: str
    size: int
    distance: str = "Cosine"


class VectorsConfig(BaseModel):
    fashionclip: VectorSpec
    caption: VectorSpec


class IndexingConfig(BaseModel):
    batch_size: int = 16
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"]
    )


class RetrievalConfig(BaseModel):
    retrieval_top_k: int = 50
    rerank_top_k: int = 10
    fashionclip_weight: float = 0.5
    caption_weight: float = 0.5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "text"


class AppSettings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    vectors: VectorsConfig
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# models.yaml models
# ---------------------------------------------------------------------------

class VLMConfig(BaseModel):
    model_name: str
    max_new_tokens: int = 512
    device: str = "auto"


class QueryParserConfig(BaseModel):
    model_name: str
    max_new_tokens: int = 256
    device: str = "auto"


class FashionCLIPConfig(BaseModel):
    model_name: str
    device: str = "auto"


class TextEmbeddingConfig(BaseModel):
    model_name: str
    device: str = "auto"
    normalize_embeddings: bool = True


class CrossEncoderConfig(BaseModel):
    model_name: str
    device: str = "auto"
    max_length: int = 512


class ModelSettings(BaseModel):
    vision_language_model: VLMConfig
    query_parser: QueryParserConfig
    fashionclip: FashionCLIPConfig
    text_embedding: TextEmbeddingConfig
    cross_encoder: CrossEncoderConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_app_settings() -> AppSettings:
    """Load and cache application settings from config.yaml.

    Qdrant settings can be overridden at runtime via:
        QDRANT_LOCAL_PATH=<path>
        QDRANT_HOST=<host>
        QDRANT_API_KEY=<key>
    """
    data = _load("config.yaml")

    # Apply environment variable overrides for infrastructure config
    if local_path := os.getenv("QDRANT_LOCAL_PATH"):
        data.setdefault("qdrant", {})["local_path"] = local_path
    if host := os.getenv("QDRANT_HOST"):
        data.setdefault("qdrant", {})["host"] = host
    if key := os.getenv("QDRANT_API_KEY"):
        data.setdefault("qdrant", {})["api_key"] = key or None

    return AppSettings(**data)


@lru_cache(maxsize=1)
def get_model_settings() -> ModelSettings:
    """Load and cache model settings from models.yaml."""
    return ModelSettings(**_load("models.yaml"))
