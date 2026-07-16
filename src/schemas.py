"""
Shared Pydantic schemas for the Fashion Retrieval System.

These models are the single source of truth for data structures used
across indexing, retrieval, VLM prompting, and Qdrant payloads.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared coercion helpers
# ---------------------------------------------------------------------------

def _coerce_str(v: object) -> str | None:
    """Coerce a value to a clean string, or None if absent/blank.

    Handles common LLM output quirks:
    - null / None     → None
    - list of strings → joined with ' and '  (e.g. ['navy', 'white'] → 'navy and white')
    - non-string      → str()
    - blank string    → None
    """
    if v is None:
        return None
    if isinstance(v, list):
        parts = [str(item).strip().lower() for item in v if item is not None and str(item).strip()]
        return " and ".join(parts) if parts else None
    if isinstance(v, str):
        stripped = v.strip().lower()
        return stripped if stripped else None
    return str(v).lower() or None


def _coerce_str_list(v: object) -> list[str]:
    """Coerce a value to a clean list of non-empty strings.

    Handles common LLM output quirks:
    - null / None          → []
    - "single string"      → ["single string"]  (model forgot to use array)
    - ["a", None, "b"]     → ["a", "b"]         (remove nulls)
    - already a list       → cleaned list
    """
    if v is None:
        return []
    if isinstance(v, str):
        stripped = v.strip().lower()
        return [stripped] if stripped else []
    if isinstance(v, list):
        return [str(item).strip().lower() for item in v if item is not None and str(item).strip()]
    return []


# ---------------------------------------------------------------------------
# Garment-level attributes
# ---------------------------------------------------------------------------


class Garment(BaseModel):
    """Attributes for a single clothing garment detected in an image."""

    category: Optional[str] = None
    """High-level garment type: 'top', 'bottom', 'dress', 'outerwear', 'skirt', 'jumpsuit'."""

    subcategory: Optional[str] = None
    """Specific item: 'crew-neck t-shirt', 'slim-fit jeans', 'maxi dress', 'raincoat'."""

    colors: list[str] = Field(default_factory=list)
    """Dominant colors, e.g. ['navy', 'white']. Empty list if unknown."""

    patterns: list[str] = Field(default_factory=list)
    """Surface patterns, e.g. ['striped', 'floral']. Empty list if plain/unknown."""

    material: Optional[str] = None
    """Fabric / material, e.g. 'cotton', 'denim', 'silk', 'leather'."""

    fit: Optional[str] = None
    """Silhouette / fit, e.g. 'slim', 'relaxed', 'oversized', 'fitted'."""

    length: Optional[str] = None
    """Hem length, e.g. 'mini', 'midi', 'maxi', 'knee-length', 'ankle-length'."""

    neckline: Optional[str] = None
    """Neckline style, e.g. 'v-neck', 'crew-neck', 'square', 'turtleneck', 'off-shoulder'."""

    @field_validator("category", "subcategory", "material", "fit", "length", "neckline", mode="before")
    @classmethod
    def _coerce_string(cls, v: object) -> str | None:
        return _coerce_str(v)

    @field_validator("colors", "patterns", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[str]:
        return _coerce_str_list(v)


# ---------------------------------------------------------------------------
# Accessory-level attributes (bags, shoes, jewellery, hats, etc.)
# ---------------------------------------------------------------------------


class Accessory(BaseModel):
    """Attributes for a single accessory detected in an image."""

    category: Optional[str] = None
    """Accessory type: 'bag', 'shoes', 'hat', 'jewellery', 'belt', 'watch',
    'sunglasses', 'scarf', 'neckwear'."""

    subcategory: Optional[str] = None
    """Specific item: 'tote bag', 'sneakers', 'hoop earrings', 'tie', 'bow-tie'."""

    colors: list[str] = Field(default_factory=list)
    """Dominant colors, e.g. ['black', 'gold']. Empty list if unknown."""

    @field_validator("category", "subcategory", mode="before")
    @classmethod
    def _coerce_string(cls, v: object) -> str | None:
        return _coerce_str(v)

    @field_validator("colors", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[str]:
        return _coerce_str_list(v)


# ---------------------------------------------------------------------------
# Outfit-level attributes (apply to the whole look, not one garment)
# ---------------------------------------------------------------------------


class Outfit(BaseModel):
    """Overall outfit-level attributes that span across all garments."""

    styles: list[str] = Field(default_factory=list)
    """Fashion styles of the whole look, e.g. ['casual', 'streetwear', 'bohemian']."""

    occasions: list[str] = Field(default_factory=list)
    """Intended occasions, e.g. ['everyday', 'workwear', 'party', 'beach']."""

    @field_validator("styles", "occasions", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[str]:
        return _coerce_str_list(v)


# ---------------------------------------------------------------------------
# Scene and person context
# ---------------------------------------------------------------------------


class SceneInfo(BaseModel):
    """Scene-level context extracted from the image."""

    location: Optional[str] = None
    """Setting, e.g. 'indoors', 'outdoors', 'studio', 'urban street'."""

    environment: Optional[str] = None
    """Broader environment, e.g. 'beach', 'forest', 'city', 'office'."""

    activity: Optional[str] = None
    """Activity being performed, e.g. 'walking', 'posing', 'sitting'."""

    @field_validator("location", "environment", "activity", mode="before")
    @classmethod
    def _coerce_string(cls, v: object) -> str | None:
        return _coerce_str(v)


class PersonInfo(BaseModel):
    """Person-level attributes extracted from the image."""

    gender: Optional[str] = None
    """Perceived gender presentation, e.g. 'woman', 'man', 'unisex'."""

    num_people: Optional[int] = None
    """Number of people visible in the image."""

    @field_validator("gender", mode="before")
    @classmethod
    def _coerce_string(cls, v: object) -> str | None:
        return _coerce_str(v)

    @field_validator("num_people", mode="before")
    @classmethod
    def _coerce_int(cls, v: object) -> int | None:
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            return int(stripped) if stripped.isdigit() else None
        return None


# ---------------------------------------------------------------------------
# Top-level metadata container
# ---------------------------------------------------------------------------


class FashionMetadata(BaseModel):
    """Structured metadata for a fashion image.

    Used both as the VLM output during offline indexing and as the parsed
    query representation during online retrieval.
    """

    garments: list[Garment] = Field(default_factory=list)
    """Clothing garments detected in the image (one entry per distinct item)."""

    accessories: list[Accessory] = Field(default_factory=list)
    """Accessories detected in the image: bags, shoes, jewellery, hats, etc."""

    outfit: Outfit = Field(default_factory=Outfit)
    """Outfit-level style and occasion attributes for the overall look."""

    scene: SceneInfo = Field(default_factory=SceneInfo)
    """Scene-level contextual information."""

    person: PersonInfo = Field(default_factory=PersonInfo)
    """Person-level attributes."""


# ---------------------------------------------------------------------------
# Qdrant payload / index record
# ---------------------------------------------------------------------------


class IndexedImage(BaseModel):
    """Full record stored in Qdrant for a single indexed image.

    ``fashionclip_embedding`` and ``caption_embedding`` are stored as
    named vectors; everything else lives in the payload.
    """

    image_id: str
    """Stable unique identifier derived from the image path (SHA-256 prefix)."""

    image_path: str
    """Absolute or dataset-relative path to the original image file."""

    caption: str
    """Free-text natural language description generated by the VLM."""

    metadata: FashionMetadata
    """Structured fashion attributes extracted by the VLM."""

    fashionclip_embedding: list[float]
    """FashionCLIP image embedding (dim=512, L2-normalised)."""

    caption_embedding: list[float]
    """BGE caption embedding (dim=768, L2-normalised)."""


# ---------------------------------------------------------------------------
# Retrieval output
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """Single result returned by the retrieval pipeline."""

    image_id: str
    image_path: str
    caption: str
    metadata: FashionMetadata
    score: float = 0.0
    """Final relevance score (higher is better). Set by the reranker."""
