"""
Canonical vocabulary and normalization for structured fashion metadata.

Both the offline metadata extractor (vlm/metadata_extractor.py) and the
online query parser (retrieval/query_parser.py) prompt a small
Qwen2.5-1.5B model with a canonical vocabulary for category/scene/gender
fields, but a model this size does not reliably obey a vocabulary list
given only as prompt instructions -- it will happily emit "hoodie" for a
coat, or invent an accessory that doesn't exist.

``normalize_metadata_vocab`` enforces the vocabulary in code: any value
that isn't an exact canonical match (or a known synonym) is dropped to
None/removed rather than trusted. This runs on both the offline and
online side so that (a) stored payload values and (b) parsed query
values end up in the same normalized form -- which is what makes exact
Qdrant filtering on them safe to rely on.

Only fields that are actually used as hard filters in
``QdrantStore.build_metadata_filter`` are normalized here: garment/
accessory category and colour, and scene/person fields. Material, fit,
length, neckline, subcategory, and inferred style/occasion are
open-vocabulary, are not hard-filtered, and are intentionally left
untouched -- they still feed FashionCLIP/BGE similarity and the
cross-encoder reranker via the caption text.

Tune the *_SYNONYMS maps and the canonical sets against your actual
indexed vocabulary as you observe real extractor/parser output -- these
are reasonable starting points, not a finished taxonomy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas import FashionMetadata

# ---------------------------------------------------------------------------
# Garment category
# ---------------------------------------------------------------------------

GARMENT_CATEGORIES = frozenset({"top", "bottom", "dress", "outerwear", "skirt", "jumpsuit"})

GARMENT_CATEGORY_SYNONYMS = {
    "hoodie": "top", "hoody": "top", "sweatshirt": "top", "sweater": "top",
    "jumper": "top", "shirt": "top", "blouse": "top", "tee": "top",
    "t-shirt": "top", "tshirt": "top", "tank top": "top", "tank": "top",
    "cami": "top", "polo": "top", "vest": "top",
    "jeans": "bottom", "trousers": "bottom", "pants": "bottom",
    "shorts": "bottom", "leggings": "bottom", "chinos": "bottom",
    "coat": "outerwear", "jacket": "outerwear", "parka": "outerwear",
    "raincoat": "outerwear", "blazer": "outerwear", "cardigan": "outerwear",
    "trench coat": "outerwear", "trench": "outerwear", "windbreaker": "outerwear",
    "puffer": "outerwear", "puffer jacket": "outerwear",
    "gown": "dress", "frock": "dress", "sundress": "dress",
    "romper": "jumpsuit", "playsuit": "jumpsuit", "overalls": "jumpsuit", "dungarees": "jumpsuit",
}
# "hoodie" is mapped to "top" (pullover convention). If your indexed data
# tags hooded outerwear (parkas, hooded coats) as "hoodie" too, a bare
# "hoodie" mention is genuinely ambiguous between top/outerwear for your
# dataset specifically -- check real extractor output and adjust.

# ---------------------------------------------------------------------------
# Accessory category
# ---------------------------------------------------------------------------

ACCESSORY_CATEGORIES = frozenset({
    "bag", "shoes", "hat", "jewellery", "belt", "watch",
    "sunglasses", "scarf", "neckwear",
})

ACCESSORY_CATEGORY_SYNONYMS = {
    "headwear": "hat", "cap": "hat", "beanie": "hat", "beret": "hat", "fedora": "hat",
    "sneakers": "shoes", "boots": "shoes", "heels": "shoes", "sandals": "shoes",
    "flats": "shoes", "loafers": "shoes", "trainers": "shoes",
    "earrings": "jewellery", "necklace": "jewellery", "bracelet": "jewellery",
    "ring": "jewellery", "jewelry": "jewellery",
    "tie": "neckwear", "bowtie": "neckwear", "bow-tie": "neckwear", "bow tie": "neckwear",
    "shades": "sunglasses", "purse": "bag", "handbag": "bag",
    "backpack": "bag", "tote": "bag", "clutch": "bag", "crossbody bag": "bag",
}
# Note: "hood" is deliberately absent from this map. A hood is a garment
# feature (part of a coat/jacket), not a standalone accessory. A parsed
# accessory entry with category "hood" fails validation and is dropped
# entirely -- that's the correct outcome, not a gap to fill in.

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

COLORS = frozenset({
    "black", "white", "grey", "red", "orange", "yellow", "green", "blue",
    "navy", "purple", "pink", "brown", "beige", "tan", "cream", "gold",
    "silver", "multicolor",
})

COLOR_SYNONYMS = {
    "gray": "grey", "charcoal": "grey", "graphite": "grey", "slate": "grey",
    "navy blue": "navy", "dark blue": "navy", "midnight blue": "navy",
    "light blue": "blue", "sky blue": "blue", "baby blue": "blue",
    "teal": "blue", "turquoise": "blue", "cobalt": "blue", "denim blue": "blue",
    "khaki": "tan", "olive": "green", "sage": "green", "mint": "green", "emerald": "green",
    "maroon": "red", "burgundy": "red", "crimson": "red", "wine": "red", "scarlet": "red",
    "mustard": "yellow", "lemon": "yellow",
    "ivory": "cream", "off-white": "white", "offwhite": "white", "eggshell": "cream",
    "lavender": "purple", "lilac": "purple", "violet": "purple", "plum": "purple",
    "magenta": "pink", "fuchsia": "pink", "salmon": "pink", "rose": "pink",
    "peach": "orange", "coral": "orange", "rust": "orange", "terracotta": "orange",
    "chocolate": "brown", "espresso": "brown", "camel": "tan", "nude": "tan", "taupe": "tan",
}
# ~18-colour palette. If the product needs finer distinctions (e.g. "navy"
# vs "royal blue" as separately filterable colours), split them here AND
# re-index existing images -- the palette here must match what's actually
# stored in the payload, in both directions.

# ---------------------------------------------------------------------------
# Person / scene
# ---------------------------------------------------------------------------

GENDERS = frozenset({"woman", "man", "person"})

SCENE_LOCATIONS = frozenset({"indoors", "outdoors", "studio"})

SCENE_ENVIRONMENTS = frozenset({
    "office", "street", "park", "home", "beach", "runway", "mall", "urban",
})
# "urban" isn't in the model prompt's original canonical list but is set
# deliberately by query_parser._normalize_metadata_for_query for
# city/walk-style queries, so it has to be treated as canonical here too.

SCENE_ACTIVITIES = frozenset({"walking", "posing", "sitting", "standing"})


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_value(value: str | None, canonical: frozenset, synonyms: dict) -> str | None:
    """Return the canonical form of ``value``, or None if it can't be trusted.

    Order: exact canonical match -> known synonym -> drop (None). We never
    guess at a mapping for an unrecognized value -- anything outside both
    sets is more likely a small-model hallucination than a legitimate new
    vocabulary item, and an exact-match Qdrant filter would fail on it
    regardless.
    """
    if not value:
        return None
    v = value.strip().lower()
    if v in canonical:
        return v
    return synonyms.get(v)


def normalize_values(values: list | None, canonical: frozenset, synonyms: dict) -> list:
    """List variant of ``normalize_value``; silently drops unrecognized entries."""
    if not values:
        return []
    out: list[str] = []
    for value in values:
        normalized = normalize_value(value, canonical, synonyms)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def normalize_metadata_vocab(metadata: "FashionMetadata") -> "FashionMetadata":
    """Snap the hard-filterable fields of ``metadata`` to canonical vocabulary
    in place, dropping anything that isn't recognized. Returns ``metadata``
    for convenient chaining.

    Only touches: garment/accessory category + colours, scene.location/
    environment/activity, and person.gender. Everything else (material,
    fit, length, neckline, subcategory, patterns, outfit styles/occasions)
    is left exactly as parsed, since those fields are not hard-filtered.
    """
    for garment in metadata.garments:
        garment.category = normalize_value(garment.category, GARMENT_CATEGORIES, GARMENT_CATEGORY_SYNONYMS)
        garment.colors = normalize_values(garment.colors, COLORS, COLOR_SYNONYMS)

    for accessory in metadata.accessories:
        accessory.category = normalize_value(accessory.category, ACCESSORY_CATEGORIES, ACCESSORY_CATEGORY_SYNONYMS)
        accessory.colors = normalize_values(accessory.colors, COLORS, COLOR_SYNONYMS)

    metadata.scene.location = normalize_value(metadata.scene.location, SCENE_LOCATIONS, {})
    metadata.scene.environment = normalize_value(metadata.scene.environment, SCENE_ENVIRONMENTS, {})
    metadata.scene.activity = normalize_value(metadata.scene.activity, SCENE_ACTIVITIES, {})
    metadata.person.gender = normalize_value(metadata.person.gender, GENDERS, {})

    return metadata
