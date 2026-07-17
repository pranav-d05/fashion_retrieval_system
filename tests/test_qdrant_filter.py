"""
Unit tests for ``QdrantStore.build_metadata_filter``.

These test the pure filter-construction logic (no live Qdrant connection
required). They guard against two historical regressions:

1. Compositional queries only constraining the first garment's category.
2. Every parsed condition being AND-ed together (``must``), which meant a
   single wrong/unreliable field (material, fit, subcategory, ...) zeroed
   out the whole result set. The filter now only considers the reliable
   fields (category, colour, scene, gender) and combines them with
   ``min_should`` so most — not necessarily all — need to match.
"""

from __future__ import annotations

from qdrant_client.http import models as qmodels

from src.qdrant_store import QdrantStore
from src.schemas import Accessory, FashionMetadata, Garment, PersonInfo, SceneInfo


def _conditions(filt: qmodels.Filter) -> list:
    """Pull the condition list out of the min_should wrapper."""
    assert filt.must is None, "conditions should be combined with min_should, not must"
    assert filt.min_should is not None
    return filt.min_should.conditions


def _nested_conditions(filt: qmodels.Filter) -> list[qmodels.NestedCondition]:
    return [c for c in _conditions(filt) if isinstance(c, qmodels.NestedCondition)]


def _extract_keys(must_list: list[qmodels.FieldCondition]) -> dict:
    res = {}
    for c in must_list:
        if isinstance(c.match, qmodels.MatchValue):
            res[c.key] = c.match.value
        elif isinstance(c.match, qmodels.MatchText):
            res[c.key] = c.match.text
        elif hasattr(c.match, "any") and c.match.any is not None:
            res[c.key] = c.match.any
    return res


class TestBuildMetadataFilter:
    def test_existing_image_ids_returns_only_ids_in_payload(self):
        class FakeClient:
            def retrieve(self, **kwargs):
                assert kwargs["with_vectors"] is False
                return [
                    type("Point", (), {"payload": {"image_id": "IMG_abc"}})(),
                    type("Point", (), {"payload": {}})(),
                ]

        store = object.__new__(QdrantStore)
        store._collection = "fashion_images"
        store._client = FakeClient()

        assert store.existing_image_ids(["IMG_abc", "IMG_def"]) == {"IMG_abc"}

    def test_empty_metadata_returns_none(self):
        assert QdrantStore.build_metadata_filter(FashionMetadata()) is None

    def test_single_garment_all_fields(self):
        metadata = FashionMetadata(
            garments=[Garment(category="raincoat", colors=["yellow"])]
        )
        filt = QdrantStore.build_metadata_filter(metadata)
        nested = _nested_conditions(filt)
        assert len(nested) == 1

        inner_keys = _extract_keys(nested[0].nested.filter.must)
        assert inner_keys == {"category": "raincoat", "colors": ["yellow"]}

    def test_compositional_query_produces_one_nested_condition_per_garment(self):
        """'A red tie and a white shirt' must constrain BOTH garments,
        not just the first one's category (the original bug)."""
        metadata = FashionMetadata(
            garments=[
                Garment(category="tie", colors=["red"]),
                Garment(category="shirt", colors=["white"]),
            ]
        )
        filt = QdrantStore.build_metadata_filter(metadata)
        nested = _nested_conditions(filt)
        assert len(nested) == 2

        garment_constraints = [
            _extract_keys(n.nested.filter.must) for n in nested
        ]
        assert {"category": "tie", "colors": ["red"]} in garment_constraints
        assert {"category": "shirt", "colors": ["white"]} in garment_constraints

    def test_garment_with_no_fields_is_skipped(self):
        metadata = FashionMetadata(garments=[Garment()])
        assert QdrantStore.build_metadata_filter(metadata) is None

    def test_partial_garment_only_includes_present_fields(self):
        metadata = FashionMetadata(garments=[Garment(colors=["blue"])])
        filt = QdrantStore.build_metadata_filter(metadata)
        nested = _nested_conditions(filt)
        inner_keys = _extract_keys(nested[0].nested.filter.must)
        assert inner_keys == {"colors": ["blue"]}

    def test_unreliable_fields_are_never_filtered(self):
        """material/fit/length/neckline/subcategory/patterns/styles/
        occasions/num_people must never reach the Qdrant filter, even if
        present on the parsed metadata (e.g. from a caller that skipped
        normalize_metadata_vocab)."""
        metadata = FashionMetadata(
            garments=[
                Garment(
                    category="outerwear",
                    subcategory="hooded coat",
                    colors=["black"],
                    patterns=["plain"],
                    material="wool",
                    fit="oversized",
                    length="knee-length",
                    neckline="collar",
                )
            ],
            person=PersonInfo(gender="woman", num_people=2),
        )
        filt = QdrantStore.build_metadata_filter(metadata)
        nested = _nested_conditions(filt)
        inner_keys = _extract_keys(nested[0].nested.filter.must)
        assert inner_keys == {"category": "outerwear", "colors": ["black"]}

        field_conditions = [
            c for c in _conditions(filt) if isinstance(c, qmodels.FieldCondition)
        ]
        keys_present = {c.key for c in field_conditions}
        assert keys_present == {"person.gender"}  # num_people is not filtered

    def test_accessory_only_uses_category_and_colors(self):
        metadata = FashionMetadata(
            accessories=[Accessory(category="bag", subcategory="tote bag", colors=["brown"])]
        )
        filt = QdrantStore.build_metadata_filter(metadata)
        nested = _nested_conditions(filt)
        inner_keys = _extract_keys(nested[0].nested.filter.must)
        assert inner_keys == {"category": "bag", "colors": ["brown"]}

    def test_scene_and_gender_conditions_included(self):
        metadata = FashionMetadata(
            scene=SceneInfo(location="indoors", environment="office", activity="posing"),
            person=PersonInfo(gender="woman", num_people=1),
        )
        filt = QdrantStore.build_metadata_filter(metadata)
        field_conditions = [
            c for c in _conditions(filt) if isinstance(c, qmodels.FieldCondition)
        ]
        keys = _extract_keys(field_conditions)
        assert keys == {
            "scene.location": "indoors",
            "scene.environment": "office",
            "scene.activity": "posing",
            "person.gender": "woman",
        }

    def test_values_are_lowercased(self):
        metadata = FashionMetadata(garments=[Garment(category="Shirt", colors=["Red"])])
        filt = QdrantStore.build_metadata_filter(metadata)
        nested = _nested_conditions(filt)

        inner_values = set()
        for c in nested[0].nested.filter.must:
            if isinstance(c.match, qmodels.MatchValue):
                inner_values.add(c.match.value)
            elif hasattr(c.match, "any") and c.match.any is not None:
                inner_values.update(c.match.any)

        assert inner_values == {"shirt", "red"}

    def test_min_count_requires_all_when_two_or_fewer_conditions(self):
        metadata = FashionMetadata(
            garments=[Garment(category="dress", colors=["red"])]
        )
        filt = QdrantStore.build_metadata_filter(metadata)
        # one nested condition (garment) -> min_count == 1 == total conditions
        assert filt.min_should.min_count == len(filt.min_should.conditions) == 1

    def test_min_count_allows_one_miss_with_three_or_more_conditions(self):
        metadata = FashionMetadata(
            garments=[Garment(category="dress", colors=["red"])],
            accessories=[Accessory(category="bag", colors=["black"])],
            scene=SceneInfo(location="outdoors"),
        )
        filt = QdrantStore.build_metadata_filter(metadata)
        total = len(filt.min_should.conditions)
        assert total == 3
        assert filt.min_should.min_count == total - 1

    def test_flat_payload_round_trips_to_metadata(self):
        payload = {
            "image_id": "IMG_001",
            "image_path": "/img.jpg",
            "caption": "Test caption",
            "garments": [{"category": "dress", "colors": ["black"]}],
            "scene": {"environment": "urban"},
            "person": {"gender": "woman", "num_people": 1},
        }
        point = qmodels.ScoredPoint(id=1, version=1, score=0.9, payload=payload)
        result = QdrantStore.scored_point_to_result(point)
        assert result.image_id == "IMG_001"
        assert result.metadata.garments[0].category == "dress"
        assert result.metadata.scene.environment == "urban"
        assert result.metadata.person.num_people == 1
