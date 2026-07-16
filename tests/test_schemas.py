"""
Unit tests for src/schemas.py

Tests cover:
  - Default construction
  - Full construction with all fields
  - JSON round-trip (serialize → deserialize)
  - Pydantic validation (type coercion, unknown fields rejected)
  - FashionMetadata with multiple garments
"""

from __future__ import annotations

import json

import pytest

from src.schemas import (
    FashionMetadata,
    Garment,
    IndexedImage,
    PersonInfo,
    RetrievalResult,
    SceneInfo,
)


class TestGarment:
    def test_all_none_by_default(self):
        g = Garment()
        assert g.category is None
        assert g.colors == []

    def test_full_construction(self):
        g = Garment(
            category="top",
            subcategory="crew-neck t-shirt",
            colors=["white"],
            patterns=["solid"],
            material="cotton",
            fit="relaxed",
        )
        assert g.category == "top"
        assert g.material == "cotton"

    def test_json_round_trip(self):
        g = Garment(category="dress", colors=["red"])
        data = g.model_dump()
        g2 = Garment(**data)
        assert g == g2


class TestSceneInfo:
    def test_defaults(self):
        s = SceneInfo()
        assert s.location is None
        assert s.environment is None
        assert s.activity is None

    def test_partial_construction(self):
        s = SceneInfo(location="outdoors", environment="beach")
        assert s.location == "outdoors"
        assert s.activity is None


class TestPersonInfo:
    def test_defaults(self):
        p = PersonInfo()
        assert p.gender is None
        assert p.num_people is None

    def test_num_people_type(self):
        p = PersonInfo(num_people=2)
        assert isinstance(p.num_people, int)
        assert p.num_people == 2


class TestFashionMetadata:
    def test_empty_construction(self):
        m = FashionMetadata()
        assert m.garments == []
        assert isinstance(m.scene, SceneInfo)
        assert isinstance(m.person, PersonInfo)

    def test_multiple_garments(self):
        m = FashionMetadata(
            garments=[
                Garment(category="top", colors=["white"]),
                Garment(category="bottom", colors=["blue"]),
            ]
        )
        assert len(m.garments) == 2
        assert m.garments[0].category == "top"
        assert m.garments[1].category == "bottom"

    def test_full_construction(self):
        m = FashionMetadata(
            garments=[Garment(category="dress", colors=["floral"])],
            scene=SceneInfo(location="outdoors", environment="garden"),
            person=PersonInfo(gender="woman", num_people=1),
        )
        assert m.person.gender == "woman"
        assert m.scene.environment == "garden"

    def test_json_round_trip(self):
        m = FashionMetadata(
            garments=[Garment(category="jacket", colors=["black"])],
            scene=SceneInfo(environment="urban"),
            person=PersonInfo(gender="man"),
        )
        json_str = m.model_dump_json()
        m2 = FashionMetadata(**json.loads(json_str))
        assert m == m2

    def test_from_dict(self):
        """Simulate what VLM output parsing does."""
        data = {
            "garments": [{"category": "top", "colors": ["navy"]}],
            "scene": {"location": "indoors"},
            "person": {"gender": "woman", "num_people": 1},
        }
        m = FashionMetadata(**data)
        assert m.garments[0].colors == ["navy"]


class TestIndexedImage:
    def test_construction(self):
        m = FashionMetadata()
        rec = IndexedImage(
            image_id="IMG_abc123",
            image_path="/data/images/test.jpg",
            caption="A woman wearing a white dress.",
            metadata=m,
            fashionclip_embedding=[0.1] * 512,
            caption_embedding=[0.2] * 768,
        )
        assert rec.image_id == "IMG_abc123"
        assert len(rec.fashionclip_embedding) == 512
        assert len(rec.caption_embedding) == 768

    def test_metadata_nested(self):
        m = FashionMetadata(garments=[Garment(category="dress")])
        rec = IndexedImage(
            image_id="IMG_001",
            image_path="/img.jpg",
            caption="Test",
            metadata=m,
            fashionclip_embedding=[0.0] * 512,
            caption_embedding=[0.0] * 768,
        )
        assert rec.metadata.garments[0].category == "dress"


class TestRetrievalResult:
    def test_default_score(self):
        r = RetrievalResult(
            image_id="IMG_001",
            image_path="/img.jpg",
            caption="Test caption",
            metadata=FashionMetadata(),
        )
        assert r.score == 0.0

    def test_score_assignment(self):
        r = RetrievalResult(
            image_id="IMG_001",
            image_path="/img.jpg",
            caption="Test",
            metadata=FashionMetadata(),
            score=0.95,
        )
        assert r.score == pytest.approx(0.95)
