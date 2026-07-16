"""
Unit tests for the offline metadata extractor.

These tests verify that the extractor uses the text-only Qwen2.5-1.5B
workflow and parses the schema-constrained JSON output into FashionMetadata.
"""

from __future__ import annotations

import torch

from src.schemas import FashionMetadata
from src.utils.config_loader import QueryParserConfig
from src.vlm.metadata_extractor import MetadataExtractor


class _FakeTokenizer:
    eos_token_id = 0
    eos_token = "<|endoftext|>"
    pad_token = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "prompt"

    def __call__(self, text, return_tensors="pt", padding=False):
        return _FakeBatch()

    def decode(self, tokens, skip_special_tokens=True):
        return (
            '{"garments":[{"category":"dress","colors":["black"]}],'
            '"scene":{"environment":"urban"},'
            '"person":{"gender":"woman","num_people":1}}'
        )

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [self.decode(t, skip_special_tokens) for t in tokens]


class _FakeBatch(dict):
    def __init__(self):
        super().__init__({"input_ids": torch.tensor([[1, 2]])})

    def to(self, device):
        return self


class _FakeModel:
    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3, 4]])


def test_metadata_extractor_uses_text_only_model(monkeypatch):
    monkeypatch.setattr(
        "src.vlm.metadata_extractor.AutoTokenizer.from_pretrained",
        lambda model_name: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "src.vlm.metadata_extractor.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: _FakeModel(),
    )

    extractor = MetadataExtractor(
        QueryParserConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens=32)
    )
    metadata = extractor.extract("A woman wearing a black dress in an urban setting.")

    assert isinstance(metadata, FashionMetadata)
    assert metadata.garments[0].category == "dress"
    assert metadata.garments[0].colors == ["black"]
    assert metadata.scene.environment == "urban"
    assert metadata.person.gender == "woman"
