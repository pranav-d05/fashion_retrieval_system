"""
Query Parser — converts a free-text user query into structured FashionMetadata.

Uses a lightweight instruction-tuned LLM (Qwen2.5-1.5B-Instruct) with a
JSON-constrained prompt. Falls back to empty metadata on any parse failure
so that the retrieval pipeline degrades gracefully to pure vector search.
"""

from __future__ import annotations

import json
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schemas import FashionMetadata
from src.utils.config_loader import QueryParserConfig

logger = logging.getLogger(__name__)

# Inline schema description (text only — no image)
_SCHEMA_DESCRIPTION = """
{
  "garments": [
    {
      "category": "string | null",
      "subcategory": "string | null",
      "color": "string | null",
      "pattern": "string | null",
      "material": "string | null",
      "fit": "string | null",
      "style": "string | null",
      "occasion": "string | null"
    }
  ],
  "scene": {
    "location": "string | null",
    "environment": "string | null",
    "activity": "string | null"
  },
  "person": {
    "gender": "string | null",
    "num_people": "integer | null"
  }
}
""".strip()

_SYSTEM_PROMPT = (
    "You are a structured data extraction system for a fashion search engine. "
    "Given a user's natural language search query, extract structured fashion "
    "attributes and return ONLY a valid JSON object matching the schema below. "
    "Use null for attributes not mentioned in the query. "
    "Return ONLY the JSON object — no explanations, no markdown, no extra text.\n\n"
    f"Schema:\n{_SCHEMA_DESCRIPTION}"
)


class QueryParser:
    """Parses a free-text query into structured ``FashionMetadata``.

    Args:
        config: ``QueryParserConfig`` from ``models.yaml``.
    """

    def __init__(self, config: QueryParserConfig) -> None:
        device_map: str = config.device
        logger.info(
            "Loading QueryParser '%s' (device_map=%s)…",
            config.model_name,
            device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype="auto",
            device_map=device_map,
        )
        self._model.eval()
        self._max_new_tokens = config.max_new_tokens
        logger.info("QueryParser loaded successfully.")

    @property
    def _device(self) -> torch.device:
        return next(self._model.parameters()).device

    def parse(self, query: str) -> FashionMetadata:
        """Convert a natural language query to structured ``FashionMetadata``.

        Args:
            query: User's search query string.

        Returns:
            ``FashionMetadata`` with extracted attributes.
            Returns empty metadata if parsing fails.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Strip the prompt tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        logger.debug("QueryParser raw output: %.150s", raw)
        return _parse_metadata(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(raw: str) -> FashionMetadata:
    """Parse the model's raw string output into a FashionMetadata object."""
    cleaned = _strip_code_fences(raw)
    try:
        data = json.loads(cleaned)
        return FashionMetadata(**data)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return FashionMetadata(**data)
            except (json.JSONDecodeError, ValueError):
                pass
    logger.warning("QueryParser could not parse output — using empty metadata.")
    return FashionMetadata()


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
