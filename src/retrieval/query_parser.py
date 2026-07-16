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

# Schema description injected into the LLM system prompt
_SCHEMA_DESCRIPTION = """
{
  "garments": [
    {
      "category": "string | null",
      "subcategory": "string | null",
      "colors": ["string"],
      "patterns": ["string"],
      "material": "string | null",
      "fit": "string | null",
      "length": "string | null",
      "neckline": "string | null"
    }
  ],
  "accessories": [
    {
      "category": "string | null",
      "subcategory": "string | null",
      "colors": ["string"]
    }
  ],
  "outfit": {
    "styles": ["string"],
    "occasions": ["string"]
  },
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
    "attributes and return ONLY a valid JSON object matching the schema below.\n\n"
    "STRICT RULES:\n"
    "1. Use null for single-value string fields not mentioned. "
    "   Use [] (empty array) for list fields not mentioned.\n"
    "2. colors and patterns MUST always be JSON arrays: e.g. [\"navy\", \"white\"].\n"
    "3. styles and occasions MUST always be JSON arrays: e.g. [\"casual\", \"streetwear\"].\n"
    "4. garments = clothing items (tops, bottoms, dresses, outerwear, skirts, jumpsuits). "
    "   accessories = non-clothing items (bags, shoes, jewellery, hats, belts, watches, sunglasses, scarves, ties).\n"
    "5. Return ONLY the JSON object — no markdown, no explanation, no extra text.\n\n"
    "CANONICAL VOCABULARY — use these exact values where applicable:\n"
    "- garment category: \"top\", \"bottom\", \"dress\", \"outerwear\", \"skirt\", \"jumpsuit\"\n"
    "- accessory category: \"bag\", \"shoes\", \"hat\", \"jewellery\", \"belt\", \"watch\", \"sunglasses\", \"scarf\", \"neckwear\"\n"
    "- outfit styles: \"casual\", \"formal\", \"streetwear\", \"sporty\", \"bohemian\", "
    "\"business\", \"elegant\", \"vintage\", \"chic\"\n"
    "- outfit occasions: \"everyday\", \"workwear\", \"party\", \"outdoor\", \"beach\", "
    "\"formal event\", \"casual outing\"\n"
    "- scene location: \"indoors\", \"outdoors\", \"studio\"\n"
    "- scene environment: \"office\", \"street\", \"park\", \"home\", \"beach\", \"runway\", \"mall\"\n"
    "- person gender: \"woman\", \"man\", \"person\" (use \"person\" when ambiguous)\n\n"
    f"Schema:\n{_SCHEMA_DESCRIPTION}"
)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device (never disk-offload)."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class QueryParser:
    """Parses a free-text query into structured ``FashionMetadata``.

    Loaded 4-bit quantized on CUDA (this model runs alongside FashionCLIP,
    BGE, and the cross-encoder simultaneously during online retrieval, so
    keeping its footprint small matters on small-VRAM GPUs).

    Args:
        config: ``QueryParserConfig`` from ``models.yaml``.
    """

    def __init__(self, config: QueryParserConfig) -> None:
        device = _resolve_device(config.device)
        logger.info("Loading QueryParser '%s' on device='%s'…", config.model_name, device)

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        load_kwargs: dict = {"low_cpu_mem_usage": True}
        if device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["device_map"] = {"": 0}
            except ImportError:
                logger.warning("bitsandbytes not installed — loading QueryParser in fp16 on CUDA.")
                load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
        if "device_map" not in load_kwargs:
            self._model = self._model.to(device)
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
        metadata = _parse_metadata(raw)
        return _normalize_metadata_for_query(query, metadata)


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


def _normalize_metadata_for_query(query: str, metadata: FashionMetadata) -> FashionMetadata:
    """Apply small deterministic fixes for common query phrases.

    The LLM parser is good at broad extraction but can produce values that are
    syntactically valid and still too strict for retrieval. This post-pass keeps
    common fashion search phrases aligned with the indexed metadata vocabulary.
    """
    normalized_query = query.lower()

    if "city walk" in normalized_query or "city stroll" in normalized_query or "walk in the city" in normalized_query:
        metadata.scene.location = "outdoors"
        metadata.scene.environment = "urban"
        metadata.scene.activity = "walking"

    if "city" in normalized_query and metadata.scene.location is None:
        metadata.scene.location = "outdoors"

    if "indoor" in normalized_query and metadata.scene.location == "indoor":
        metadata.scene.location = "indoors"

    if "weekend" in normalized_query and not metadata.outfit.occasions:
        metadata.outfit.occasions = ["casual outing"]

    if "casual" in normalized_query and "casual" not in metadata.outfit.styles:
        metadata.outfit.styles.append("casual")

    if "walk" in normalized_query and not metadata.scene.activity:
        metadata.scene.activity = "walking"

    return metadata
