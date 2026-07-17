"""
Query Parser — converts a free-text user query into structured FashionMetadata.

Uses a lightweight instruction-tuned LLM (Qwen2.5-1.5B-Instruct) with a
JSON-constrained prompt. Falls back to empty metadata on any parse failure
so that the retrieval pipeline degrades gracefully to pure vector search.

The schema requested here is deliberately smaller than the offline
metadata-extraction schema (see vlm/metadata_extractor.py): it only asks
for the fields that ``QdrantStore.build_metadata_filter`` actually hard-
filters on (garment/accessory category + colour, scene, gender). Material,
fit, length, neckline, subcategory, and inferred style/occasion are
open-vocabulary and are never hard-filtered, so asking the parser to guess
at them for every query only adds hallucination surface and latency for
no downstream benefit.
"""

from __future__ import annotations

import json
import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schemas import FashionMetadata
from src.utils.config_loader import QueryParserConfig
from src.vocab import normalize_metadata_vocab

logger = logging.getLogger(__name__)

# Schema description injected into the LLM system prompt. Trimmed to the
# fields that are actually hard-filtered — see module docstring.
_SCHEMA_DESCRIPTION = """
{
  "garments": [
    {
      "category": "string | null",
      "colors": ["string"]
    }
  ],
  "accessories": [
    {
      "category": "string | null",
      "colors": ["string"]
    }
  ],
  "scene": {
    "location": "string | null",
    "environment": "string | null",
    "activity": "string | null"
  },
  "person": {
    "gender": "string | null"
  }
}
""".strip()

_SYSTEM_PROMPT = (
    "You are a structured data extraction system for a fashion search engine. "
    "Given a user's natural language search query, extract structured fashion "
    "attributes and return ONLY a valid JSON object matching the schema below.\n\n"
    "STRICT RULES:\n"
    "1. Use null for single-value string fields not mentioned or not certain. "
    "   Use [] (empty array) for list fields not mentioned.\n"
    "2. colors MUST always be a JSON array, e.g. [\"navy\", \"white\"]. If no colour "
    "   is mentioned, use [].\n"
    "3. garments = clothing items (tops, bottoms, dresses, outerwear, skirts, jumpsuits). "
    "   accessories = non-clothing items (bags, shoes, jewellery, hats, belts, watches, "
    "   sunglasses, scarves, ties).\n"
    "4. A garment feature (hood, collar, sleeve, pocket, zip) is NOT a separate accessory. "
    "   'hooded coat' is ONE garment (category=outerwear) — do not also emit a "
    "   hat/headwear accessory for the hood.\n"
    "5. category and colors MUST use only the canonical values listed below. If the "
    "   query implies something close but not listed, pick the closest canonical value "
    "   rather than inventing a new one. If nothing fits, use null.\n"
    "6. Return ONLY the JSON object — no markdown, no explanation, no extra text.\n\n"
    "CANONICAL VOCABULARY — use these exact values, nothing else:\n"
    "- garment category: \"top\", \"bottom\", \"dress\", \"outerwear\", \"skirt\", \"jumpsuit\"\n"
    "- accessory category: \"bag\", \"shoes\", \"hat\", \"jewellery\", \"belt\", \"watch\", "
    "\"sunglasses\", \"scarf\", \"neckwear\"\n"
    "- colour: \"black\", \"white\", \"grey\", \"red\", \"orange\", \"yellow\", \"green\", "
    "\"blue\", \"navy\", \"purple\", \"pink\", \"brown\", \"beige\", \"tan\", \"cream\", "
    "\"gold\", \"silver\", \"multicolor\"\n"
    "- scene location: \"indoors\", \"outdoors\", \"studio\"\n"
    "- scene environment: \"office\", \"street\", \"park\", \"home\", \"beach\", \"runway\", "
    "\"mall\", \"urban\"\n"
    "- scene activity: \"walking\", \"posing\", \"sitting\", \"standing\"\n"
    "- person gender: \"woman\", \"man\", \"person\" (use \"person\" when ambiguous)\n\n"
    "EXAMPLES:\n"
    "Query: black hooded coat\n"
    "{\"garments\": [{\"category\": \"outerwear\", \"colors\": [\"black\"]}], "
    "\"accessories\": [], \"scene\": {\"location\": null, \"environment\": null, "
    "\"activity\": null}, \"person\": {\"gender\": null}}\n\n"
    "Query: woman in a red dress walking on the street\n"
    "{\"garments\": [{\"category\": \"dress\", \"colors\": [\"red\"]}], "
    "\"accessories\": [], \"scene\": {\"location\": \"outdoors\", "
    "\"environment\": \"street\", \"activity\": \"walking\"}, "
    "\"person\": {\"gender\": \"woman\"}}\n\n"
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
            ``FashionMetadata`` with extracted attributes. Category/colour/
            scene/gender values are validated against the canonical
            vocabulary (src/vocab.py) before being returned, so anything
            the model hallucinated outside that vocabulary is dropped
            rather than passed through to the Qdrant filter.
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
        metadata = normalize_metadata_vocab(metadata)
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

    These only touch fields that are actually hard-filtered (scene.*).
    Outfit styles/occasions are no longer part of the query-time schema
    and are never filtered on (see src/vocab.py + qdrant_store.py), so
    they're not set here either — doing so would be dead code.
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

    if "walk" in normalized_query and not metadata.scene.activity:
        metadata.scene.activity = "walking"

    return metadata
