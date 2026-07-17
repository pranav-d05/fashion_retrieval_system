"""
Metadata extractor — uses a text-only Qwen2.5-1.5B model to extract
structured FashionMetadata from the generated image caption.

The extractor receives the VLM caption as input, then prompts the same
schema-constrained JSON interface used by the online query parser.
A robust fallback returns empty metadata on any parse error.
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

# JSON schema description injected into the VLM prompt
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

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a structured fashion metadata extraction system. "
    "Given a natural language caption describing a fashion image, extract "
    "structured attributes and return ONLY a valid JSON object matching the schema below.\n\n"
    "STRICT RULES:\n"
    "1. Use null for single-value string fields you cannot determine. "
    "   Use [] (empty array) for list fields you cannot determine.\n"
    "2. colors and patterns MUST always be JSON arrays of strings: e.g. [\"navy\", \"white\"].\n"
    "3. styles and occasions MUST always be JSON arrays: e.g. [\"casual\", \"streetwear\"].\n"
    "4. garments = clothing items (tops, bottoms, dresses, outerwear, skirts, jumpsuits). "
    "   accessories = non-clothing items (bags, shoes, jewellery, hats, belts, watches, sunglasses, scarves).\n"
    "5. A garment feature (hood, collar, sleeve, pocket, zip) is NOT a separate accessory. "
    "   'hooded coat' is ONE garment (category=outerwear) — do not also emit a "
    "   hat/headwear accessory for the hood.\n"
    "6. outfit.styles and outfit.occasions describe the WHOLE look, not individual garments.\n"
    "7. Return ONLY the JSON object — no markdown, no explanation, no extra text.\n\n"
    "CANONICAL VOCABULARY — use these exact values where applicable:\n"
    "- garment category: \"top\", \"bottom\", \"dress\", \"outerwear\", \"skirt\", \"jumpsuit\"\n"
    "- accessory category: \"bag\", \"shoes\", \"hat\", \"jewellery\", \"belt\", \"watch\", \"sunglasses\", \"scarf\", \"neckwear\"\n"
    "- colour: \"black\", \"white\", \"grey\", \"red\", \"orange\", \"yellow\", \"green\", "
    "\"blue\", \"navy\", \"purple\", \"pink\", \"brown\", \"beige\", \"tan\", \"cream\", "
    "\"gold\", \"silver\", \"multicolor\"\n"
    "- outfit styles: \"casual\", \"formal\", \"streetwear\", \"sporty\", \"bohemian\", "
    "\"business\", \"elegant\", \"vintage\", \"chic\"\n"
    "- outfit occasions: \"everyday\", \"workwear\", \"party\", \"outdoor\", \"beach\", "
    "\"formal event\", \"casual outing\"\n"
    "- scene location: \"indoors\", \"outdoors\", \"studio\"\n"
    "- scene environment: \"office\", \"street\", \"park\", \"home\", \"beach\", \"runway\", \"mall\", \"urban\"\n"
    "- person gender: \"woman\", \"man\", \"person\" (use \"person\" when ambiguous)\n\n"
    f"Schema:\n{_SCHEMA_DESCRIPTION}"
)


def _build_extraction_prompt(caption: str) -> str:
    return (
        "Caption:\n"
        f"{caption}\n\n"
        "Extract structured fashion metadata from the caption above and return ONLY JSON."
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


class MetadataExtractor:
    """Extracts structured ``FashionMetadata`` from a fashion image.

    Uses the same Qwen2.5-1.5B model as the online query parser so offline
    metadata extraction and online query parsing share the same structured
    schema and reasoning style. Loaded 4-bit quantized on CUDA to fit
    comfortably on small-VRAM GPUs.

    Args:
        config: ``QueryParserConfig`` from ``models.yaml``.
    """

    def __init__(self, config: QueryParserConfig) -> None:
        device = _resolve_device(config.device)
        logger.info("Loading MetadataExtractor '%s' on device='%s'…", config.model_name, device)

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict = {"low_cpu_mem_usage": True}
        if device == "cuda":
            # Disabled 4-bit quantization to drastically speed up generation.
            # 1.5B parameters in FP16 takes ~3GB VRAM (Fits in RTX 3050 4GB safely).
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self._model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
        self._model = self._model.to(device)
        self._model.eval()
        self._max_new_tokens = config.max_new_tokens
        logger.info("MetadataExtractor loaded successfully.")

    @property
    def _device(self) -> torch.device:
        return next(self._model.parameters()).device

    def extract(self, caption: str) -> FashionMetadata:
        """Extract structured metadata from a single caption."""
        return self.extract_batch([caption])[0]

    def extract_batch(self, captions: list[str]) -> list[FashionMetadata]:
        """Extract structured metadata from a batch of captions."""
        messages_list = [
            [
                {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_extraction_prompt(c)},
            ]
            for c in captions
        ]

        texts = [
            self._tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_list
        ]
        
        inputs = self._tokenizer(texts, return_tensors="pt", padding=True).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        raw_outputs = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        # Normalize category/colour/scene/gender to the canonical vocabulary
        # so stored payload values line up with what the (separately
        # normalized) online query parser produces — see src/vocab.py.
        return [normalize_metadata_vocab(_parse_metadata(raw)) for raw in raw_outputs]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(raw: str) -> FashionMetadata:
    """Parse a raw LLM string into FashionMetadata with robust fallback.

    Attempts in order:
    1. Direct JSON parse
    2. Regex-extract first JSON object
    3. Repair truncated JSON and re-parse
    4. Fall back to empty FashionMetadata
    """
    cleaned = _strip_code_fences(raw)

    # Attempt 1: direct parse
    try:
        return FashionMetadata(**json.loads(cleaned))
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: extract first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return FashionMetadata(**json.loads(match.group()))
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3: try to repair a truncated JSON object
    repaired = _repair_truncated_json(cleaned)
    if repaired:
        try:
            return FashionMetadata(**json.loads(repaired))
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse metadata from model output — using empty metadata.")
    return FashionMetadata()


def _repair_truncated_json(text: str) -> str | None:
    """Attempt to close an incomplete JSON object truncated by max_new_tokens.

    Finds the last complete key-value pair, strips the trailing incomplete
    portion, and closes all open braces / brackets so the result is valid JSON.
    Returns None if no reasonable repair is possible.
    """
    # Find the start of a JSON object
    start = text.find("{")
    if start == -1:
        return None

    fragment = text[start:].rstrip()

    # Remove trailing comma or incomplete token (e.g. `"color": "si`)
    fragment = re.sub(r',\s*$', '', fragment)
    fragment = re.sub(r',\s*"[^"]*$', '', fragment)  # trailing key with no value
    fragment = re.sub(r':\s*"[^"]*$', '', fragment)  # trailing value with no closing quote
    fragment = re.sub(r',\s*$', '', fragment)          # trailing comma after removal

    # Count open brackets/braces and close them
    open_braces = fragment.count('{') - fragment.count('}')
    open_brackets = fragment.count('[') - fragment.count(']')

    if open_braces < 0 or open_brackets < 0:
        return None  # malformed beyond repair

    fragment += ']' * open_brackets + '}' * open_braces
    return fragment


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
