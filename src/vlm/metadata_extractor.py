"""
Metadata extractor — uses Qwen2.5-VL to extract structured FashionMetadata
from an image, guided by its generated caption.

The VLM is prompted to output valid JSON that matches the FashionMetadata
Pydantic schema. A robust fallback returns empty metadata on any parse error.
"""

from __future__ import annotations

import json
import logging
import re

from PIL import Image

from src.schemas import FashionMetadata
from src.vlm.vlm_backend import VLMBackend

logger = logging.getLogger(__name__)

# JSON schema description injected into the VLM prompt
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

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a structured data extraction system. "
    "Your job is to analyze a fashion image and its caption, "
    "then return ONLY a valid JSON object that strictly follows the given schema. "
    "Use null for any field you cannot determine with confidence. "
    "Return ONLY the JSON object — no explanations, no markdown code blocks."
)


def _build_extraction_prompt(caption: str) -> str:
    return (
        f"Caption: {caption}\n\n"
        "Extract structured fashion metadata from this image and its caption.\n"
        f"Return a JSON object that matches this schema exactly:\n{_SCHEMA_DESCRIPTION}"
    )


class MetadataExtractor:
    """Extracts structured ``FashionMetadata`` from a fashion image.

    Shares a ``VLMBackend`` instance with ``CaptionGenerator`` so the
    large VLM is loaded only once.

    Args:
        backend: Pre-loaded ``VLMBackend`` instance.
    """

    def __init__(self, backend: VLMBackend) -> None:
        self._backend = backend

    def extract(self, image: Image.Image, caption: str) -> FashionMetadata:
        """Extract structured metadata from an image and its caption.

        Args:
            image:   A PIL ``Image`` object.
            caption: The VLM-generated caption for this image.

        Returns:
            A ``FashionMetadata`` instance. Returns a default (empty) instance
            if the VLM output cannot be parsed as valid JSON or schema.
        """
        messages = [
            {
                "role": "system",
                "content": _EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _build_extraction_prompt(caption)},
                ],
            },
        ]

        raw = self._backend.generate(messages, max_new_tokens=512)
        return _parse_metadata(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(raw: str) -> FashionMetadata:
    """Parse a raw VLM string into FashionMetadata with robust fallback."""
    # Strip markdown code fences if present
    cleaned = _strip_code_fences(raw)

    try:
        data = json.loads(cleaned)
        return FashionMetadata(**data)
    except (json.JSONDecodeError, ValueError) as exc:
        # Try to extract a JSON object using regex
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return FashionMetadata(**data)
            except (json.JSONDecodeError, ValueError):
                pass
        logger.warning("Failed to parse metadata JSON: %s. Using empty metadata.", exc)
        return FashionMetadata()


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```)."""
    text = text.strip()
    # Remove ```json or ``` at start
    text = re.sub(r"^```(?:json)?\s*", "", text)
    # Remove ``` at end
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
