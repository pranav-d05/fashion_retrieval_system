"""
Caption generator — uses Qwen2.5-VL to produce rich natural language
descriptions of fashion images.
"""

from __future__ import annotations

import logging

from PIL import Image

from src.vlm.vlm_backend import VLMBackend

logger = logging.getLogger(__name__)

_CAPTION_SYSTEM_PROMPT = (
    "You are a fashion expert and visual analyst. "
    "Generate a factual description of the clothing and context visible in the image. "
    "Write 3-5 concise sentences using only what is directly observable or clearly inferrable from visual context.\n\n"
    "COVER ALL THAT ARE VISIBLE:\n"
    "- Garments: category (top/bottom/dress/outerwear/skirt/jumpsuit), color, pattern, "
    "material if visually obvious, fit, length, neckline, sleeve style\n"
    "- Footwear and accessories (bags, hats, jewellery, belts, sunglasses, scarves)\n"
    "- People: gender presentation (woman/man/person) and number of people visible\n"
    "- Outfit style: e.g. casual, formal, streetwear, sporty, bohemian, business, elegant, vintage — "
    "if clearly inferrable from the overall look\n"
    "- Occasion: e.g. workwear, party, everyday, outdoor, beach, formal event — "
    "only when the setting or outfit strongly implies it\n"
    "- Setting: e.g. office, urban street, park, home, studio, runway, beach — if visible in the scene\n\n"
    "Do NOT mention: camera angles, lighting quality, or irrelevant background details. "
    "Do NOT guess brand, fabric composition, age, or ethnicity. "
    "Focus on attributes that would help someone find this image by natural language search."
)

_CAPTION_USER_PROMPT = (
    "Describe the clothing, accessories, people, style, occasion context, and setting visible in this fashion image."
)


class CaptionGenerator:
    """Generates natural language captions for fashion images.

    Shares a ``VLMBackend`` instance with ``MetadataExtractor`` so the
    large VLM is loaded only once.

    Args:
        backend: Pre-loaded ``VLMBackend`` instance.
    """

    def __init__(self, backend: VLMBackend) -> None:
        self._backend = backend

    def generate_caption(self, image: Image.Image) -> str:
        """Generate a fashion-focused caption for a PIL image.

        Args:
            image: A PIL ``Image`` object (any mode/size).

        Returns:
            A rich natural language description of the fashion items.
        """
        messages = [
            {
                "role": "system",
                "content": _CAPTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _CAPTION_USER_PROMPT},
                ],
            },
        ]
        caption = self._backend.generate(messages)
        logger.debug("Generated caption (%d chars): %.80s…", len(caption), caption)
        return caption
