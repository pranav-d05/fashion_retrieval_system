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
    "Describe the fashion image concisely but richly, covering: "
    "the garments worn (type, color, pattern, material, fit), "
    "the overall style, the occasion it suits, "
    "and any notable accessories. "
    "Write 2-4 sentences of fluent prose. "
    "Do not describe the background or non-fashion elements."
)

_CAPTION_USER_PROMPT = (
    "Please provide a detailed fashion description of this image."
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
