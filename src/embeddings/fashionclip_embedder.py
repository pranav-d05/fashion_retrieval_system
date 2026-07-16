"""
FashionCLIP image and text encoder.

Uses ``transformers.CLIPModel`` and ``CLIPProcessor`` to load
``patrickjohncyh/fashion-clip`` — the correct API for this HuggingFace model.
Produces L2-normalised float32 embeddings of dimension 512.

Note: ``patrickjohncyh/fashion-clip`` is a standard HuggingFace transformers
CLIP model and does NOT support the open_clip ``hf-hub:`` loading path.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils.config_loader import FashionCLIPConfig

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class FashionCLIPEmbedder:
    """Encodes images and texts using FashionCLIP (patrickjohncyh/fashion-clip).

    Uses ``transformers.CLIPModel`` + ``CLIPProcessor``.

    Args:
        config: ``FashionCLIPConfig`` loaded from ``models.yaml``.
    """

    def __init__(self, config: FashionCLIPConfig) -> None:
        self._device = _resolve_device(config.device)
        logger.info(
            "Loading FashionCLIP '%s' on device='%s'…",
            config.model_name,
            self._device,
        )

        self._model = CLIPModel.from_pretrained(
            config.model_name,
            low_cpu_mem_usage=True,
        ).to(self._device)
        self._processor = CLIPProcessor.from_pretrained(config.model_name)
        self._model.eval()

        logger.info("FashionCLIP loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images into FashionCLIP embeddings.

        Args:
            images: List of PIL ``Image`` objects (any size / mode).

        Returns:
            Float32 numpy array of shape ``(N, 512)``, L2-normalised.
        """
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        inputs = self._processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self._device)

        with torch.no_grad():
            vision_outputs = self._model.vision_model(pixel_values=pixel_values)
            features = self._model.visual_projection(vision_outputs.pooler_output)
            features = _l2_normalize(features)

        return features.cpu().float().numpy()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into FashionCLIP embeddings.

        Args:
            texts: List of text prompts.

        Returns:
            Float32 numpy array of shape ``(N, 512)``, L2-normalised.
        """
        if not texts:
            return np.empty((0, 512), dtype=np.float32)

        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True,
            truncation=True, max_length=77,  # CLIP hard limit
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)

        with torch.no_grad():
            text_outputs = self._model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            features = self._model.text_projection(text_outputs.pooler_output)
            features = _l2_normalize(features)

        return features.cpu().float().numpy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised tensor (row-wise)."""
    return torch.nn.functional.normalize(tensor, p=2, dim=-1)
