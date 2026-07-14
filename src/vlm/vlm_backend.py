"""
VLM model loader — loads Qwen2.5-VL-3B-Instruct once and exposes
a shared (model, processor) pair to CaptionGenerator and MetadataExtractor.

Keeping a single loader prevents the large VLM from being loaded twice.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.utils.config_loader import VLMConfig

logger = logging.getLogger(__name__)


class VLMBackend:
    """Loads and owns the Qwen2.5-VL model and processor.

    Intended to be instantiated once and injected into both
    ``CaptionGenerator`` and ``MetadataExtractor``.
    """

    def __init__(self, config: VLMConfig) -> None:
        device_map: Any = config.device  # "auto", "cuda", "cpu", …
        logger.info(
            "Loading VLM '%s' (device_map=%s) — this may take a minute…",
            config.model_name,
            device_map,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(config.model_name)
        self.max_new_tokens = config.max_new_tokens

        logger.info("VLM '%s' loaded successfully.", config.model_name)

    @property
    def device(self) -> torch.device:
        """Return the primary device of the model."""
        return next(self.model.parameters()).device

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> str:
        """Run a single VLM inference pass.

        Args:
            messages: OpenAI-style chat messages list (supports image content).
            max_new_tokens: Override for this specific call.

        Returns:
            Decoded model output string (stripped).
        """
        n_tokens = max_new_tokens or self.max_new_tokens
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        from qwen_vl_utils import process_vision_info  # type: ignore

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=n_tokens)

        # Strip the prompt tokens from the output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs["input_ids"], generated_ids)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
