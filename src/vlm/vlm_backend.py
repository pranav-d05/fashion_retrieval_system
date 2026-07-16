"""
VLM model loader — loads Qwen2.5-VL-3B-Instruct once and exposes
a shared (model, processor) pair to CaptionGenerator and MetadataExtractor.

Keeping a single loader prevents the large VLM from being loaded twice.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.utils.config_loader import VLMConfig

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available torch device.

    Deliberately never returns a disk-offload / accelerate 'auto' device_map —
    on small-VRAM GPUs that silently pages weights to disk, which is
    catastrophically slow (effectively hangs) rather than erroring out.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class VLMBackend:
    """Loads and owns the Qwen2.5-VL model and processor.

    Intended to be instantiated once and injected into both
    ``CaptionGenerator`` and ``MetadataExtractor``.

    On CUDA, the model is loaded 4-bit quantized (via bitsandbytes) so a
    ~3B-parameter model fits comfortably on small-VRAM GPUs (e.g. 4GB laptop
    GPUs) without falling back to disk offloading. On CPU, it loads in
    float32 with no device_map, so it either fits in RAM or fails with a
    clear OOM instead of silently crawling from disk.
    """

    def __init__(self, config: VLMConfig) -> None:
        device = _resolve_device(config.device)
        logger.info(
            "Loading VLM '%s' on device='%s' — this may take a minute…",
            config.model_name,
            device,
        )

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
                logger.info("Loading VLM 4-bit quantized (bitsandbytes) on CUDA.")
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed — loading VLM in fp16 on CUDA "
                    "without quantization. Install bitsandbytes if this OOMs "
                    "or falls back to disk offload on small-VRAM GPUs."
                )
                load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            **load_kwargs,
        )
        if "device_map" not in load_kwargs:
            self.model = self.model.to(device)
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
