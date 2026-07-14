"""
build_index — Offline indexing entry point.

Usage:
    uv run build-index --image-dir ./data/images

This script:
  1. Loads configuration.
  2. Instantiates all components via dependency injection.
  3. Calls Indexer.index_directory() to process every image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build-index",
        description="Offline fashion image indexing pipeline.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Root directory containing fashion images to index.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip images already indexed in Qdrant (not yet implemented).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # ----------------------------------------------------------------
    # Imports are deferred so that --help is always fast
    # ----------------------------------------------------------------
    from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
    from src.embeddings.text_embedder import TextEmbedder
    from src.indexing.indexer import Indexer
    from src.qdrant_store import QdrantStore
    from src.utils.config_loader import get_app_settings, get_model_settings
    from src.utils.logging_config import setup_logging
    from src.vlm.caption_generator import CaptionGenerator
    from src.vlm.metadata_extractor import MetadataExtractor
    from src.vlm.vlm_backend import VLMBackend

    # ---- Setup ----
    app_cfg = get_app_settings()
    model_cfg = get_model_settings()
    setup_logging(level=app_cfg.logging.level, fmt=app_cfg.logging.format)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting offline indexing pipeline.")
    logger.info("Image directory: %s", args.image_dir.resolve())

    # ---- Instantiate components ----
    # VLM shared backend (loaded once)
    vlm_backend = VLMBackend(model_cfg.vision_language_model)
    caption_gen = CaptionGenerator(vlm_backend)
    metadata_ext = MetadataExtractor(vlm_backend)

    # Embedding models
    clip_embedder = FashionCLIPEmbedder(model_cfg.fashionclip)
    text_embedder = TextEmbedder(model_cfg.text_embedding)

    # Qdrant
    store = QdrantStore(app_cfg)

    # Indexer
    indexer = Indexer(
        settings=app_cfg,
        caption_gen=caption_gen,
        metadata_ext=metadata_ext,
        clip_embedder=clip_embedder,
        text_embedder=text_embedder,
        qdrant_store=store,
    )

    # ---- Run ----
    try:
        indexer.index_directory(args.image_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Indexing interrupted by user.")
        sys.exit(130)

    logger.info("Indexing pipeline finished successfully.")


if __name__ == "__main__":
    main()
