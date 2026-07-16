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
import os
import sys
from pathlib import Path


def _configure_hf_cache() -> None:
    # Honour any HF_HOME already set in the environment / .env file;
    # only fall back to D:\hf_cache if nothing is configured.
    hf_home = os.environ.get("HF_HOME", r"D:\hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))


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
        help="Skip images already indexed in Qdrant.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _configure_hf_cache()
    args = _parse_args(argv)

    # ----------------------------------------------------------------
    # Imports are deferred so that --help is always fast
    # ----------------------------------------------------------------
    from src.indexing.staged_indexer import StagedIndexer
    from src.qdrant_store import QdrantStore
    from src.utils.config_loader import get_app_settings, get_model_settings
    from src.utils.logging_config import setup_logging

    # ---- Setup ----
    app_cfg = get_app_settings()
    model_cfg = get_model_settings()
    setup_logging(level=app_cfg.logging.level, fmt=app_cfg.logging.format)

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting offline indexing pipeline.")
    logger.info("Image directory: %s", args.image_dir.resolve())

    # Only Qdrant is opened here. StagedIndexer loads and releases each model
    # family immediately around the stage that uses it.
    store = QdrantStore(app_cfg)
    indexer = StagedIndexer(
        settings=app_cfg,
        model_settings=model_cfg,
        store=store,
        staging_dir=Path("data/.index_staging"),
    )

    # ---- Run ----
    try:
        indexer.index_directory(args.image_dir, skip_existing=args.skip_existing)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Indexing interrupted by user.")
        sys.exit(130)

    logger.info("Indexing pipeline finished successfully.")


if __name__ == "__main__":
    main()
