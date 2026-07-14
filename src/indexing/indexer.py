"""
Offline Indexing Pipeline.

The ``Indexer`` orchestrates the complete offline pipeline:
  image → caption → metadata → embeddings → Qdrant upsert

Design principles:
  - Per-image error handling: a single failure never aborts the whole run.
  - Progress reporting via ``tqdm``.
  - Batch embedding for efficiency (FashionCLIP + BGE operate on batches).
  - VLM runs per-image (sequential) due to memory constraints.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
from src.embeddings.text_embedder import TextEmbedder
from src.indexing._image_loader import load_image
from src.qdrant_store import QdrantStore
from src.schemas import FashionMetadata, IndexedImage
from src.utils.config_loader import AppSettings
from src.utils.helpers import chunk_list, generate_image_id, iter_images
from src.vlm.caption_generator import CaptionGenerator
from src.vlm.metadata_extractor import MetadataExtractor

logger = logging.getLogger(__name__)


class Indexer:
    """Orchestrates the offline fashion image indexing pipeline.

    Args:
        settings:        Application settings (batch size, extensions, etc.).
        caption_gen:     VLM-backed caption generator.
        metadata_ext:    VLM-backed metadata extractor.
        clip_embedder:   FashionCLIP encoder for images.
        text_embedder:   BGE encoder for captions.
        qdrant_store:    Qdrant client wrapper.
    """

    def __init__(
        self,
        settings: AppSettings,
        caption_gen: CaptionGenerator,
        metadata_ext: MetadataExtractor,
        clip_embedder: FashionCLIPEmbedder,
        text_embedder: TextEmbedder,
        qdrant_store: QdrantStore,
    ) -> None:
        self._settings = settings
        self._caption_gen = caption_gen
        self._metadata_ext = metadata_ext
        self._clip_embedder = clip_embedder
        self._text_embedder = text_embedder
        self._store = qdrant_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_directory(self, image_dir: Path) -> None:
        """Index all supported images found under ``image_dir``.

        Args:
            image_dir: Root directory to scan recursively for images.

        Raises:
            FileNotFoundError: If ``image_dir`` does not exist.
        """
        cfg = self._settings.indexing
        self._store.ensure_collection()

        image_paths = list(iter_images(image_dir, cfg.image_extensions))
        total = len(image_paths)
        logger.info("Found %d images to index in '%s'.", total, image_dir)

        if total == 0:
            logger.warning("No images found — nothing to index.")
            return

        processed = 0
        failed = 0

        # ----------------------------------------------------------------
        # Step 1: Per-image VLM pass (caption + metadata)
        # We accumulate results here and embed in batches below.
        # ----------------------------------------------------------------
        vlm_results: list[tuple[Path, Image.Image, str, FashionMetadata]] = []

        with tqdm(total=total, desc="VLM captioning", unit="img") as pbar:
            for path in image_paths:
                try:
                    pil = load_image(path)
                    caption = self._caption_gen.generate_caption(pil)
                    metadata = self._metadata_ext.extract(pil, caption)
                    vlm_results.append((path, pil, caption, metadata))
                except Exception as exc:  # noqa: BLE001
                    logger.error("VLM failed for '%s': %s", path, exc)
                    failed += 1
                finally:
                    pbar.update(1)

        logger.info(
            "VLM pass complete: %d succeeded, %d failed.",
            len(vlm_results),
            failed,
        )

        # ----------------------------------------------------------------
        # Step 2: Batch embedding + upsert
        # ----------------------------------------------------------------
        for batch in chunk_list(vlm_results, cfg.batch_size):
            paths, pils, captions, metadatas = zip(*batch)
            paths = list(paths)
            pils = list(pils)
            captions = list(captions)
            metadatas = list(metadatas)

            try:
                clip_embs = self._clip_embedder.encode_images(pils)
                bge_embs = self._text_embedder.encode(captions)
            except Exception as exc:  # noqa: BLE001
                logger.error("Embedding batch failed: %s — skipping %d images.", exc, len(batch))
                failed += len(batch)
                continue

            records: list[IndexedImage] = []
            for i, path in enumerate(paths):
                records.append(
                    IndexedImage(
                        image_id=generate_image_id(path),
                        image_path=str(path),
                        caption=captions[i],
                        metadata=metadatas[i],
                        fashionclip_embedding=clip_embs[i].tolist(),
                        caption_embedding=bge_embs[i].tolist(),
                    )
                )

            try:
                self._store.upsert_batch(records)
                processed += len(records)
            except Exception as exc:  # noqa: BLE001
                logger.error("Qdrant upsert failed for batch: %s", exc)
                failed += len(records)

        logger.info(
            "Indexing complete. Indexed: %d | Failed: %d | Total Qdrant points: %d",
            processed,
            failed,
            self._store.count(),
        )
