"""Memory-safe, resumable offline indexing pipeline.

Each model family is loaded only for the stage that needs it.  Intermediate
caption and metadata records are stored as JSONL so a long CPU run can resume
without keeping every image or model in memory at the same time.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
from src.embeddings.text_embedder import TextEmbedder
from src.indexing._image_loader import load_image
from src.qdrant_store import QdrantStore
from src.schemas import FashionMetadata, IndexedImage
from src.utils.config_loader import AppSettings, ModelSettings
from src.utils.helpers import chunk_list, generate_image_id, iter_images
from src.vlm.caption_generator import CaptionGenerator
from src.vlm.metadata_extractor import MetadataExtractor
from src.vlm.vlm_backend import VLMBackend

logger = logging.getLogger(__name__)


class StagedIndexer:
    """Index images without concurrently retaining all ML models in memory."""

    def __init__(
        self,
        settings: AppSettings,
        model_settings: ModelSettings,
        store: QdrantStore,
        staging_dir: Path,
    ) -> None:
        self._settings = settings
        self._models = model_settings
        self._store = store
        self._staging_dir = staging_dir
        self._captions_file = staging_dir / "captions.jsonl"
        self._metadata_file = staging_dir / "metadata.jsonl"
        self._failed_file = staging_dir / "failed.jsonl"

    def index_directory(self, image_dir: Path, *, skip_existing: bool = False) -> None:
        """Run caption, metadata, and embedding stages with disk checkpoints."""
        self._store.ensure_collection()
        paths = list(iter_images(image_dir, self._settings.indexing.image_extensions))
        if skip_existing:
            existing = self._store.existing_image_ids([generate_image_id(path) for path in paths])
            paths = [path for path in paths if generate_image_id(path) not in existing]
            logger.info("Skipping %d already-indexed images.", len(existing))
        if not paths:
            logger.info("No images require indexing.")
            return

        self._staging_dir.mkdir(parents=True, exist_ok=True)
        wanted = {generate_image_id(path): path for path in paths}
        caption_ok = self._caption_stage(wanted)
        if not caption_ok:
            logger.warning(
                "Caption stage aborted — CUDA context is poisoned for this process. "
                "Re-run build-index to continue with the remaining images."
            )
            return
        self._metadata_stage(wanted)
        self._embedding_stage(wanted)
        logger.info("Indexing complete. Total Qdrant points: %d", self._store.count())

    def _caption_stage(self, wanted: dict[str, Path]) -> bool:
        """Run caption stage. Returns True on full completion, False if aborted due to CUDA error."""
        existing = self._read_records(self._captions_file)
        failed_ids = self._read_failed_ids()
        pending = [
            (image_id, path)
            for image_id, path in wanted.items()
            if image_id not in existing and image_id not in failed_ids
        ]
        if not pending:
            logger.info("Caption stage already complete.")
            return True

        if failed_ids:
            logger.warning(
                "%d image(s) are permanently skipped (CUDA failures from a previous run): %s",
                len(failed_ids), list(failed_ids),
            )

        logger.info("Caption stage: %d images pending.", len(pending))
        backend = VLMBackend(self._models.vision_language_model)
        generator = CaptionGenerator(backend)
        cuda_aborted = False
        try:
            with self._captions_file.open("a", encoding="utf-8") as output:
                for image_id, path in tqdm(pending, desc="Captioning", unit="img"):
                    try:
                        caption = generator.generate_caption(load_image(path))
                        self._write_record(output, {
                            "image_id": image_id,
                            "image_path": str(path),
                            "caption": caption,
                        })
                    except RuntimeError as exc:
                        if "CUDA" not in str(exc):
                            logger.error("Caption failed for '%s': %s", path, exc)
                            continue
                        # A CUDA device-side assert permanently poisons the process-level
                        # CUDA context. Reinitializing the model inside the same process
                        # is not possible — the only recovery is to restart the process.
                        # Mark this image as permanently failed so the next run skips it
                        # and processes the remaining images cleanly.
                        logger.error(
                            "CUDA error on '%s': %s",
                            path, exc,
                        )
                        self._write_failed_id(image_id, str(path))
                        logger.warning(
                            "Image '%s' marked as permanently failed and will be skipped "
                            "on future runs. Re-run build-index to continue with the "
                            "remaining %d images.",
                            path,
                            len(pending) - pending.index((image_id, path)) - 1,
                        )
                        cuda_aborted = True
                        break  # CUDA context is poisoned; must exit and restart process
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Caption failed for '%s': %s", path, exc)
        finally:
            try:
                del generator, backend
            except Exception:  # noqa: BLE001
                pass
            self._release_memory()
        return not cuda_aborted

    def _metadata_stage(self, wanted: dict[str, Path]) -> None:
        captions = self._read_records(self._captions_file)
        existing = self._read_records(self._metadata_file)
        pending = [record for image_id, record in captions.items() if image_id in wanted and image_id not in existing]
        if not pending:
            logger.info("Metadata stage already complete.")
            return

        logger.info("Metadata stage: %d captions pending.", len(pending))
        extractor = MetadataExtractor(self._models.query_parser)
        try:
            from src.utils.helpers import chunk_list
            with self._metadata_file.open("a", encoding="utf-8") as output:
                batch_size = 4
                batches = list(chunk_list(pending, batch_size))
                for batch in tqdm(batches, desc="Extracting metadata", unit="batch"):
                    try:
                        captions = [record["caption"] for record in batch]
                        metadatas = extractor.extract_batch(captions)
                        for record, metadata in zip(batch, metadatas):
                            self._write_record(output, {**record, "metadata": metadata.model_dump()})
                    except RuntimeError as exc:
                        if "CUDA out of memory" in str(exc):
                            logger.warning("CUDA OOM during batch processing. Falling back to sequential extraction...")
                            for record in batch:
                                try:
                                    metadata = extractor.extract(record["caption"])
                                    self._write_record(output, {**record, "metadata": metadata.model_dump()})
                                except Exception as inner_exc:  # noqa: BLE001
                                    logger.error("Metadata extraction failed for '%s': %s", record["image_path"], inner_exc)
                        else:
                            raise
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Metadata batch extraction failed: %s", exc)
        finally:
            del extractor
            self._release_memory()

    def _embedding_stage(self, wanted: dict[str, Path]) -> None:
        metadata_records = self._read_records(self._metadata_file)
        pending = [record for image_id, record in metadata_records.items() if image_id in wanted]
        if not pending:
            logger.warning("No metadata records available for embedding.")
            return

        logger.info("Embedding stage: %d records pending.", len(pending))
        clip = FashionCLIPEmbedder(self._models.fashionclip)
        text = TextEmbedder(self._models.text_embedding)
        try:
            for batch in tqdm(list(chunk_list(pending, self._settings.indexing.batch_size)), desc="Embedding", unit="batch"):
                try:
                    images = [load_image(Path(record["image_path"])) for record in batch]
                    captions = [record["caption"] for record in batch]
                    clip_embeddings = clip.encode_images(images)
                    text_embeddings = text.encode(captions)
                    records = [
                        IndexedImage(
                            image_id=record["image_id"],
                            image_path=record["image_path"],
                            caption=record["caption"],
                            metadata=FashionMetadata(**record["metadata"]),
                            fashionclip_embedding=clip_embeddings[index].tolist(),
                            caption_embedding=text_embeddings[index].tolist(),
                        )
                        for index, record in enumerate(batch)
                    ]
                    self._store.upsert_batch(records)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Embedding/upsert batch failed: %s", exc)
        finally:
            del clip, text
            self._release_memory()

    @staticmethod
    def _read_records(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records[record["image_id"]] = record
        return records

    @staticmethod
    def _write_record(output, record: dict[str, Any]) -> None:
        output.write(json.dumps(record) + "\n")
        output.flush()

    def _read_failed_ids(self) -> set[str]:
        """Return image IDs that have permanently failed captioning (CUDA errors)."""
        if not self._failed_file.exists():
            return set()
        failed: set[str] = set()
        with self._failed_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    failed.add(record["image_id"])
        return failed

    def _write_failed_id(self, image_id: str, image_path: str) -> None:
        """Append a permanently-failed image to failed.jsonl so future runs skip it."""
        with self._failed_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"image_id": image_id, "image_path": image_path}) + "\n")

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                # CUDA context may already be poisoned (e.g. after a device-side
                # assert). Swallow the error — the context will be recreated when
                # the next model is instantiated.
                pass
