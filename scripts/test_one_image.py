"""
test_one_image.py — End-to-end pipeline test on a single real image.

Tests every component in sequence:
  1. VLM caption generator  (Qwen2.5-VL-3B, 4-bit on CUDA)
  2. Metadata extractor      (Qwen2.5-1.5B, 4-bit on CUDA)
  3. FashionCLIP image + text embed
  4. BGE caption embed
  5. Cross-encoder reranker  (sanity score check)

Usage:
    uv run python scripts/test_one_image.py
    uv run python scripts/test_one_image.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

# ── HF cache ──────────────────────────────────────────────────────────────────
hf_home = os.environ.get("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_HOME", hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))

import torch
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich import print as rprint

console = Console()

_DEFAULT_IMAGE = Path("data/images/test/003d41dd20f271d27219fe7ee6de727d.jpg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image pipeline test.")
    parser.add_argument(
        "--image", type=Path, default=_DEFAULT_IMAGE,
        help="Path to a fashion image to test with.",
    )
    return parser.parse_args()


def section(title: str) -> None:
    console.print(f"\n{Rule(title, style='bold cyan')}")


def ok(msg: str) -> None:
    console.print(f"  [bold green]✓[/bold green] {msg}")


def fail(label: str, exc: Exception) -> None:
    console.print(f"  [bold red]✗[/bold red] {label}")
    console.print(f"    [red]{exc}[/red]")


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = _parse_args()
    image_path = args.image.resolve()

    if not image_path.exists():
        console.print(f"[red]Image not found: {image_path}[/red]")
        console.print("Pass a valid path with --image <path>")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Fashion Retrieval — Single Image Pipeline Test[/bold]\n"
        f"Image: [cyan]{image_path}[/cyan]",
        expand=False,
    ))

    from PIL import Image as PILImage
    img = PILImage.open(image_path).convert("RGB")
    console.print(f"  Loaded: [dim]{img.size[0]}×{img.size[1]} px[/dim]")

    from src.utils.config_loader import get_model_settings, get_app_settings
    cfg = get_model_settings()
    app_cfg = get_app_settings()

    caption: str = ""
    from src.schemas import FashionMetadata
    metadata = FashionMetadata()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. VLM Caption Generator
    # ──────────────────────────────────────────────────────────────────────────
    section("1 · VLM Caption Generator  (Qwen2.5-VL-3B · 4-bit CUDA)")
    try:
        from src.vlm.vlm_backend import VLMBackend
        from src.vlm.caption_generator import CaptionGenerator

        backend = VLMBackend(cfg.vision_language_model)
        generator = CaptionGenerator(backend)
        caption = generator.generate_caption(img)
        ok(f"Caption generated ({len(caption)} chars)")
        console.print(Panel(caption, title="Caption", border_style="green"))
        del generator, backend
        _release()
    except Exception as exc:
        fail("Caption generation failed", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Metadata Extractor
    # ──────────────────────────────────────────────────────────────────────────
    section("2 · Metadata Extractor  (Qwen2.5-1.5B · 4-bit CUDA)")
    if caption:
        try:
            from src.vlm.metadata_extractor import MetadataExtractor

            extractor = MetadataExtractor(cfg.query_parser)
            metadata = extractor.extract(caption)
            ok("Metadata extracted")
            rprint(metadata.model_dump())

            has_person = metadata.person.gender is not None
            has_scene  = metadata.scene.location is not None or metadata.scene.environment is not None
            console.print(f"  Person gender populated: {'[green]Yes ✓[/green]' if has_person else '[yellow]No[/yellow]'}")
            console.print(f"  Scene populated:         {'[green]Yes ✓[/green]' if has_scene  else '[yellow]No[/yellow]'}")

            del extractor
            _release()
        except Exception as exc:
            fail("Metadata extraction failed", exc)
    else:
        console.print("  [yellow]Skipped — no caption available.[/yellow]")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. FashionCLIP
    # ──────────────────────────────────────────────────────────────────────────
    section("3 · FashionCLIP Embedder  (image + text → 512-dim)")
    try:
        import numpy as np
        from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder

        clip = FashionCLIPEmbedder(cfg.fashionclip)
        img_vec = clip.encode_images([img])[0]
        txt_vec = clip.encode_texts([caption or "fashion clothing"])[0]
        sim = float(np.dot(img_vec, txt_vec))

        ok(f"Image vector dim={len(img_vec)}")
        ok(f"Text  vector dim={len(txt_vec)}")
        ok(f"Cosine similarity (image↔caption): [bold]{sim:.4f}[/bold]")
        if sim > 0.2:
            console.print("  [green]Good alignment ✓[/green]")
        else:
            console.print("  [yellow]Low similarity — caption may not describe the image well.[/yellow]")
    except Exception as exc:
        fail("FashionCLIP failed", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. BGE Text Embedder
    # ──────────────────────────────────────────────────────────────────────────
    section("4 · BGE Text Embedder  (caption → 768-dim)")
    try:
        from src.embeddings.text_embedder import TextEmbedder

        bge = TextEmbedder(cfg.text_embedding)
        bge_vec = bge.encode([caption or "fashion clothing"])[0]
        ok(f"Caption embedding dim={len(bge_vec)}, norm≈{float((bge_vec**2).sum()**0.5):.4f}")
    except Exception as exc:
        fail("BGE embedding failed", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Cross-Encoder Reranker
    # ──────────────────────────────────────────────────────────────────────────
    section("5 · Cross-Encoder Reranker")
    try:
        from src.retrieval.reranker import Reranker
        from src.schemas import RetrievalResult

        reranker = Reranker(cfg.cross_encoder, app_cfg)
        candidates = [
            RetrievalResult(
                image_id="IMG_FASHION",
                image_path=str(image_path),
                caption=caption or "fashion clothing",
                metadata=metadata,
            ),
            RetrievalResult(
                image_id="IMG_NOISE",
                image_path="noise.jpg",
                caption="A medieval knight in heavy iron armour on horseback.",
                metadata=FashionMetadata(),
            ),
        ]
        ranked = reranker.rerank("fashion outfit", candidates)
        for r in ranked:
            console.print(f"  {r.image_id}: score={r.score:.4f}")

        if ranked[0].image_id == "IMG_FASHION":
            ok("[green]Fashion image ranked above noise candidate ✓[/green]")
        else:
            console.print("  [yellow]Warning: noise ranked higher — unusual.[/yellow]")
    except Exception as exc:
        fail("Reranker failed", exc)

    # ──────────────────────────────────────────────────────────────────────────
    console.print(f"\n{Rule('All 5 components tested', style='bold green')}\n")


if __name__ == "__main__":
    main()
