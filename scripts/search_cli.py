"""
search_cli — Interactive online retrieval entry point.

Usage:
    uv run search                        # Interactive REPL mode
    uv run search --query "blue dress"   # Single-query mode
    uv run search --top-k 5              # Override rerank_top_k

This script:
  1. Loads configuration and all retrieval components.
  2. Accepts a natural language query (CLI arg or interactive REPL).
  3. Runs the full pipeline: parse → retrieve → rerank.
  4. Prints ranked results to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path



def _configure_hf_cache() -> None:
    # Honour any HF_HOME already set in the environment / .env file;
    # only fall back to D:\hf_cache if nothing is configured.
    hf_home = os.environ.get("HF_HOME", r"D:\hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="search",
        description="Interactive fashion image retrieval CLI.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Search query. If omitted, starts an interactive REPL.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to display (overrides config rerank_top_k).",
    )
    return parser.parse_args(argv)


def _print_results(results, top_k: int, parsed_metadata=None) -> None:
    """Pretty-print retrieval results to stdout."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    table = Table(
        title=f"Top {min(len(results), top_k)} Fashion Matches",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Rank", style="dim", width=5, justify="right")
    table.add_column("Score", width=8, justify="right")
    table.add_column("Caption", min_width=40)

    for rank, result in enumerate(results[:top_k], start=1):
        table.add_row(
            str(rank),
            f"{result.score:.4f}",
            result.caption[:120] + ("…" if len(result.caption) > 120 else ""),
        )

    console.print(table)

    if parsed_metadata is not None:
        console.print()
        console.print(
            Panel(
                json.dumps(parsed_metadata.model_dump(), indent=2),
                title="Parsed Query Metadata",
                border_style="green",
            )
        )

    for rank, result in enumerate(results[:top_k], start=1):
        image_path = Path(result.image_path)
        console.print()
        console.print(
            Panel(
                f"[bold]Rank {rank}[/bold]  [cyan]{result.score:.4f}[/cyan]\n"
                f"[bold]Caption[/bold]\n{result.caption}\n\n"
                f"[bold]Metadata[/bold]\n{json.dumps(result.metadata.model_dump(), indent=2)}",
                title=image_path.name,
                border_style="cyan",
            )
        )

        if image_path.exists():
            try:
                _open_image(image_path)
                console.print(f"[dim]Opened image: {image_path.name}[/dim]")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[dim]Could not open image: {exc}[/dim]")
        else:
            console.print(f"[dim]Image not found: {image_path}[/dim]")


def _open_image(image_path: Path) -> None:
    """Open an image in the default viewer for the current platform."""
    if sys.platform.startswith("win"):
        os.startfile(image_path)  # type: ignore[attr-defined]
        return

    if sys.platform == "darwin":
        subprocess.Popen(["open", str(image_path)])
        return

    subprocess.Popen(["xdg-open", str(image_path)])


def _run_query(query: str, pipeline: dict, top_k: int) -> None:
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Processing query: '%s'", query)

    parsed = pipeline["query_parser"].parse(query)
    candidates = pipeline["retriever"].retrieve(query, parsed)
    results = pipeline["reranker"].rerank(query, candidates)

    _print_results(results, top_k, parsed_metadata=parsed)


def main(argv: list[str] | None = None) -> None:
    _configure_hf_cache()
    args = _parse_args(argv)

    # ---- Deferred imports ----
    from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
    from src.embeddings.text_embedder import TextEmbedder
    from src.qdrant_store import QdrantStore
    from src.retrieval.query_parser import QueryParser
    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import Retriever
    from src.utils.config_loader import get_app_settings, get_model_settings
    from src.utils.logging_config import setup_logging

    import logging

    # ---- Setup ----
    app_cfg = get_app_settings()
    model_cfg = get_model_settings()
    setup_logging(level=app_cfg.logging.level, fmt=app_cfg.logging.format)
    logger = logging.getLogger(__name__)

    top_k = args.top_k or app_cfg.retrieval.rerank_top_k

    # ---- Validate collection exists ----
    store = QdrantStore(app_cfg)
    if not store.collection_exists():
        logger.error(
            "Qdrant collection '%s' not found. Run 'build-index' first.",
            app_cfg.qdrant.collection_name,
        )
        sys.exit(1)

    n_indexed = store.count()
    logger.info("Qdrant collection has %d indexed images.", n_indexed)

    # ---- Load retrieval components ----
    clip_embedder = FashionCLIPEmbedder(model_cfg.fashionclip)
    text_embedder = TextEmbedder(model_cfg.text_embedding)
    query_parser = QueryParser(model_cfg.query_parser)
    retriever = Retriever(app_cfg, clip_embedder, text_embedder, store)
    reranker = Reranker(model_cfg.cross_encoder, app_cfg)
    reranker.attach_store(store)

    pipeline = {
        "query_parser": query_parser,
        "retriever": retriever,
        "reranker": reranker,
    }

    # ---- Single-query or REPL mode ----
    if args.query:
        _run_query(args.query, pipeline, top_k)
    else:
        # Interactive REPL
        from rich.console import Console
        console = Console()
        console.print(
            "[bold green]Fashion Retrieval System[/bold green] — "
            "Type a query and press Enter. Type 'quit' or Ctrl-C to exit.\n"
        )
        while True:
            try:
                query = input("🔍 Query: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not query:
                continue
            if query.lower() in {"quit", "exit", "q"}:
                console.print("[dim]Goodbye![/dim]")
                break

            try:
                _run_query(query, pipeline, top_k)
            except Exception as exc:  # noqa: BLE001
                logger.error("Query failed: %s", exc)


if __name__ == "__main__":
    main()
