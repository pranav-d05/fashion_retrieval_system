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
import sys



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


def _print_results(results, top_k: int) -> None:
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
    table.add_column("Path", style="dim")

    for rank, result in enumerate(results[:top_k], start=1):
        table.add_row(
            str(rank),
            f"{result.score:.4f}",
            result.caption[:120] + ("…" if len(result.caption) > 120 else ""),
            result.image_path,
        )

    console.print(table)


def _run_query(query: str, pipeline: dict, top_k: int) -> None:
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Processing query: '%s'", query)

    parsed = pipeline["query_parser"].parse(query)
    candidates = pipeline["retriever"].retrieve(query, parsed)
    results = pipeline["reranker"].rerank(query, candidates)

    _print_results(results, top_k)


def main(argv: list[str] | None = None) -> None:
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
