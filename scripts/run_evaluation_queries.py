"""Run catalog-grounded evaluation queries against the live retrieval pipeline.

The default manifest is ``configs/evaluation_queries.json``.  It retains the
five evaluation categories in the assignment brief, but grounds every prompt
in an image that genuinely exists in the selected dataset.  This is important:
the current catalog is mostly fashion-show/runway imagery and has no yellow
raincoat or park-bench example, so reporting those prompts as an accuracy test
would produce an invalid metric.

The manifest includes relevant image IDs, allowing this script to report
Hit@K and MRR as well as the retrieved images.  A different dataset can be
evaluated by supplying another JSON manifest with the same schema.

Usage:
    uv run python scripts/run_evaluation_queries.py --top-k 5
    uv run python scripts/run_evaluation_queries.py --query-file data/my_queries.json
    uv run python scripts/run_evaluation_queries.py --top-k 5 --copy-images
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

_DEFAULT_QUERY_FILE = "configs/evaluation_queries.json"


def _configure_hf_cache() -> None:
    if load_dotenv is not None:
        load_dotenv()
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run catalog-grounded evaluation queries.")
    p.add_argument("--top-k", type=int, default=5, help="Number of results to show per query.")
    p.add_argument("--output", type=str, default="dataset_eval_results.json",
                    help="Where to write full JSON results.")
    p.add_argument(
        "--query-file",
        type=str,
        default=_DEFAULT_QUERY_FILE,
        help=(
            "JSON list of {label, query, relevant_image_ids, rationale}. "
            f"Default: {_DEFAULT_QUERY_FILE}"
        ),
    )
    p.add_argument("--copy-images", action="store_true",
                    help="Copy top-k result images into eval_query_outputs/<query_slug>/ for visual inspection.")
    return p.parse_args(argv)


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:40].strip("_")


def _load_queries(path: Path) -> list[dict]:
    """Load and lightly validate a portable dataset-specific query manifest."""
    if not path.exists():
        raise SystemExit(f"Evaluation query file not found: {path}")

    with open(path, encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list) or not queries:
        raise SystemExit("Evaluation query file must contain a non-empty JSON list.")

    required_fields = {"label", "query", "relevant_image_ids"}
    for index, record in enumerate(queries, start=1):
        if not isinstance(record, dict) or not required_fields.issubset(record):
            raise SystemExit(
                f"Query {index} must contain: {', '.join(sorted(required_fields))}."
            )
        if not isinstance(record["relevant_image_ids"], list) or not record["relevant_image_ids"]:
            raise SystemExit(f"Query {index} must have at least one relevant_image_id.")
    return queries


def main(argv: list[str] | None = None) -> None:
    _configure_hf_cache()
    args = _parse_args(argv)
    query_manifest = _load_queries(Path(args.query_file))

    from src.embeddings.fashionclip_embedder import FashionCLIPEmbedder
    from src.embeddings.text_embedder import TextEmbedder
    from src.qdrant_store import QdrantStore
    from src.retrieval.query_parser import QueryParser
    from src.retrieval.reranker import Reranker
    from src.retrieval.retriever import Retriever
    from src.utils.config_loader import get_app_settings, get_model_settings
    from src.utils.logging_config import setup_logging

    app_cfg = get_app_settings()
    model_cfg = get_model_settings()
    setup_logging(level=app_cfg.logging.level, fmt=app_cfg.logging.format)
    logger = logging.getLogger(__name__)

    store = QdrantStore(app_cfg)
    if not store.collection_exists():
        raise SystemExit(
            f"Qdrant collection '{app_cfg.qdrant.collection_name}' not found. Run 'uv run build-index' first."
        )
    logger.info("Qdrant collection has %d indexed images.", store.count())

    clip_embedder = FashionCLIPEmbedder(model_cfg.fashionclip)
    text_embedder = TextEmbedder(model_cfg.text_embedding)
    query_parser = QueryParser(model_cfg.query_parser)
    retriever = Retriever(app_cfg, clip_embedder, text_embedder, store)
    reranker = Reranker(model_cfg.cross_encoder, app_cfg)
    reranker.attach_store(store)

    all_results = []
    reciprocal_ranks: list[float] = []
    hits_at_k = 0
    out_root = Path("eval_query_outputs")
    if args.copy_images:
        out_root.mkdir(exist_ok=True)

    for query_spec in query_manifest:
        label = query_spec["label"]
        query = query_spec["query"]
        relevant_ids = set(query_spec["relevant_image_ids"])
        print(f"\n{'='*70}\n[{label}] {query!r}\n{'='*70}")

        parsed = query_parser.parse(query)
        candidates = retriever.retrieve(query, parsed)
        ranked_results = reranker.rerank(query, candidates)
        relevant_rank = next(
            (rank for rank, result in enumerate(ranked_results, start=1) if result.image_id in relevant_ids),
            None,
        )
        results = ranked_results[: args.top_k]
        reciprocal_ranks.append(1.0 / relevant_rank if relevant_rank else 0.0)
        hits_at_k += int(relevant_rank is not None and relevant_rank <= args.top_k)

        print("Parsed structured metadata:")
        print(json.dumps(parsed.model_dump(), indent=2))

        query_record = {
            "label": label,
            "query": query,
            "rationale": query_spec.get("rationale", ""),
            "relevant_image_ids": sorted(relevant_ids),
            "relevant_rank": relevant_rank,
            f"hit_at_{args.top_k}": relevant_rank is not None and relevant_rank <= args.top_k,
            "parsed_metadata": parsed.model_dump(),
            "results": [],
        }

        print(f"Known relevant image ID(s): {', '.join(sorted(relevant_ids))}")
        print(f"Best relevant rank: {relevant_rank or 'not retrieved'}")

        print(f"\nTop {len(results)} results:")
        for rank, r in enumerate(results, start=1):
            filename = Path(r.image_path).name
            print(f"  {rank}. score={r.score:.4f}  file={filename}")
            print(f"     caption: {r.caption[:160]}{'...' if len(r.caption) > 160 else ''}")
            query_record["results"].append({
                "rank": rank,
                "score": round(float(r.score), 4),
                "image_id": r.image_id,
                "filename": filename,
                "image_path": r.image_path,
                "caption": r.caption,
                "metadata": r.metadata.model_dump(),
            })

            if args.copy_images:
                dest_dir = out_root / f"{_slugify(label)}"
                dest_dir.mkdir(exist_ok=True)
                src_path = Path(r.image_path)
                if src_path.exists():
                    shutil.copy(src_path, dest_dir / f"rank{rank}_{filename}")

        all_results.append(query_record)

    summary = {
        "query_file": str(args.query_file),
        "num_queries": len(all_results),
        f"hit_at_{args.top_k}": round(hits_at_k / len(all_results), 4),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": all_results}, f, indent=2)

    print(f"\n\nEvaluation summary: {json.dumps(summary)}")
    print(f"Full results written to: {args.output}")
    if args.copy_images:
        print(f"Result images copied into: {out_root}/<query>/")


if __name__ == "__main__":
    main()
