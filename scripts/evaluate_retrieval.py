"""
evaluate_retrieval.py — Self-retrieval Recall@K evaluation for the Fashion
Retrieval System.

Methodology
-----------
This uses the standard cross-modal retrieval evaluation protocol (the same
family of metrics used for CLIP/ALIGN-style Recall@K reporting on COCO /
Flickr30k image-text retrieval benchmarks):

  For each already-indexed image, we already have a VLM-generated caption
  (produced during offline indexing). We feed that caption back into the
  LIVE production pipeline (QueryParser -> hybrid Retriever -> Reranker) as
  a search query, and check whether the system retrieves the original
  source image among the top-K results.

This is a reasonable proxy for real-world query performance because real
user queries are also natural-language descriptions of a desired image —
here we simply already know the ground-truth answer for each query, which
lets us measure retrieval quality quantitatively without manual relevance
labeling.

Metrics reported
-----------------
- Recall@1 / Recall@5 / Recall@10 — fraction of queries where the
  ground-truth image appears in the top-K final (reranked) results.
- MRR (Mean Reciprocal Rank) over the reranked list.
- Average end-to-end query latency (parse + retrieve + rerank), excluding
  one-time model load time.

Usage
-----
    uv run python scripts/evaluate_retrieval.py --num-samples 100 --seed 42

Output
------
Writes a JSON file (default: evaluation_results.json) containing the
summary metrics and a per-query breakdown, and prints the summary to
stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _configure_hf_cache() -> None:
    if load_dotenv is not None:
        load_dotenv()
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate-retrieval",
        description="Self-retrieval Recall@K / MRR evaluation for the fashion retrieval pipeline.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of (image, caption) pairs to sample and evaluate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used for sampling (for reproducibility).",
    )
    parser.add_argument(
        "--captions-path", type=str, default="data/.index_staging/captions.jsonl",
        help="Path to the JSONL file of {image_id, image_path, caption} written during indexing.",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.json",
        help="Where to write the full JSON results (summary + per-query).",
    )
    return parser.parse_args(argv)


def _load_caption_pairs(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Captions file not found at {path}. "
            "Make sure you run this from the project root after indexing has been run."
        )
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs


def main(argv: list[str] | None = None) -> None:
    _configure_hf_cache()
    args = _parse_args(argv)
    random.seed(args.seed)

    # ---- Deferred imports (match search_cli.py wiring exactly) ----
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
            f"Qdrant collection '{app_cfg.qdrant.collection_name}' not found. "
            "Run 'uv run build-index' first."
        )

    n_indexed = store.count()
    logger.info("Qdrant collection has %d indexed images.", n_indexed)

    caption_pairs = _load_caption_pairs(Path(args.captions_path))
    logger.info("Loaded %d (image_id, caption) pairs from staging.", len(caption_pairs))

    sample_size = min(args.num_samples, len(caption_pairs))
    sample = random.sample(caption_pairs, sample_size)
    logger.info("Evaluating on %d sampled queries (seed=%d).", sample_size, args.seed)

    # ---- Load pipeline components (identical wiring to search_cli.py) ----
    clip_embedder = FashionCLIPEmbedder(model_cfg.fashionclip)
    text_embedder = TextEmbedder(model_cfg.text_embedding)
    query_parser = QueryParser(model_cfg.query_parser)
    retriever = Retriever(app_cfg, clip_embedder, text_embedder, store)
    reranker = Reranker(model_cfg.cross_encoder, app_cfg)
    reranker.attach_store(store)

    hits_at = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    per_query_records: list[dict] = []

    for i, pair in enumerate(sample, start=1):
        image_id = pair["image_id"]
        caption = pair["caption"]

        t0 = time.perf_counter()
        parsed = query_parser.parse(caption)
        candidates = retriever.retrieve(caption, parsed)
        results = reranker.rerank(caption, candidates)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        rank = None
        for idx, r in enumerate(results, start=1):
            if r.image_id == image_id:
                rank = idx
                break

        if rank is not None:
            reciprocal_ranks.append(1.0 / rank)
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
        else:
            reciprocal_ranks.append(0.0)

        per_query_records.append({
            "image_id": image_id,
            "caption_snippet": caption[:150],
            "rank_found": rank,
            "latency_sec": round(elapsed, 3),
        })

        logger.info(
            "[%d/%d] image_id=%s rank_found=%s latency=%.2fs",
            i, sample_size, image_id, rank, elapsed,
        )

    n = sample_size
    summary = {
        "num_indexed_images": n_indexed,
        "num_queries_evaluated": n,
        "recall_at_1": round(hits_at[1] / n, 4),
        "recall_at_5": round(hits_at[5] / n, 4),
        "recall_at_10": round(hits_at[10] / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "avg_query_latency_sec": round(sum(latencies) / n, 3),
        "rerank_top_k": app_cfg.retrieval.rerank_top_k,
        "retrieval_top_k": app_cfg.retrieval.retrieval_top_k,
        "seed": args.seed,
    }

    output = {"summary": summary, "per_query": per_query_records}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Evaluation Summary (self-retrieval Recall@K) ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nFull results (incl. per-query breakdown) written to: {args.output}")


if __name__ == "__main__":
    main()
