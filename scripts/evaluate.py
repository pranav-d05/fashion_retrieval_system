"""Unified evaluation script for the Fashion Retrieval System.

This script consolidates three evaluation modes:
1. golden-set: Evaluates a balanced golden query set against the live retrieval pipeline.
2. self-retrieval: Self-retrieval Recall@K evaluation using offline generated captions.
3. inspect-queries: Runs catalog-grounded queries and provides detailed printed output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def _slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower())[:40].strip("_")


def _summarise_metrics(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not records:
        return {"num_queries": 0, "mrr": 0.0, "map": 0.0, "precision_at_1": 0.0, "recall_at_1": 0.0, "hit_at_1": 0.0, "precision_at_5": 0.0, "recall_at_5": 0.0, "hit_at_5": 0.0, "precision_at_10": 0.0, "recall_at_10": 0.0, "hit_at_10": 0.0}

    summary: dict[str, float | int] = {"num_queries": len(records)}
    metric_keys = ["mrr", "map", "precision_at_1", "recall_at_1", "hit_at_1", "precision_at_5", "recall_at_5", "hit_at_5", "precision_at_10", "recall_at_10", "hit_at_10"]
    for key in metric_keys:
        summary[key] = round(sum(float(record.get(key, 0.0)) for record in records) / len(records), 4)
    return summary


def _load_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Dataset file not found: {path}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise SystemExit("Dataset must be a non-empty JSON list.")

    required_fields = {"label", "query", "relevant_image_ids"}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or not required_fields.issubset(record):
            raise SystemExit(
                f"Dataset row {index} must contain: {', '.join(sorted(required_fields))}."
            )
        if not isinstance(record["relevant_image_ids"], list) or not record["relevant_image_ids"]:
            raise SystemExit(f"Dataset row {index} must have at least one relevant_image_id.")
    return records


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


def _setup_pipeline() -> dict[str, Any]:
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

    return {
        "app_cfg": app_cfg,
        "store": store,
        "query_parser": query_parser,
        "retriever": retriever,
        "reranker": reranker,
        "logger": logger,
    }


def run_golden_set(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    dataset = _load_dataset(Path(args.dataset))
    if args.shuffle:
        random.shuffle(dataset)
        
    pipeline = _setup_pipeline()
    logger = pipeline["logger"]
    query_parser = pipeline["query_parser"]
    retriever = pipeline["retriever"]
    reranker = pipeline["reranker"]
    store = pipeline["store"]
    
    from src.retrieval.metrics import aggregate_metrics, evaluate_rankings

    per_query_records: list[dict[str, object]] = []
    category_metrics: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    latencies: list[float] = []

    for index, item in enumerate(dataset, start=1):
        category = str(item.get("category", "Uncategorized"))
        label = str(item["label"])
        query = str(item["query"])
        relevant_ids = set(item["relevant_image_ids"])

        t0 = time.perf_counter()
        parsed = query_parser.parse(query)
        candidates = retriever.retrieve(query, parsed)
        ranked_results = reranker.rerank(query, candidates)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        ranked_ids = [result.image_id for result in ranked_results]
        metrics = evaluate_rankings(ranked_ids, relevant_ids, ks=(1, 5, 10))
        category_metrics[category].append(metrics)
        relevant_rank = next(
            (rank for rank, result in enumerate(ranked_results, start=1) if result.image_id in relevant_ids),
            None,
        )

        per_query_records.append(
            {
                "category": category,
                "label": label,
                "query": query,
                "rationale": item.get("rationale", ""),
                "relevant_image_ids": sorted(relevant_ids),
                "relevant_rank": relevant_rank,
                "latency_sec": round(elapsed, 3),
                **metrics,
            }
        )

        logger.info(
            "[%d/%d] category=%s label=%s relevant_rank=%s latency=%.2fs",
            index,
            len(dataset),
            category,
            label,
            relevant_rank,
            elapsed,
        )

    overall_summary = aggregate_metrics([record for record in per_query_records if isinstance(record, dict)], ks=(1, 5, 10))
    category_summary = {
        category: _summarise_metrics(metrics)
        for category, metrics in category_metrics.items()
    }
    balanced_summary = _summarise_metrics(list(category_summary.values()))
    balanced_summary["avg_query_latency_sec"] = round(sum(latencies) / len(latencies), 3)
    balanced_summary["num_categories"] = len(category_summary)
    balanced_summary["num_queries"] = len(per_query_records)
    balanced_summary["num_indexed_images"] = store.count()

    payload = {
        "summary": balanced_summary,
        "overall_summary": overall_summary,
        "category_summary": category_summary,
        "queries": per_query_records,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Balanced Golden Evaluation Summary ===")
    for key, value in balanced_summary.items():
        print(f"{key}: {value}")
    print(f"\nFull results written to: {args.output}")


def run_self_retrieval(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    
    pipeline = _setup_pipeline()
    logger = pipeline["logger"]
    query_parser = pipeline["query_parser"]
    retriever = pipeline["retriever"]
    reranker = pipeline["reranker"]
    store = pipeline["store"]
    app_cfg = pipeline["app_cfg"]

    n_indexed = store.count()
    caption_pairs = _load_caption_pairs(Path(args.captions_path))
    logger.info("Loaded %d (image_id, caption) pairs from staging.", len(caption_pairs))

    sample_size = min(args.num_samples, len(caption_pairs))
    sample = random.sample(caption_pairs, sample_size)
    logger.info("Evaluating on %d sampled queries (seed=%d).", sample_size, args.seed)

    from src.retrieval.metrics import aggregate_metrics, evaluate_rankings

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

        ranked_ids = [result.image_id for result in results]
        metrics = evaluate_rankings(ranked_ids, [image_id], ks=(1, 5, 10))

        rank = next((idx for idx, result in enumerate(results, start=1) if result.image_id == image_id), None)

        per_query_records.append({
            "image_id": image_id,
            "caption_snippet": caption[:150],
            "rank_found": rank,
            "latency_sec": round(elapsed, 3),
            **metrics,
        })

        logger.info(
            "[%d/%d] image_id=%s rank_found=%s latency=%.2fs",
            i, sample_size, image_id, rank, elapsed,
        )

    n = sample_size
    summary = aggregate_metrics(per_query_records, ks=(1, 5, 10))
    summary.update({
        "num_indexed_images": n_indexed,
        "avg_query_latency_sec": round(sum(latencies) / n, 3),
        "rerank_top_k": app_cfg.retrieval.rerank_top_k,
        "retrieval_top_k": app_cfg.retrieval.retrieval_top_k,
        "seed": args.seed,
        "num_queries_evaluated": sample_size,
    })

    output = {"summary": summary, "per_query": per_query_records}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Evaluation Summary (self-retrieval metrics) ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nFull results (incl. per-query breakdown) written to: {args.output}")


def run_inspect_queries(args: argparse.Namespace) -> None:
    query_manifest = _load_dataset(Path(args.query_file))

    pipeline = _setup_pipeline()
    query_parser = pipeline["query_parser"]
    retriever = pipeline["retriever"]
    reranker = pipeline["reranker"]

    from src.retrieval.metrics import aggregate_metrics, evaluate_rankings

    all_results = []
    per_query_metrics: list[dict[str, float | int]] = []
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
        ranked_ids = [result.image_id for result in ranked_results]
        metrics = evaluate_rankings(ranked_ids, relevant_ids, ks=(args.top_k,))
        relevant_rank = next(
            (rank for rank, result in enumerate(ranked_results, start=1) if result.image_id in relevant_ids),
            None,
        )
        results = ranked_results[: args.top_k]
        per_query_metrics.append(metrics)

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
            **metrics,
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

    summary = aggregate_metrics(per_query_metrics, ks=(args.top_k,))
    summary.update({
        "query_file": str(args.query_file),
        "num_queries": len(all_results),
    })
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": all_results}, f, indent=2)

    print(f"\n\nEvaluation summary: {json.dumps(summary)}")
    print(f"Full results written to: {args.output}")
    if args.copy_images:
        print(f"Result images copied into: {out_root}/<query>/")


def main(argv: list[str] | None = None) -> None:
    _configure_hf_cache()
    
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Unified evaluation script for the Fashion Retrieval System.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Evaluation mode to run")

    # Mode 1: golden-set
    p_golden = subparsers.add_parser("golden-set", help="Evaluate a balanced golden query set against the live pipeline.")
    p_golden.add_argument("--dataset", type=str, default="configs/golden_dataset.json", help="Path to the golden query manifest.")
    p_golden.add_argument("--output", type=str, default="golden_eval_results.json", help="Where to write the JSON evaluation report.")
    p_golden.add_argument("--top-k", type=int, default=5, help="How many top results to include in the per-query report.")
    p_golden.add_argument("--seed", type=int, default=42, help="Optional seed used only when the dataset is shuffled.")
    p_golden.add_argument("--shuffle", action="store_true", help="Shuffle the golden queries before running them.")

    # Mode 2: self-retrieval
    p_self = subparsers.add_parser("self-retrieval", help="Self-retrieval Recall@K / MRR evaluation.")
    p_self.add_argument("--num-samples", type=int, default=100, help="Number of (image, caption) pairs to sample and evaluate.")
    p_self.add_argument("--seed", type=int, default=42, help="Random seed used for sampling (for reproducibility).")
    p_self.add_argument("--captions-path", type=str, default="data/.index_staging/captions.jsonl", help="Path to the JSONL file of {image_id, image_path, caption} written during indexing.")
    p_self.add_argument("--output", type=str, default="evaluation_results.json", help="Where to write the full JSON results (summary + per-query).")

    # Mode 3: inspect-queries
    p_inspect = subparsers.add_parser("inspect-queries", help="Run catalog-grounded evaluation queries and inspect results.")
    p_inspect.add_argument("--top-k", type=int, default=5, help="Number of results to show per query.")
    p_inspect.add_argument("--output", type=str, default="dataset_eval_results.json", help="Where to write full JSON results.")
    p_inspect.add_argument("--query-file", type=str, default="configs/evaluation_queries.json", help="JSON list of {label, query, relevant_image_ids, rationale}.")
    p_inspect.add_argument("--copy-images", action="store_true", help="Copy top-k result images into eval_query_outputs/<query_slug>/ for visual inspection.")

    args = parser.parse_args(argv)

    if args.mode == "golden-set":
        run_golden_set(args)
    elif args.mode == "self-retrieval":
        run_self_retrieval(args)
    elif args.mode == "inspect-queries":
        run_inspect_queries(args)

if __name__ == "__main__":
    main()
