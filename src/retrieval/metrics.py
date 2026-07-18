"""Reusable retrieval metrics for ranked fashion-search results.

The helpers in this module operate on ranked image IDs and a set of relevant
image IDs. They are shared by the evaluation scripts so precision@k,
recall@k, hit rate, MRR, and MAP are computed consistently everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _top_k(rankings: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        return []
    return list(rankings[:k])


def _relevant_set(relevant_ids: Iterable[str]) -> set[str]:
    return {image_id for image_id in relevant_ids if image_id}


def hit_rate_at_k(rankings: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Return 1.0 if any relevant item appears in the top-k results."""
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    return float(any(image_id in relevant for image_id in _top_k(rankings, k)))


def precision_at_k(rankings: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Return the fraction of top-k results that are relevant."""
    top_k = _top_k(rankings, k)
    if not top_k:
        return 0.0
    relevant = _relevant_set(relevant_ids)
    hits = sum(1 for image_id in top_k if image_id in relevant)
    return hits / len(top_k)


def recall_at_k(rankings: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Return the fraction of relevant items recovered in the top-k results."""
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = _top_k(rankings, k)
    hits = sum(1 for image_id in top_k if image_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(rankings: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    for index, image_id in enumerate(rankings, start=1):
        if image_id in relevant:
            return 1.0 / index
    return 0.0


def average_precision_at_k(rankings: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Return average precision at k for a ranked list.

    This is the mean of the precision values measured at each rank where a
    relevant item appears, truncated to the first k results.
    """
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0

    top_k = _top_k(rankings, k)
    hits = 0
    precision_sum = 0.0
    for index, image_id in enumerate(top_k, start=1):
        if image_id in relevant:
            hits += 1
            precision_sum += hits / index

    if hits == 0:
        return 0.0
    return precision_sum / min(len(relevant), k)


def mean_reciprocal_rank(rankings: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """Alias for reciprocal_rank for metric naming consistency."""
    return reciprocal_rank(rankings, relevant_ids)


def evaluate_rankings(
    rankings: Sequence[str],
    relevant_ids: Iterable[str],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float | int]:
    """Compute a standard retrieval metric bundle for one ranked list."""
    relevant = tuple(relevant_ids)
    summary: dict[str, float | int] = {
        "mrr": round(reciprocal_rank(rankings, relevant), 4),
        "map": round(average_precision_at_k(rankings, relevant, max(ks, default=1)), 4),
    }
    for k in ks:
        summary[f"precision_at_{k}"] = round(precision_at_k(rankings, relevant, k), 4)
        summary[f"recall_at_{k}"] = round(recall_at_k(rankings, relevant, k), 4)
        summary[f"hit_at_{k}"] = round(hit_rate_at_k(rankings, relevant, k), 4)
    return summary


def aggregate_metrics(
    records: Sequence[dict[str, float | int]],
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float | int]:
    """Aggregate per-query metric dictionaries into a dataset-level summary."""
    if not records:
        summary: dict[str, float | int] = {"num_queries": 0}
        for k in ks:
            summary[f"precision_at_{k}"] = 0.0
            summary[f"recall_at_{k}"] = 0.0
            summary[f"hit_at_{k}"] = 0.0
        summary["mrr"] = 0.0
        summary["map"] = 0.0
        return summary

    summary: dict[str, float | int] = {"num_queries": len(records)}
    metric_keys = ["mrr", "map"]
    for k in ks:
        metric_keys.extend([f"precision_at_{k}", f"recall_at_{k}", f"hit_at_{k}"])

    for key in metric_keys:
        summary[key] = round(sum(float(record.get(key, 0.0)) for record in records) / len(records), 4)
    return summary