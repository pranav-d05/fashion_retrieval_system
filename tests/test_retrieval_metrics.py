from __future__ import annotations

from src.retrieval.metrics import (
    aggregate_metrics,
    average_precision_at_k,
    evaluate_rankings,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ranked_metrics_compute_expected_values():
    rankings = ["A", "B", "C", "D"]
    relevant = {"B", "D"}

    assert precision_at_k(rankings, relevant, 1) == 0.0
    assert precision_at_k(rankings, relevant, 2) == 0.5
    assert recall_at_k(rankings, relevant, 2) == 0.5
    assert hit_rate_at_k(rankings, relevant, 2) == 1.0
    assert reciprocal_rank(rankings, relevant) == 0.5
    assert average_precision_at_k(rankings, relevant, 4) == 0.5


def test_evaluate_and_aggregate_rankings():
    query_metrics = [
        evaluate_rankings(["A", "B", "C"], ["B"], ks=(1, 3)),
        evaluate_rankings(["X", "Y", "Z"], ["Z"], ks=(1, 3)),
    ]

    summary = aggregate_metrics(query_metrics, ks=(1, 3))

    assert summary["num_queries"] == 2
    assert summary["precision_at_1"] == 0.0
    assert summary["recall_at_1"] == 0.0
    assert summary["precision_at_3"] == 0.3333
    assert summary["recall_at_3"] == 1.0
    assert summary["hit_at_3"] == 1.0
    assert summary["mrr"] == 0.4167