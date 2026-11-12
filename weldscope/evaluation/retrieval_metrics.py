"""Deterministic metrics for explicitly labelled positive PDF-page anchors."""

from math import log2
from pathlib import Path


def page_key(document):
    return (Path(document.metadata.get("source", "")).name, document.metadata.get("page", -1) + 1)


def anchor_metrics(documents, evidence, k=5):
    """Unknown pages receive no credit; duplicate chunks do not repeat page gain.

    These are positive-anchor metrics, not exhaustive relevance judgements.
    """
    relevant = {(item["source"], item["page"]) for item in evidence}
    if not relevant:
        raise ValueError("Retrieval metrics require positive evidence anchors")
    found = set()
    first = None
    dcg = 0.0
    for rank, document in enumerate(documents[:k], 1):
        page = page_key(document)
        if page in relevant and page not in found:
            found.add(page)
            first = first or rank
            dcg += 1 / log2(rank + 1)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {
        "recall": len(found) / len(relevant),
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / ideal,
    }


def regression_failures(summary, routing_accuracy, dataset_sha256, thresholds):
    """Use frozen metric floors; never regenerate thresholds as part of a test run."""
    failures = []
    if thresholds["dataset_sha256"] != dataset_sha256:
        failures.append("Dataset changed; review labels and version thresholds")
    for variant, minima in thresholds["minimum"].items():
        for metric, minimum in minima.items():
            if summary[variant][metric] < minimum:
                failures.append(f"{variant}.{metric} below {minimum}")
    if routing_accuracy < thresholds["routing_accuracy"]:
        failures.append("Routing regression")
    return failures
