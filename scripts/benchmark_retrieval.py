"""Run the versioned FSW development benchmark without model-provider requests."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

from weldscope.conversation import plan_conversation
from weldscope.core.config import RAGConfig
from weldscope.evaluation.retrieval_metrics import anchor_metrics, page_key, regression_failures
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.retrieval.reranker import Reranker

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(dataset_path, output, gate=None):
    dataset = json.loads(dataset_path.read_text())
    for name, fingerprint in dataset["corpus"].items():
        if digest(ROOT / "data" / name) != fingerprint["sha256"]:
            raise ValueError(f"Corpus changed: {name}; review and version the dataset")
    config = RAGConfig(_env_file=None, groq_api_key=None, langfuse_enabled=False)
    start = perf_counter()
    # Chroma may migrate its SQLite file on open: never mutate the shipped index.
    with tempfile.TemporaryDirectory(prefix="weldscope-benchmark-") as temporary:
        index = Path(temporary) / "index"
        shutil.copytree(ROOT / "vectorstore", index)
        store, documents = VectorStoreBuilder(config).load_vectorstore(
            str(index), corpus_directory=str(ROOT / "data")
        )
        sources = sorted({doc.metadata["source"] for doc in documents})
        retriever = HybridRetriever(store, documents, config, allowed_sources=sources)
        reranker = Reranker(config)
        initialization_ms = (perf_counter() - start) * 1000
        known_pages = {page_key(doc) for doc in documents}
        for case in dataset["cases"]:
            for anchor in case["evidence"]:
                if (anchor["source"], anchor["page"]) not in known_pages:
                    raise ValueError(f"Missing evidence page: {case['id']} {anchor}")
        rows, routing = [], []
        for case in dataset["cases"]:
            plan = plan_conversation(case["question"], case["history"])
            routing.append(
                dict(
                    id=case["id"],
                    expected=case["expected_route"],
                    actual=plan.kind,
                    correct=plan.kind == case["expected_route"],
                )
            )
            if not case["answerable"]:
                continue
            query = plan.retrieval_query or case["question"]
            for variant in ("bm25", "vector", "hybrid", "reranked"):
                began = perf_counter()
                if variant == "bm25":
                    selected = retriever.retrieve_bm25_only(query, k=5)
                elif variant == "vector":
                    selected = retriever.retrieve_vector_only(query, k=5)
                elif variant == "hybrid":
                    selected = retriever.retrieve(query, k=5)
                else:
                    selected = reranker.rerank(query, retriever.retrieve(query, k=20), top_k=5)
                row = dict(
                    id=case["id"],
                    category=case["category"],
                    variant=variant,
                    question=case["question"],
                    retrieval_query=query,
                    latency_ms=(perf_counter() - began) * 1000,
                    **anchor_metrics(selected, case["evidence"]),
                )
                row["retrieved"] = [
                    dict(id=doc.id, source=page_key(doc)[0], page=page_key(doc)[1])
                    for doc in selected
                ]
                rows.append(row)
            print(case["id"], flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = {}
    for variant, results in grouped.items():
        latency = sorted(row["latency_ms"] for row in results)
        summary[variant] = {
            **{
                metric: mean(row[metric] for row in results) for metric in ("recall", "mrr", "ndcg")
            },
            "mean_latency_ms": mean(latency),
            "observed_p95_latency_ms": latency[min(len(latency) - 1, int(0.95 * len(latency)))],
            "cases": len(results),
        }
    provenance = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_sha256=digest(dataset_path),
        index_sha256=digest(ROOT / "vectorstore/chroma.sqlite3"),
        python=platform.python_version(),
        platform=platform.platform(),
        cpu=platform.processor(),
        logical_cpus=os.cpu_count(),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        source_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        source_hashes={
            str(p.relative_to(ROOT)): digest(p)
            for p in [Path(__file__).resolve(), *sorted((ROOT / "weldscope").rglob("*.py"))]
        },
        packages={
            name: importlib.metadata.version(name)
            for name in ["bm25s", "chromadb", "sentence-transformers", "torch", "langchain-chroma"]
        },
    )
    report = dict(
        provenance=provenance,
        protocol=dict(
            k=5,
            rerank_candidate_k=20,
            labels="provisional positive physical-PDF-page anchors; duplicate-page gains suppressed",
            independent_expert_review=False,
            held_out=False,
            model_provider_calls=0,
            runtime_rerank_depth=config.rerank_top_k,
            runtime_candidate_k=config.retrieval_k,
            note="Reranked variant uses a 20-candidate research ablation; production uses 5 candidates -> 3 passages. Latency includes first-query caches/model loading where incurred and is not randomized or a hosted SLA.",
        ),
        corpus_chunks=len(documents),
        excluded_stale_chunks=2165 - len(documents),
        categories=dict(Counter(case["category"] for case in dataset["cases"])),
        initialization_ms=initialization_ms,
        routing_accuracy=mean(row["correct"] for row in routing),
        routing=routing,
        summary=summary,
        results=rows,
    )
    failures = []
    if gate:
        thresholds = json.loads(gate.read_text())
        failures = regression_failures(
            summary, report["routing_accuracy"], provenance["dataset_sha256"], thresholds
        )
    report["gate"] = dict(evaluated=bool(gate), passed=not failures, failures=failures)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "retrieved"])
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "retrieved"} for row in rows
        )
    print(
        json.dumps(
            dict(summary=summary, routing_accuracy=report["routing_accuracy"], gate=report["gate"]),
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/fsw-v1/questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/retrieval")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    run(args.dataset, args.output, args.gate)


if __name__ == "__main__":
    main()
