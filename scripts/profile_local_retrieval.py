"""Profile the bundled runtime on two CPU cores, before numeric libraries create threads."""

import os
import time

START = time.perf_counter()
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[variable] = "2"

import argparse
import hashlib
import json
import resource
from pathlib import Path
from statistics import mean

from weldscope.conversation import plan_conversation
from weldscope.core.config import RAGConfig
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.retrieval.reranker import Reranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-profile"))
    args = parser.parse_args()
    config = RAGConfig(_env_file=None)
    dataset = Path("benchmarks/fsw-v1/questions.json")
    cases = json.loads(dataset.read_text())["cases"]
    import_ms = (time.perf_counter() - START) * 1000
    began = time.perf_counter()
    store, documents = VectorStoreBuilder(config).load_vectorstore(corpus_directory="data")
    retriever = HybridRetriever(store, documents, config)
    reranker = Reranker(config)
    initialization_ms = (time.perf_counter() - began) * 1000
    rows = []
    for case in cases:
        if not case["answerable"]:
            continue
        plan = plan_conversation(case["question"], case["history"])
        for name, pool, count in [
            ("production5to3", 5, 3),
            ("pool20to3", 20, 3),
            ("pool20to5", 20, 5),
        ]:
            began = time.perf_counter()
            timings = {}
            docs = reranker.rerank(
                plan.retrieval_query,
                retriever.retrieve(plan.retrieval_query, k=pool),
                top_k=count,
                timings=timings,
            )
            rows.append(
                dict(
                    id=case["id"],
                    variant=name,
                    category=case["category"],
                    latency_ms=(time.perf_counter() - began) * 1000,
                    timings=timings,
                    retrieved=[
                        dict(
                            id=d.id,
                            source=Path(d.metadata["source"]).name,
                            page=d.metadata["page"] + 1,
                        )
                        for d in docs
                    ],
                )
            )
    thread_affinities = [
        sorted(os.sched_getaffinity(int(path.name))) for path in Path("/proc/self/task").iterdir()
    ]
    assert all(
        set(affinity) <= os.sched_getaffinity(0) for affinity in thread_affinities
    ), "A worker escaped the configured CPU affinity"
    report = dict(
        provenance=dict(
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            index_sha256=hashlib.sha256(
                Path("indexes/fsw-v2/manifest.json").read_bytes()
            ).hexdigest(),
        ),
        results=rows,
        initialization_ms=initialization_ms,
        import_ms=import_ms,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        all_thread_affinities=thread_affinities,
        limitations="Local two-core process, one run, fixed variant order, excludes Groq generation and hosted network. First row includes reranker initialization; warm summaries omit it.",
    )
    report["cold_retrieval_ms"] = initialization_ms + rows[0]["latency_ms"]
    report["summary"] = {}
    for name in {r["variant"] for r in rows}:
        values = sorted(
            r["latency_ms"]
            for r in rows
            if r["variant"] == name and not r["timings"]["reranker_load_ms"]
        )
        report["summary"][name] = dict(
            mean_ms=mean(values),
            p95_ms=values[min(len(values) - 1, int(len(values) * 0.95))],
            samples=len(values),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rankings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["results", "all_thread_affinities"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
