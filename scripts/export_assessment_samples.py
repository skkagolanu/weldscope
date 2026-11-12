"""Generate the versioned assessment samples with the real application pipeline."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from time import sleep
from weldscope.core.config import RAGConfig
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.pipeline.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "benchmarks/fsw-v1/questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/assessment-samples.json")
    parser.add_argument("--limit", type=int, choices=range(1, 51), default=50)
    parser.add_argument("--request-interval", type=float, default=2.1)
    args = parser.parse_args()
    if args.request_interval < 0:
        parser.error("Request interval must be non-negative")
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "Configure GROQ_API_KEY privately for offline answer generation; no samples were generated."
        )
    dataset = json.loads(args.dataset.read_text())
    for name, data in dataset["corpus"].items():
        if hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest() != data["sha256"]:
            raise SystemExit("Corpus fingerprints changed; review the dataset before assessment.")
    config = RAGConfig(_env_file=None, langfuse_enabled=False)
    payload = dict(
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        generation_model=config.generation_model,
        samples=[],
        failures=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weldscope-answer-samples-") as directory:
        copied = Path(directory) / "index"
        shutil.copytree(ROOT / "vectorstore", copied)
        store, docs = VectorStoreBuilder(config).load_vectorstore(
            str(copied), corpus_directory=str(ROOT / "data")
        )
        retriever = HybridRetriever(
            store, docs, config, allowed_sources=[doc.metadata["source"] for doc in docs]
        )
        pipeline = RAGPipeline(retriever, config)
        try:
            for case in dataset["cases"][: args.limit]:
                try:
                    response = pipeline.query(case["question"], history=case["history"])
                    payload["samples"].append(
                        dict(
                            id=case["id"],
                            question=case["question"],
                            response=response.answer,
                            reference=case["reference_answer"] if case["answerable"] else None,
                            answerable=case["answerable"],
                            contexts=[source.chunk_text for source in response.sources],
                            route=response.metadata["route"],
                            request_id=response.metadata["request_id"],
                            token_usage=response.metadata["token_usage"],
                        )
                    )
                except Exception as error:
                    payload["failures"].append(dict(id=case["id"], error=type(error).__name__))
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                sleep(args.request_interval)
        finally:
            pipeline.generator.close()
    if payload["failures"]:
        raise SystemExit("Some samples failed; inspect sanitized failure classes in the artifact.")


if __name__ == "__main__":
    main()
