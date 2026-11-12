"""Export controlled retrieval/index comparisons; quality scoring is delegated to RAGAS."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from time import perf_counter

import numpy as np
import resource
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from weldscope.conversation import plan_conversation
from weldscope.core.config import RAGConfig
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.retrieval.reranker import Reranker

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "l3": "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "l6": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(doc):
    return hashlib.sha256(
        (
            Path(doc.metadata["source"]).name
            + "\0"
            + str(doc.metadata["page"])
            + "\0"
            + doc.page_content
        ).encode()
    ).hexdigest()


class ExactStore:
    def __init__(self, documents, vectors, model, normalize, prefix="", metric="cosine"):
        self.documents = documents
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.model = model
        self.normalize = normalize
        self.prefix = prefix
        self.metric = metric
        self.norms = np.linalg.norm(self.vectors, axis=1)

    def similarity_search(self, query, k, filter=None):
        vector = self.model.encode(
            self.prefix + query, normalize_embeddings=self.normalize, show_progress_bar=False
        )
        if self.metric == "cosine":
            scores = (self.vectors @ vector) / (self.norms * np.linalg.norm(vector) + 1e-12)
            indices = np.argsort(-scores, kind="stable")[:k]
        else:
            indices = np.argsort(np.sum((self.vectors - vector) ** 2, axis=1), kind="stable")[:k]
        return [self.documents[int(index)] for index in indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-offset", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    affinity = sorted(os.sched_getaffinity(0))
    chosen = affinity[args.cpu_offset : args.cpu_offset + 2] or affinity[:2]
    os.sched_setaffinity(0, chosen)
    torch.set_num_threads(2)
    config = RAGConfig(_env_file=None, groq_api_key=None, telemetry_path="")
    dataset_path = ROOT / "benchmarks/fsw-v1/questions.json"
    dataset = json.loads(dataset_path.read_text())
    for name, fingerprint in dataset["corpus"].items():
        if sha(ROOT / "data" / name) != fingerprint["sha256"]:
            raise ValueError("Corpus changed")
    model_started = perf_counter()
    model = SentenceTransformer(MODELS[args.model], device="cpu")
    load_ms = (perf_counter() - model_started) * 1000
    prefix = (
        "Represent this sentence for searching relevant passages: " if args.model == "bge" else ""
    )
    reranker = Reranker(config)
    builder = VectorStoreBuilder(config)
    pages = builder.load_pdfs(str(ROOT / "data"))
    for doc in pages:
        doc.metadata["source"] = "data/" + Path(doc.metadata["source"]).name
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", doc.page_content)
        doc.page_content = re.sub(r"[ \t]+", " ", text)
    exported = []
    variants = []
    builds = []

    def evaluate(label, documents, store):
        retriever = HybridRetriever(store, documents, config)
        labels = [
            label + "/" + suffix
            for suffix in ["vector5", "hybrid5", "production5to3", "pool20to3", "pool20to5"]
        ]
        variants.extend(labels)
        for case in dataset["cases"]:
            if not case["answerable"]:
                continue
            query = plan_conversation(case["question"], case["history"]).retrieval_query
            selected = {}
            timings = {}
            start = perf_counter()
            selected["vector5"] = retriever.retrieve_vector_only(query, k=5)
            timings["vector5"] = (perf_counter() - start) * 1000
            start = perf_counter()
            selected["hybrid5"] = retriever.retrieve(query, k=5)
            timings["hybrid5"] = (perf_counter() - start) * 1000
            start = perf_counter()
            selected["production5to3"] = reranker.rerank(
                query, retriever.retrieve(query, k=5), top_k=3
            )
            timings["production5to3"] = (perf_counter() - start) * 1000
            start = perf_counter()
            large = reranker.rerank(query, retriever.retrieve(query, k=20), top_k=5)
            elapsed = (perf_counter() - start) * 1000
            selected["pool20to3"] = large[:3]
            selected["pool20to5"] = large
            timings["pool20to3"] = elapsed
            timings["pool20to5"] = elapsed
            for suffix, docs in selected.items():
                exported.append(
                    dict(
                        id=case["id"],
                        variant=label + "/" + suffix,
                        category=case["category"],
                        latency_ms=timings[suffix],
                        rss_mb=int(Path("/proc/self/statm").read_text().split()[1])
                        * os.sysconf("SC_PAGE_SIZE")
                        / 2**20,
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
        print(label, flush=True)

    if args.model == "l3":
        with tempfile.TemporaryDirectory(prefix="weldscope-existing-") as t:
            copied = Path(t) / "index"
            shutil.copytree(ROOT / "vectorstore", copied)
            original, docs = builder.load_vectorstore(
                str(copied), corpus_directory=str(ROOT / "data")
            )
            # Benchmark the actual packaged HNSW search and the exact cosine control on identical vectors.
            allowed = sorted({d.metadata["source"] for d in docs})

            class CurrentStore:
                def similarity_search(self, query, k, filter=None):
                    return original.similarity_search(
                        query, k=k, filter={"source": {"$in": allowed}}
                    )

            evaluate("existing-hnsw-l2", docs, CurrentStore())
            raw = original.get(
                ids=[d.id for d in docs], include=["embeddings", "documents", "metadatas"]
            )
            ordered = [
                Document(id=i, page_content=t, metadata=m)
                for i, t, m in zip(raw["ids"], raw["documents"], raw["metadatas"])
            ]
            evaluate(
                "existing-exact-cosine",
                ordered,
                ExactStore(ordered, raw["embeddings"], model, True),
            )
    for size, overlap in [(500, 50), (1000, 100)]:
        started = perf_counter()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        ).split_documents(pages)
        unique = {identity(doc): doc for doc in chunks}
        for identifier, doc in unique.items():
            doc.id = identifier
        chunks = list(unique.values())
        vectors = model.encode(
            [doc.page_content for doc in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
        build_ms = (perf_counter() - started) * 1000
        label = args.model + f"-clean-{size}"
        folder = args.output / label
        folder.mkdir(exist_ok=True)
        np.save(folder / "vectors.npy", vectors)
        (folder / "chunks.json").write_text(
            json.dumps([dict(id=d.id, text=d.page_content, metadata=d.metadata) for d in chunks])
        )
        token_lengths = [
            len(t)
            for t in model.tokenizer([d.page_content for d in chunks], truncation=False)[
                "input_ids"
            ]
        ]
        manifest = dict(
            version="fsw-2026-11-13-" + label,
            model=MODELS[args.model],
            normalize_embeddings=True,
            query_prefix=prefix,
            metric="cosine",
            chunk_size=size,
            chunk_overlap=overlap,
            corpus=dataset["corpus"],
            chunks=len(chunks),
            source_files=sorted({Path(d.metadata["source"]).name for d in chunks}),
            build_ms=build_ms,
            model_max_tokens=model.max_seq_length,
            truncation_fraction=sum(length > model.max_seq_length for length in token_lengths)
            / len(chunks),
        )
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        builds.append(manifest)
        evaluate(label, chunks, ExactStore(chunks, vectors, model, True, prefix))
        del vectors
        gc.collect()
    report = dict(
        provenance=dict(dataset_sha256=sha(dataset_path), experiment_sha256=sha(__file__)),
        variants=variants,
        results=exported,
        builds=builds,
        model_load_ms=load_ms,
        cpu_affinity=chosen,
        torch_threads=2,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        limitations="Local two-core constrained experiment; exact vector search for candidates, actual HNSW for packaged baseline. RSS includes model, reranker and index data; not hosted memory certification. No provider calls. Quality scores computed separately by native RAGAS.",
    )
    (args.output / "rankings.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
