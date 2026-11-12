"""Exercise bundled inference and index integrity without provider/network substitutes."""

import importlib.metadata
import json
import socket
from pathlib import Path

import numpy as np
import pytest
from langchain_core.documents import Document

from weldscope.core.local_models import LocalModel
from weldscope.ingestion.store import NumpyVectorStore


def test_runtime_distribution_has_no_hf_dependency():
    installed = {
        item.metadata["Name"].lower().replace("_", "-")
        for item in importlib.metadata.distributions()
    }
    assert not installed.intersection(
        {
            "huggingface-hub",
            "transformers",
            "sentence-transformers",
            "tokenizers",
            "datasets",
            "torch",
            "chromadb",
            "ragas",
        }
    )


def test_real_local_inference_needs_no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Local inference attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    embedding = LocalModel("embedding")
    tokens = embedding.tokens("What is FSW?")
    assert tokens["input_ids"].tolist() == [[101, 2054, 2003, 1042, 26760, 1029, 102]]
    vectors = embedding.encode(["What is FSW?", "What is FSW?"])
    assert vectors.shape == (2, 384)
    np.testing.assert_array_equal(vectors[0], vectors[1])
    reranker = LocalModel("reranker")
    scores = reranker.predict(
        [
            (
                "What is friction stir welding?",
                "Friction stir welding uses a rotating tool to join metals in the solid state.",
            ),
            ("What is friction stir welding?", "Butter chicken is a recipe."),
        ]
    )
    assert np.isfinite(scores).all()
    assert scores[0] > scores[1]
    truncated = reranker.tokens("word " * 700, "word " * 900)
    assert truncated["input_ids"].shape == (1, 512)
    assert int((truncated["token_type_ids"] == 0).sum()) == 256
    assert truncated["input_ids"][0, -1] == 102


def test_store_preserves_identity_filters_and_exact_distance(tmp_path):
    class Embeddings:
        def embed_documents(self, texts):
            return [[0, 0], [1, 0], [2, 0]]

        def embed_query(self, text):
            return [0, 0]

    docs = [
        Document(page_content="Same text", metadata={"source": "a.pdf", "page": page})
        for page in range(3)
    ]
    store = NumpyVectorStore.from_documents(docs, ["one", "two", "three"], Embeddings(), tmp_path)
    assert [d.id for d in store.similarity_search("question", k=2)] == ["one", "two"]
    assert [
        d.id
        for d in store.similarity_search(
            "question", filter={"$and": [{"source": {"$in": ["a.pdf"]}}, {"page": 2}]}
        )
    ] == ["three"]
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {"chunks.json": "invalid"}}))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        NumpyVectorStore(tmp_path, Embeddings())


def test_bundled_index_has_only_current_pdf_passages():
    path = Path("indexes/fsw-v2")
    rows = json.loads((path / "chunks.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(rows) == manifest["chunks"] == 2124
    assert len({row["id"] for row in rows}) == len(rows)
    assert {Path(row["metadata"]["source"]).name for row in rows} == set(manifest["corpus"])
