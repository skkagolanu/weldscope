"""Exact local vector search over a versioned JSON/NumPy index; no database service."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from langchain_core.documents import Document


class NumpyVectorStore:
    def __init__(self, persist_directory, embedding_function):
        self.path = Path(persist_directory)
        self.embeddings = embedding_function
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for name, expected in manifest["files"].items():
                if sha256((self.path / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Index fingerprint mismatch: {name}")
        self.rows = json.loads((self.path / "chunks.json").read_text())
        self.vectors = np.load(self.path / "vectors.npy", allow_pickle=False)
        if self.vectors.ndim != 2 or len(self.rows) != len(self.vectors):
            raise ValueError("Index dimensions do not match its passages")
        if not np.isfinite(self.vectors).all():
            raise ValueError("Index contains non-finite embeddings")

    @classmethod
    def from_documents(cls, documents, ids, embedding, persist_directory):
        path = Path(persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        rows = [
            dict(id=key, text=doc.page_content, metadata=doc.metadata)
            for key, doc in zip(ids, documents)
        ]
        (path / "chunks.json").write_text(json.dumps(rows, ensure_ascii=False))
        np.save(
            path / "vectors.npy",
            np.asarray(
                embedding.embed_documents([doc.page_content for doc in documents]), dtype=np.float32
            ),
            allow_pickle=False,
        )
        return cls(path, embedding)

    def get(self, include=None):
        return dict(
            ids=[row["id"] for row in self.rows],
            documents=[row["text"] for row in self.rows],
            metadatas=[row["metadata"] for row in self.rows],
        )

    @staticmethod
    def _matches(metadata, filters):
        if not filters:
            return True
        if "$and" in filters:
            return all(NumpyVectorStore._matches(metadata, item) for item in filters["$and"])
        return all(
            (
                metadata.get(key) in value["$in"]
                if isinstance(value, dict)
                else metadata.get(key) == value
            )
            for key, value in filters.items()
        )

    def similarity_search(self, query, k=5, filter=None):
        vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        distances = np.sum((self.vectors - vector) ** 2, axis=1)
        indices = np.argsort(distances, kind="stable")
        found = []
        for index in indices:
            row = self.rows[int(index)]
            if self._matches(row["metadata"], filter):
                found.append(
                    Document(id=row["id"], page_content=row["text"], metadata=dict(row["metadata"]))
                )
                if len(found) == k:
                    break
        return found
