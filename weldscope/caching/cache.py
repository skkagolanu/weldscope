"""Query/response caching layer."""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
from weldscope.core.local_models import LocalModel

from weldscope.core.models import RAGResponse


@dataclass
class CacheEntry:
    """Cached query/response entry."""

    query: str
    response: RAGResponse
    embedding: List[float]
    timestamp: datetime
    hit_count: int = 0


class QueryCache:
    """Cache for query/response pairs with similarity matching."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
        embedding_model: str = "embedding",
    ):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        self._embedder: Optional[LocalModel] = None
        self.embedding_model = embedding_model

    def _get_embedder(self) -> LocalModel:
        """Lazy load embedding model."""
        if self._embedder is None:
            self._embedder = LocalModel(self.embedding_model)
        return self._embedder

    def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding for text."""
        embedder = self._get_embedder()
        return [float(value) for value in embedder.encode(text).tolist()]

    def _compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute cosine similarity between embeddings."""
        import numpy as np

        emb1_arr = np.array(emb1)
        emb2_arr = np.array(emb2)

        dot_product = np.dot(emb1_arr, emb2_arr)
        norm1 = np.linalg.norm(emb1_arr)
        norm2 = np.linalg.norm(emb2_arr)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _generate_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.sha256(query.encode()).hexdigest()

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        age = datetime.now() - entry.timestamp
        return age.total_seconds() > self.ttl_seconds

    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
        del self.cache[oldest_key]

    def get(self, query: str) -> Optional[RAGResponse]:
        """Get cached response for similar query."""
        query_embedding = self._compute_embedding(query)

        best_match: Optional[Tuple[str, float]] = None
        best_similarity = 0.0

        for key, entry in list(self.cache.items()):
            if self._is_expired(entry):
                del self.cache[key]
                continue

            similarity = self._compute_similarity(query_embedding, entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (key, similarity)

        if best_match and best_similarity >= self.similarity_threshold:
            key = best_match[0]
            self.cache[key].hit_count += 1
            self.hits += 1
            return self.cache[key].response

        self.misses += 1
        return None

    def put(self, query: str, response: RAGResponse):
        """Cache query/response pair."""
        if len(self.cache) >= self.max_entries:
            self._evict_oldest()

        key = self._generate_key(query)
        embedding = self._compute_embedding(query)

        entry = CacheEntry(
            query=query, response=response, embedding=embedding, timestamp=datetime.now()
        )

        self.cache[key] = entry

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "total_entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }
