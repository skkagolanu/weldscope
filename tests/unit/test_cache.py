"""Test query cache."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime, timedelta

from weldscope.caching.cache import QueryCache, CacheEntry
from weldscope.core.models import RAGResponse, Source


@pytest.fixture
def cache():
    """Create cache instance."""
    return QueryCache(similarity_threshold=0.95, ttl_seconds=3600)


@pytest.fixture
def sample_response():
    """Create sample RAG response."""
    return RAGResponse(
        query="Test query",
        answer="Test answer",
        sources=[
            Source(
                doc_name="test.pdf",
                page_num=1,
                chunk_text="Test content",
                relevance_score=0.9,
                metadata={},
            )
        ],
    )


def test_cache_initialization(cache):
    """Cache initializes with config."""
    assert cache.similarity_threshold == 0.95
    assert cache.ttl_seconds == 3600
    assert len(cache.cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0


@patch("weldscope.caching.cache.SentenceTransformer")
def test_compute_embedding(mock_transformer, cache):
    """Computes embedding for text."""
    mock_model = Mock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer.return_value = mock_model

    embedding = cache._compute_embedding("test query")

    assert embedding == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once_with("test query")


def test_compute_similarity(cache):
    """Computes cosine similarity."""
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [1.0, 0.0, 0.0]
    emb3 = [0.0, 1.0, 0.0]

    assert cache._compute_similarity(emb1, emb2) == pytest.approx(1.0)
    assert cache._compute_similarity(emb1, emb3) == pytest.approx(0.0)


def test_generate_key(cache):
    """Generates consistent hash key."""
    key1 = cache._generate_key("test query")
    key2 = cache._generate_key("test query")
    key3 = cache._generate_key("different query")

    assert key1 == key2
    assert key1 != key3
    assert len(key1) == 64  # SHA256 hex


def test_is_expired(cache):
    """Checks if entry is expired."""
    fresh_entry = CacheEntry(query="test", response=Mock(), embedding=[], timestamp=datetime.now())

    old_entry = CacheEntry(
        query="test",
        response=Mock(),
        embedding=[],
        timestamp=datetime.now() - timedelta(seconds=7200),
    )

    assert not cache._is_expired(fresh_entry)
    assert cache._is_expired(old_entry)


@patch("weldscope.caching.cache.SentenceTransformer")
def test_put_and_get_exact_match(mock_transformer, cache, sample_response):
    """Caches and retrieves exact match."""
    mock_model = Mock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer.return_value = mock_model

    cache.put("test query", sample_response)

    result = cache.get("test query")

    assert result is not None
    assert result.answer == "Test answer"
    assert cache.hits == 1
    assert cache.misses == 0


@patch("weldscope.caching.cache.SentenceTransformer")
def test_get_cache_miss(mock_transformer, cache):
    """Returns None on cache miss."""
    mock_model = Mock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer.return_value = mock_model

    result = cache.get("nonexistent query")

    assert result is None
    assert cache.hits == 0
    assert cache.misses == 1


@patch("weldscope.caching.cache.SentenceTransformer")
def test_get_similar_query(mock_transformer, cache, sample_response):
    """Retrieves response for similar query."""
    mock_model = Mock()

    # First call for put
    mock_model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),  # put
        np.array([0.96, 0.0, 0.0]),  # get (similar)
    ]
    mock_transformer.return_value = mock_model

    cache.put("what causes defects", sample_response)
    result = cache.get("what causes defects?")

    assert result is not None
    assert cache.hits == 1


@patch("weldscope.caching.cache.SentenceTransformer")
def test_get_dissimilar_query(mock_transformer, cache, sample_response):
    """Returns None for dissimilar query."""
    mock_model = Mock()

    mock_model.encode.side_effect = [
        np.array([1.0, 0.0, 0.0]),  # put
        np.array([0.0, 1.0, 0.0]),  # get (dissimilar)
    ]
    mock_transformer.return_value = mock_model

    cache.put("what causes defects", sample_response)
    result = cache.get("completely different question")

    assert result is None
    assert cache.misses == 1


def test_evict_oldest(cache, sample_response):
    """Evicts oldest entry when max reached."""
    cache.max_entries = 2

    with patch.object(cache, "_compute_embedding", return_value=[0.1, 0.2]):
        cache.put("query1", sample_response)
        cache.put("query2", sample_response)

        assert len(cache.cache) == 2

        cache.put("query3", sample_response)

        assert len(cache.cache) == 2


def test_clear(cache, sample_response):
    """Clears all cache entries and stats."""
    with patch.object(cache, "_compute_embedding", return_value=[0.1, 0.2]):
        cache.put("query1", sample_response)
        cache.put("query2", sample_response)

    cache.hits = 5
    cache.misses = 3

    cache.clear()

    assert len(cache.cache) == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_get_stats(cache):
    """Returns cache statistics."""
    cache.hits = 8
    cache.misses = 2
    cache.cache = {"key1": Mock(), "key2": Mock()}

    stats = cache.get_stats()

    assert stats["total_entries"] == 2
    assert stats["hits"] == 8
    assert stats["misses"] == 2
    assert stats["hit_rate"] == 0.8
    assert stats["total_requests"] == 10


def test_get_stats_empty(cache):
    """Returns zero stats for empty cache."""
    stats = cache.get_stats()

    assert stats["total_entries"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0.0
    assert stats["total_requests"] == 0


@patch("weldscope.caching.cache.SentenceTransformer")
def test_expired_entries_removed_on_get(mock_transformer, cache, sample_response):
    """Expired entries are removed during get."""
    mock_model = Mock()
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_transformer.return_value = mock_model

    cache.put("test query", sample_response)

    # Manually expire the entry
    key = list(cache.cache.keys())[0]
    cache.cache[key].timestamp = datetime.now() - timedelta(seconds=7200)

    result = cache.get("test query")

    assert result is None
    assert len(cache.cache) == 0
