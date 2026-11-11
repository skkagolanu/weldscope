"""Test performance monitor."""

import pytest
import time
from datetime import datetime
from weldscope.monitoring.performance import PerformanceMonitor, PerformanceMetrics


@pytest.fixture
def monitor():
    """Create monitor instance."""
    return PerformanceMonitor(max_latency_ms=2000)


def test_monitor_initialization(monitor):
    """Monitor initializes with threshold."""
    assert monitor.max_latency_ms == 2000
    assert len(monitor.metrics_history) == 0


def test_start_end_timer(monitor):
    """Timer measures elapsed time."""
    start = monitor.start_timer()
    time.sleep(0.01)
    elapsed = monitor.end_timer(start)

    assert elapsed >= 10.0
    assert elapsed < 50.0


def test_record_metrics(monitor):
    """Record metrics creates PerformanceMetrics."""
    metrics = monitor.record_metrics(
        query_id="test-1",
        retrieval_latency_ms=100.0,
        reranking_latency_ms=50.0,
        generation_latency_ms=200.0,
        retrieved_docs=5,
        reranked_docs=3,
        tokens_generated=100,
    )

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.query_id == "test-1"
    assert metrics.total_latency_ms == 350.0
    assert metrics.retrieved_docs == 5
    assert metrics.reranked_docs == 3
    assert len(monitor.metrics_history) == 1


def test_is_slow_query(monitor):
    """Identifies slow queries."""
    fast_metrics = PerformanceMetrics(
        query_id="fast",
        timestamp=datetime.now(),
        retrieval_latency_ms=100.0,
        reranking_latency_ms=50.0,
        generation_latency_ms=200.0,
        total_latency_ms=350.0,
        retrieved_docs=5,
        reranked_docs=3,
        tokens_generated=100,
    )

    slow_metrics = PerformanceMetrics(
        query_id="slow",
        timestamp=datetime.now(),
        retrieval_latency_ms=1000.0,
        reranking_latency_ms=500.0,
        generation_latency_ms=1000.0,
        total_latency_ms=2500.0,
        retrieved_docs=5,
        reranked_docs=3,
        tokens_generated=100,
    )

    assert not monitor.is_slow_query(fast_metrics)
    assert monitor.is_slow_query(slow_metrics)


def test_get_average_latency(monitor):
    """Calculates average latency."""
    monitor.record_metrics("q1", 100.0, 50.0, 200.0, 5, 3)  # 350
    monitor.record_metrics("q2", 150.0, 75.0, 250.0, 5, 3)  # 475

    avg = monitor.get_average_latency()
    assert avg == pytest.approx(412.5)  # (350 + 475) / 2


def test_get_average_latency_empty(monitor):
    """Returns 0 for empty history."""
    assert monitor.get_average_latency() == 0.0


def test_get_p95_latency(monitor):
    """Calculates 95th percentile latency."""
    for i in range(100):
        monitor.record_metrics(f"q{i}", 100.0, 50.0, 200.0, 5, 3)

    monitor.record_metrics("slow", 1000.0, 500.0, 1000.0, 5, 3)

    p95 = monitor.get_p95_latency()
    assert p95 >= 350.0


def test_get_slow_queries(monitor):
    """Returns queries exceeding threshold."""
    monitor.record_metrics("fast1", 100.0, 50.0, 200.0, 5, 3)
    monitor.record_metrics("slow1", 1000.0, 500.0, 1000.0, 5, 3)
    monitor.record_metrics("fast2", 150.0, 75.0, 250.0, 5, 3)
    monitor.record_metrics("slow2", 1200.0, 600.0, 1200.0, 5, 3)

    slow = monitor.get_slow_queries()
    assert len(slow) == 2
    assert all(m.total_latency_ms > 2000 for m in slow)


def test_get_summary(monitor):
    """Returns summary statistics."""
    monitor.record_metrics("q1", 100.0, 50.0, 200.0, 5, 3)  # 350
    monitor.record_metrics("q2", 150.0, 75.0, 250.0, 6, 4)  # 475

    summary = monitor.get_summary()

    assert summary["total_queries"] == 2
    assert summary["avg_latency_ms"] == pytest.approx(412.5)  # (350 + 475) / 2
    assert summary["slow_queries"] == 0
    assert summary["avg_retrieved_docs"] == 5.5
    assert summary["avg_reranked_docs"] == 3.5


def test_get_summary_empty(monitor):
    """Returns zero summary for empty history."""
    summary = monitor.get_summary()

    assert summary["total_queries"] == 0
    assert summary["avg_latency_ms"] == 0.0
    assert summary["slow_queries"] == 0


def test_clear_history(monitor):
    """Clears metrics history."""
    monitor.record_metrics("q1", 100.0, 50.0, 200.0, 5, 3)
    monitor.record_metrics("q2", 150.0, 75.0, 250.0, 5, 3)

    assert len(monitor.metrics_history) == 2

    monitor.clear_history()

    assert len(monitor.metrics_history) == 0


def test_record_metrics_with_metadata(monitor):
    """Records metrics with custom metadata."""
    metrics = monitor.record_metrics(
        query_id="test",
        retrieval_latency_ms=100.0,
        reranking_latency_ms=50.0,
        generation_latency_ms=200.0,
        retrieved_docs=5,
        reranked_docs=3,
        metadata={"user_id": "123", "session": "abc"},
    )

    assert metrics.metadata["user_id"] == "123"
    assert metrics.metadata["session"] == "abc"
