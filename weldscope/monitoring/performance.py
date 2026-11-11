"""Performance monitoring for RAG system."""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single query."""

    query_id: str
    timestamp: datetime
    retrieval_latency_ms: float
    reranking_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    retrieved_docs: int
    reranked_docs: int
    tokens_generated: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor and track RAG system performance."""

    def __init__(self, max_latency_ms: int = 2000):
        self.max_latency_ms = max_latency_ms
        self.metrics_history: List[PerformanceMetrics] = []

    def start_timer(self) -> float:
        """Start performance timer."""
        return time.perf_counter()

    def end_timer(self, start_time: float) -> float:
        """End timer and return elapsed milliseconds."""
        return (time.perf_counter() - start_time) * 1000

    def record_metrics(
        self,
        query_id: str,
        retrieval_latency_ms: float,
        reranking_latency_ms: float,
        generation_latency_ms: float,
        retrieved_docs: int,
        reranked_docs: int,
        tokens_generated: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceMetrics:
        """Record performance metrics for a query."""
        total_latency = retrieval_latency_ms + reranking_latency_ms + generation_latency_ms

        metrics = PerformanceMetrics(
            query_id=query_id,
            timestamp=datetime.now(),
            retrieval_latency_ms=retrieval_latency_ms,
            reranking_latency_ms=reranking_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency,
            retrieved_docs=retrieved_docs,
            reranked_docs=reranked_docs,
            tokens_generated=tokens_generated,
            metadata=metadata or {},
        )

        self.metrics_history.append(metrics)
        return metrics

    def is_slow_query(self, metrics: PerformanceMetrics) -> bool:
        """Check if query exceeded latency threshold."""
        return metrics.total_latency_ms > self.max_latency_ms

    def get_average_latency(self) -> float:
        """Get average total latency across all queries."""
        if not self.metrics_history:
            return 0.0
        return sum(m.total_latency_ms for m in self.metrics_history) / len(self.metrics_history)

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.metrics_history:
            return 0.0

        latencies = sorted(m.total_latency_ms for m in self.metrics_history)
        idx = int(len(latencies) * 0.95)
        return latencies[idx] if idx < len(latencies) else latencies[-1]

    def get_slow_queries(self) -> List[PerformanceMetrics]:
        """Get all queries that exceeded latency threshold."""
        return [m for m in self.metrics_history if self.is_slow_query(m)]

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "slow_queries": 0,
                "avg_retrieved_docs": 0.0,
                "avg_reranked_docs": 0.0,
            }

        return {
            "total_queries": len(self.metrics_history),
            "avg_latency_ms": self.get_average_latency(),
            "p95_latency_ms": self.get_p95_latency(),
            "slow_queries": len(self.get_slow_queries()),
            "avg_retrieved_docs": sum(m.retrieved_docs for m in self.metrics_history)
            / len(self.metrics_history),
            "avg_reranked_docs": sum(m.reranked_docs for m in self.metrics_history)
            / len(self.metrics_history),
            "avg_retrieval_latency_ms": sum(m.retrieval_latency_ms for m in self.metrics_history)
            / len(self.metrics_history),
            "avg_reranking_latency_ms": sum(m.reranking_latency_ms for m in self.metrics_history)
            / len(self.metrics_history),
            "avg_generation_latency_ms": sum(m.generation_latency_ms for m in self.metrics_history)
            / len(self.metrics_history),
        }

    def clear_history(self):
        """Clear metrics history."""
        self.metrics_history.clear()
