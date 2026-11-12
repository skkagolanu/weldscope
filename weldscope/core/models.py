"""Data models for RAG responses."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Source:
    """Source document citation."""

    doc_name: str
    page_num: Optional[int]
    chunk_text: str
    fusion_score: Optional[float]
    metadata: Dict[str, Any]
    rerank_score: Optional[float] = None
    chunk_id: Optional[str] = None


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""

    query: str
    answer: str
    sources: List[Source]
    confidence: Optional[float] = None
    faithfulness_score: Optional[float] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
