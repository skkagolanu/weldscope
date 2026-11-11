# Architecture

## System Overview

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
┌────────▼────────┐
│  RAG Pipeline   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│Hybrid│  │Reranker│
│Retriever│  └───┬───┘
└───┬──┘      │
    │         │
┌───▼─────────▼───┐
│  LLM Generator   │
└──────────────────┘
```

## Module Structure

### Core (`weldscope/core/`)
- **config.py**: Pydantic configuration with validation
- **models.py**: Data models (Source, RAGResponse)
- **exceptions.py**: Custom exception hierarchy

### Retrieval (`weldscope/retrieval/`)
- **retriever.py**: Hybrid retrieval (BM25 + vector)
- **reranker.py**: Cross-encoder reranking

### Generation (`weldscope/generation/`)
- **generator.py**: LLM generation with streaming
- **formatter.py**: Response formatting with citations

### Pipeline (`weldscope/pipeline/`)
- **pipeline.py**: End-to-end RAG orchestration

### Ingestion (`weldscope/ingestion/`)
- **builder.py**: Document loading and vector store creation

### Observability (`weldscope/observability/`)
- **tracer.py**: Langfuse integration for tracing

### Evaluation (`weldscope/evaluation/`)
- **evaluator.py**: Explicit model-judged grounded answer evaluation

## Data Flow

1. **Query Input** → User submits question via Streamlit UI
2. **Retrieval** → Hybrid retriever fetches relevant documents
3. **Reranking** → Cross-encoder scores and reorders results
4. **Generation** → LLM generates answer with context
5. **Formatting** → Response formatted with source citations
6. **Display** → Answer and sources shown in UI

## Key Design Decisions

### Hybrid Retrieval
Combines BM25 (keyword) and vector (semantic) search for better recall.

### Cross-Encoder Reranking
Improves precision by scoring query-document pairs.

### Streaming Responses
Provides real-time feedback for better UX.

### Modular Architecture
Each component is independently testable and replaceable.

### Optional Observability
Langfuse tracing can be enabled/disabled without code changes.
