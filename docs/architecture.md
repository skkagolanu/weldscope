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

### Evaluation (`evaluation/`)
- Native RAGAS metrics and quota-aware Groq judging in an isolated offline environment.

## Data Flow

1. **Query Input** → User submits question via Streamlit UI
2. **Retrieval** → Hybrid retriever fetches relevant documents
3. **Reranking** → Cross-encoder scores and reorders results
4. **Generation** → LLM generates answer with context
5. **Formatting** → Response formatted with source citations
6. **Display** → Answer and sources shown in UI

## Key Design Decisions

### Hybrid Retrieval
Combines BM25S keyword search and exact local vector search over bundled ONNX embeddings. Development-set comparisons measure each configuration; hybrid is not assumed to always improve recall.

### Cross-Encoder Reranking
Scores query-document pairs locally with a bundled ONNX cross-encoder. Production retains five candidates and three passages pending hosted cost/quality evidence.

### Streaming Responses
Provider chunks are accumulated privately. Research output is delivered only after structural citation validation; provider first-token and first-visible timings remain separate.

### Modular Architecture
Each component is independently testable and replaceable.

### Optional Observability
Langfuse tracing can be enabled/disabled without code changes.
