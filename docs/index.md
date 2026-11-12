# WeldScope Documentation

Retrieval-Augmented Generation research assistant for Friction Stir Welding defect analysis.

## Contents

```{toctree}
:maxdepth: 2
:caption: Contents

quickstart
api/index
architecture
deployment
hosting
maintenance-2026-11-13
streamlit-review
dependencies
latency
evaluation
operations
```

## Features

- Hybrid retrieval (BM25 + vector search)
- Cross-encoder reranking
- Research output buffered for citation checks before display
- Source attribution and citations
- Langfuse observability
- Performance monitoring
- Optional cache module (not integrated into the query path)
- Model-judged evaluation

## Quick Links

- [GitHub Repository](https://github.com/skkagolanu/weldscope)
- [API Reference](api/index.md)
- [Architecture](architecture.md)
