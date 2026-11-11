# WeldScope

A research assistant for friction stir welding defects, causes, and corrective actions.

[Open the Streamlit app](https://rag-demo-skullkrak7.streamlit.app/) · [Hosting status](https://github.com/skkagolanu/weldscope/actions/workflows/deploy.yml)

Community Cloud may put the app to sleep. See [hosting and recovery](docs/hosting.md) for the current status and redeployment steps.

[![CI Pipeline](https://github.com/skkagolanu/weldscope/actions/workflows/ci.yml/badge.svg)](https://github.com/skkagolanu/weldscope/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-96.71%25-brightgreen)](https://github.com/skkagolanu/weldscope)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11–3.12](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/downloads/)

Retrieval-Augmented Generation (RAG) system for intelligent defect analysis in Friction Stir Welding (FSW) processes.

## Features

- Hybrid Retrieval: BM25 + vector search with cross-encoder reranking
- Streaming Responses: Real-time LLM output with HuggingFace API
- Conversation Memory: Context-aware multi-turn conversations (last 10 turns)
- Memory Counter: Visual indicator showing remaining context window
- Sensor Data Integration: Bundled sensor log summaries included in question context
- Source Attribution: Full citation tracking with relevance scores
- Observability: Langfuse integration for tracing and monitoring
- Modular Architecture: Clean separation of concerns with automated tests
- Production Ready: Error handling, validation, and graceful degradation
- Session History: In-memory conversations; optional tracing sends data to Langfuse when enabled

## Quick Start

### Installation

Use Python 3.11 or 3.12 with the pinned dependencies.

```bash
# Clone repository
git clone https://github.com/skkagolanu/weldscope.git
cd weldscope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required: HF_TOKEN
# Optional: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
```

### Build Vector Store

```bash
# Place FSW PDF documents in data/ directory
python build_vectorstore.py --pdf-dir ./data --output-dir ./vectorstore
```

### Run Application

```bash
streamlit run app.py
```

Access at `http://localhost:8501`

## Architecture

### System Components

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

### Module Structure

```
weldscope/
├── core/              # Configuration, models, exceptions
├── ingestion/         # Document loading and vectorstore building
├── retrieval/         # Hybrid retrieval and reranking
├── generation/        # LLM generation and response formatting
├── pipeline/          # End-to-end RAG orchestration
└── observability/     # Langfuse tracing integration
```

## Usage

### Basic Query

```python
from weldscope.core.config import RAGConfig
from weldscope.ingestion.builder import VectorStoreBuilder
from weldscope.retrieval.retriever import HybridRetriever
from weldscope.pipeline.pipeline import RAGPipeline

# Initialize
config = RAGConfig()
builder = VectorStoreBuilder(config)
vectorstore, documents = builder.load_vectorstore()

# Create retriever
retriever = HybridRetriever(vectorstore, documents, config)

# Create pipeline
pipeline = RAGPipeline(retriever, config)

# Query
response = pipeline.query("What causes wormhole defects in FSW?")
print(response.answer)

for source in response.sources:
    print(f"- {source.doc_name} (Page {source.page_num})")
```

### Streaming Query

```python
for chunk in pipeline.stream_query("Explain FSW process parameters"):
    print(chunk, end="", flush=True)
```

### With Observability

```python
from weldscope.observability.tracer import RAGTracer

config = RAGConfig(langfuse_enabled=True)
tracer = RAGTracer(config)
pipeline = RAGPipeline(retriever, config, tracer=tracer)

response = pipeline.query("What are common FSW defects?")

# Feedback
tracer.score_feedback(1.0, "user_feedback")
tracer.flush()
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace API token | Required |
| `MODEL_NAME` | LLM model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `TEMPERATURE` | LLM temperature | `0.05` |
| `RETRIEVAL_K` | Documents to retrieve | `5` |
| `RERANK_TOP_K` | Documents after reranking | `3` |
| `CHUNK_SIZE` | Document chunk size | `500` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |
| `LANGFUSE_ENABLED` | Enable tracing | `false` |

See `.env.example` for complete configuration options.

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v --cov=weldscope

# Specific module
pytest tests/unit/test_pipeline.py -v

# With coverage report
pytest tests/ --cov=weldscope --cov-report=html
```

### Code Quality

```bash
# Format code
black weldscope/ tests/

# Lint
pylint weldscope/

# Type checking
mypy weldscope/
```

## Project Status

### Completed Features

- Modular RAG architecture
- Hybrid retrieval (BM25 + vector)
- Cross-encoder reranking
- Streaming LLM responses
- Source attribution and citations
- Langfuse observability
- Vector store builder
- Streamlit UI
- 107 passing tests; 96.71% package coverage (Python 3.11)

### Roadmap

- RAGAS evaluation framework
- Performance monitoring
- API documentation

## Technical Specifications

### Dependencies

- **LangChain**: 1.2.8
- **Streamlit**: 1.53.1
- **ChromaDB**: 1.4.1
- **Sentence Transformers**: 5.2.2
- **Langfuse**: 3.12.1
- **PyPDF**: 6.6.2

### Models

- **Embeddings**: `sentence-transformers/paraphrase-MiniLM-L3-v2` (384 dims)
- **LLM**: `meta-llama/Llama-3.1-8B-Instruct` (8B params)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Performance

Latency depends on model downloads, hosting resources, and the inference provider.
The bundled vector store contains 2,165 chunks across five source documents; it is about 21 MB.

## License

MIT License - Copyright (c) 2026 Sai Karthik Kagolanu

## Contact

- **Author**: Sai Karthik Kagolanu
- **GitHub**: [@skkagolanu](https://github.com/skkagolanu)
- **Project**: [weldscope](https://github.com/skkagolanu/weldscope)
