# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-11-11

- Require Python 3.12 across package metadata, local configuration, Docker, dev container and CI.
- Update the maintained runtime/tooling stack with resolved dependency constraints.
- Replace deprecated Community/Classic and RAGAS imports, rank-bm25 and unused provider integrations.
- Use direct PyPDF and Sentence Transformers APIs, BM25S, and Groq generation; keep the bundled Chroma index.
- Preserve stored chunk IDs and assign deterministic IDs to newly ingested chunks.
- Separate fusion/reranking score labels and leave unmeasured confidence unknown.
- Connect per-stage timing, reported token usage and error types to the session performance monitor.
- Validate real SDK trace correlation locally with an in-memory exporter.


- Add Groq generation with GPT-OSS 20B, explicit provider selection, and friendly billing/rate-limit errors.
- Route greetings and app/source questions locally without retrieving documents or requiring inference credentials.
- Keep bounded completed conversation history separate from focused document search queries.
- Resolve research follow-ups against the active topic, omit failed exchanges, and include sensor data only when requested.
- Update hosted URLs and health checks for https://weldscope.streamlit.app/.


- Rebrand the repository, package, Streamlit headings, examples, and docs as WeldScope.
- Deliver generation chunks to the UI as they arrive and preserve feedback across reruns.
- Repair optional Langfuse SDK calls and prevent duplicating the current question in history.
- Fix the package README, explicit dependencies, Docker assets, health checks, and writable index mount.
- Use CPU-only PyTorch wheels for the hosted demo.
- Replace permissive HTTP checks with rendered-chat availability checks and failure artifacts.
- Add Streamlit regression coverage and document the observed hosting outage and recovery steps.
- Remove stale generated documentation; rebuild from the maintained sources.

## [1.1.0] - 2026-02-04

### Added
- Conversation memory with context-aware multi-turn conversations (last 10 turns)
- Memory retention counter in sidebar showing remaining context window
- Visual indicators for memory status (active/warning)
- Sensor data integration for real-time FSW process monitoring
- Sensor-based defect correlation and analysis
- Session management with temporary in-memory storage
- Clear conversation button for privacy
- Session ID tracking for debugging
- Message and question counters

### Changed
- Updated UI to display conversation history
- Enhanced query augmentation with conversation context
- Added sensor data summary in relevant queries
- Improved sidebar with session information and memory metrics

### Privacy
- All conversation data stored in-memory only (Streamlit session_state)
- No persistent storage of user conversations
- Data automatically deleted when browser session ends
- No user tracking or data collection

## [1.0.1] - 2026-02-04

### Fixed
- Updated LLM generator to use ChatHuggingFace wrapper for HuggingFace Inference API compatibility
- Fixed "not supported for task text-generation" error by using conversational task format
- Added rank-bm25 dependency for hybrid retrieval BM25 component

### Changed
- Generator now wraps HuggingFaceEndpoint with ChatHuggingFace for proper chat format
- Updated tests to mock ChatHuggingFace wrapper correctly
- Improved test coverage to 96.19%

## [1.0.0] - 2026-02-04

### Added

#### Core RAG System
- Modular RAG architecture with clean separation of concerns
- Hybrid retrieval combining BM25 and vector search
- Cross-encoder reranking for improved relevance
- Streaming LLM responses via HuggingFace API
- Source attribution with citation tracking and relevance scores
- Langfuse observability integration for tracing and monitoring
- Vector store builder for PDF document ingestion
- Streamlit web interface for interactive queries

#### Enterprise Features
- RAGAS evaluation framework (faithfulness, relevancy, precision, recall)
- Performance monitoring with latency tracking and P95 metrics
- Query/response caching with semantic similarity matching
- Comprehensive API documentation with Sphinx
- Docker containerization with multi-stage builds
- Docker Compose for local development

#### Quality Infrastructure
- CI/CD pipeline with GitHub Actions
- Automated testing with 95%+ code coverage
- Code quality gates (Black, Bandit, coverage thresholds)
- Pre-commit hooks for local validation
- Integration and unit test suites

### Technical Specifications
- Python 3.11+ support
- 105 passing tests
- 95.66% test coverage
- Pinned dependency versions
- Security scanning with Bandit
- Type hints throughout codebase

### Documentation
- Production-ready README with quickstart guide
- Architecture documentation
- Deployment guides (local, Docker, cloud)
- API reference with code examples
- CI/CD workflow documentation
