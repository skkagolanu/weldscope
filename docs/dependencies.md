# Dependency maintenance

The project targets **Python 3.12 only**. Direct requirements and package metadata agree, and `constraints.txt` pins the resolved runtime and tooling dependencies. CPU PyTorch is used for Streamlit and Docker.

The tested stack includes Streamlit 1.67.0, pandas 3.1.1, Sentence Transformers 6.3.0, Transformers 5.22.2, PyTorch 2.15.0 CPU, Hugging Face Hub 2.4.0, ChromaDB 1.5.9, PyPDF 7.0.1, Groq 1.7.0 and Langfuse 5.2.0. Sphinx 9.1.1 is compatible with Python 3.12; Sphinx 9.2 requires Python 3.13 and is not selected.

The deprecated LangChain Community/Classic path, RAGAS, `rank-bm25`, unused Ollama integration, and HF generation wrapper were removed. Supported LangChain Core document/embedding interfaces, Chroma integration and text splitters remain. PDF loading uses PyPDF directly, embeddings use Sentence Transformers directly, BM25S supplies lexical search, and Groq supplies generation. Hugging Face Hub can now use its current major version. Current quality assessment uses RAGAS in a separate environment; the old custom model judge has been retired. No RAGAS import is present in the application runtime.

Development dependency auditing uses `pip-audit`; no Safety CLI compatibility backtracking is required. The resolver and `pip check` validate declared dependency compatibility. Real model loading, retrieval and reranking validate the application path beyond dependency declarations. These checks are not a guarantee of future upstream support or answer quality.

When updating, resolve the full direct package set together under Python 3.12, pin the resulting transitive versions, and run the tests, real index smoke test, docs build and browser checks. Do not change package pins in isolation or rebuild the shipped index merely to update libraries.

Sources: [Streamlit release notes](https://docs.streamlit.io/develop/quick-reference/release-notes), [Sentence Transformers installation](https://sbert.net/docs/installation.html), [Langfuse SDK migration](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v4-to-v5), [BM25S](https://bm25s.github.io/), and package metadata from [PyPI](https://pypi.org/).

## Offline evaluation dependency decision

The owner subsequently explicitly requested RAGAS and approved an offline-only exception to the earlier no-legacy rule. The active assessment environment is locked separately in `evaluation/requirements.lock`. RAGAS 0.4.3 requires Community; its latest 0.4.2 dependency fails due to removed VertexAI imports, so the tested **Community 0.4.1** combination is pinned only for assessment. Runtime requirements, constraints and package metadata are unchanged. [Assessment setup and API boundaries](https://github.com/skkagolanu/weldscope/blob/main/evaluation/README.md).
