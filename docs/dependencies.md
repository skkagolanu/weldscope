# Dependency maintenance

The project targets **Python 3.12 only**. Runtime and tooling versions are resolved in `constraints.txt`; direct requirements agree with `pyproject.toml`.

The application uses Streamlit 1.67.0, pandas 3.1.1, ONNX Runtime 1.31.1, ORT Extensions 0.15.2, ONNX 1.23.2, NumPy 2.5.4, BM25S 0.3.16, PyPDF 7.0.1, Groq 1.7.0 and Langfuse 5.2.0. Maintained LangChain Core document/embedding interfaces and text splitters remain. Sphinx 9.1.1 supports the required Python version.

**HF packages and services have been removed from the Streamlit runtime.** This includes Hugging Face Hub, Transformers, Sentence Transformers, tokenizers, Torch, Chroma and its LangChain integration. Chroma's mandatory tokenizer dependency would otherwise retain HF Hub. Local semantic retrieval now uses a JSON/NumPy index and the same MiniLM weights exported to bundled ONNX files. The tokenizer runs through Microsoft's ORT Extensions. Neither runtime model loading nor index loading has a network-download fallback. See [assets and provenance](https://github.com/skkagolanu/weldscope/blob/main/models/README.md).

Groq supplies generation. Its documented API has no embedding endpoint, so local embeddings and reranking are retained rather than simulated with chat completions. See [Groq API reference](https://console.groq.com/docs/api-reference).

## Explicit offline exception

The owner explicitly chose to **keep RAGAS and allow its required HF dependencies in the offline evaluator only**. `evaluation/requirements.lock` contains that separate environment; it must never be merged into app requirements or installed into Streamlit. RAGAS 0.4.3 requires datasets/HF Hub and Community; the tested Community 0.4.1 pin avoids removed VertexAI imports. This is an acknowledged compatibility exception, not a claim that those dependencies disappeared.

Even offline assessment now uses the bundled ONNX model through RAGAS's public embedding interface, instead of its HF provider. The workflow blocks hub downloads, disables RAGAS analytics, and sends judge requests only to Groq. Sentence Transformers and Torch were removed from the evaluator lock too.

A fresh application environment, installed distribution checks, network-denied real inference tests, `pip check`, retrieval regression, software tests, mypy and error-level Pylint verify these boundaries. They do not guarantee future upstream support, answer correctness or hosted latency.
