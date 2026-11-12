# Bundled local inference assets

The app loads FP32 ONNX models from this directory using ONNX Runtime and Microsoft's ORT Extensions BERT tokenizer. It never downloads models, reads an HF token, or calls a model hub. Groq supplies answer generation and the separate RAGAS judge. Groq has no embedding endpoint, so embeddings and reranking remain local.

Assets preserve the previously used MiniLM weights and vocabulary:

| Directory | Original model | Original snapshot | Licence |
| --- | --- | --- | --- |
| embedding | sentence-transformers/paraphrase-MiniLM-L3-v2 | 4ca70771034acceecb2e72475f72050fcdde4ddc | Apache 2.0 |
| reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | 233902d25c440f23af6f7d6e94d2946bac0bee0a | Apache 2.0 |

Original cached model cards identify both licences as Apache 2.0; the licence text is included. These are local model assets, not Hugging Face Python packages or hosted services. Origin names are retained for attribution. The exported models are derived artifacts, not new training.

Conversion used the existing cached weights, PyTorch's ONNX export at opset 17, dynamic batch/sequence axes, FP32, evaluation mode, and the same BERT input IDs/masks/types. Embeddings retain 128-token truncation and mean pooling, without normalization. Reranker scores are raw logits with 512-token longest-first truncation. Tokenization and outputs were compared on 50 development questions plus Unicode and long-input cases before retiring the HF runtime. Numerical parity results are stored in `benchmarks/fsw-v1/maintenance-2026-11-13/onnx-parity.json`.

Models total approximately 154 MiB; each ONNX file is below GitHub's 100 MiB file limit. They are checked into the repository for self-contained deployment. No first-run download or global cache is needed. Do not introduce a download fallback when an asset is missing; restore a verified artifact instead.

[Microsoft ORT Extensions documentation](https://onnxruntime.ai/docs/extensions/) describes the tokenizer runtime. Package licences remain those of their upstream authors.
