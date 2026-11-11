# WeldScope

A research assistant for **friction stir welding (FSW)** defects, causes, and corrective actions.

[Open the app](https://weldscope.streamlit.app/) · [CI](https://github.com/skkagolanu/weldscope/actions/workflows/ci.yml) · [Hosted checks](https://github.com/skkagolanu/weldscope/actions/workflows/deploy.yml) · [Hosting status](docs/hosting.md)

## What it does

- Answers greetings and questions about WeldScope and its sources locally, without inference calls or document search.
- Searches five bundled research PDFs using BM25S keyword retrieval and Chroma semantic search, then reranks passages with a cross-encoder.
- Uses Groq to generate streamed answers with inspectable source passages. The prompt asks it to abstain when evidence is insufficient; answer quality has not yet been benchmarked.
- Keeps up to ten completed exchanges within a bounded context budget. Search queries stay separate from chat history and sensor summaries; failed exchanges are excluded.
- Includes example sensor readings only when requested. The CSV is a static demonstration, not a live feed.
- Reports raw fusion and reranker scores separately. They are not probabilities; unmeasured response confidence is `None`.
- Records retrieval, reranking, generation, first-token and total latency per reply. Token counts remain unknown unless the provider reports them.

The packaged index contains **2,165 chunks**. The corpus remains local to the app; no new hosted vector database is required. Rebuild the index from the source PDFs when the corpus changes. Runtime edits to Community Cloud storage are not durable.

## Run locally

Use **Python 3.12**. The runtime, development tools, Docker image and CI all target this interpreter.

```bash
git clone https://github.com/skkagolanu/weldscope.git
cd weldscope
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
cp .env.example .env
streamlit run app.py
```

Dependencies and compatible transitive versions are pinned in `requirements*.txt`, `pyproject.toml`, and `constraints.txt`. PyTorch uses CPU wheels. Deprecated LangChain Community/Classic and their RAGAS-dependent path have been removed. PDF loading uses PyPDF directly; local embeddings use Sentence Transformers directly.

## Configure Groq generation

Use a Groq **Free-plan** account, then set these **root-level TOML values** in Streamlit app settings, outside any `[table]`:

```toml
LLM_PROVIDER = "groq"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = "your-key-added-privately"
```

For local use, set these variables in the ignored `.env` file instead. Greetings and app/source explanations work without a key. Never commit `.env` or `.streamlit/secrets.toml`.

Groq publishes free-plan limits of 30 requests/minute, 1,000/day, 8,000 tokens/minute and 200,000/day for this model; exact account limits may differ. A payment method is required to upgrade to its paid Developer plan. See [rate limits](https://console.groq.com/docs/rate-limits) and [billing FAQ](https://console.groq.com/docs/billing-faqs).

HF Inference Providers currently lists no included credits for free accounts. Changing the model through the same HF billing route does not resolve an HTTP 402. This app uses Groq directly for generation; the embedding and reranking weights are downloaded from Hugging Face and run locally. See [HF pricing](https://huggingface.co/docs/inference-providers/en/pricing). No paid-provider fallback is attempted.

Other settings include `RETRIEVAL_K`, `RERANK_TOP_K`, `MAX_TOKENS`, `TEMPERATURE`, and `VECTORSTORE_PATH`. See [.env.example](.env.example).

## Rebuild the index

```bash
python build_vectorstore.py --pdf-dir ./data --output-dir ./vectorstore
```

New chunks receive deterministic IDs from their source, page and text. Existing stored IDs are retained on load. Metadata filters are applied consistently to lexical and semantic search; the hybrid API supports exact metadata filters.

## Verification

The current Python 3.12 suite passes 140 tests with 96.81% package coverage.

```bash
pytest tests/ --cov-fail-under=90
black --check weldscope tests app.py scripts examples
pip check
pip install -r requirements-docs.txt
sphinx-build -b html -W docs /tmp/weldscope-docs
```

The tests cover routing, session context, filters, citations, provider request serialization and errors, real in-memory Langfuse spans, and the Streamlit chat flow. Provider tests use a local HTTP transport; they do not establish live account access. Real embedding, retrieval and reranking have also been exercised against a copy of the bundled index on the updated Python 3.12 stack.

The browser check handles Community Cloud's embedded app frame and verifies a greeting through the local reply path:

```bash
playwright install chromium
python scripts/check_hosted_app.py --url https://weldscope.streamlit.app/
```

Daily and post-CI checks detect a sleep page, startup failure, stale deployment, or broken greeting. They do not guarantee always-on hosting or live inference availability.

## Evaluation and observability

Optional Langfuse tracing connects query, retrieval and generation spans with feedback. Enable it only after configuring the deployment keys. Query content can be sent to Langfuse when enabled.

`GroundedAnswerEvaluator` provides explicit model-judged scores through the configured generator, with one provider request per example. It validates score ranges and input lengths, and leaves reference recall unknown when no reference is supplied. These are model judgements, not calibrated confidence or RAGAS benchmark results.

A labeled retrieval/answer-quality benchmark, measured quality regression gates, and cache integration remain future work. Software test coverage does not establish RAG answer quality.

## Code layout

`weldscope/` contains configuration, ingestion, retrieval, generation, conversation routing, orchestration, monitoring, tracing and optional evaluation. `app.py` is the Streamlit entrypoint. See [architecture](docs/architecture.md), [Streamlit documentation assessment](docs/streamlit-review.md), [dependency maintenance](docs/dependencies.md), and [hosting recovery](docs/hosting.md).

MIT License · Sai Karthik Kagolanu
