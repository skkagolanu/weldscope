# WeldScope

A research assistant for **friction stir welding (FSW)** defects, causes, and corrective actions.

[Open the app](https://weldscope.streamlit.app/) · [CI](https://github.com/skkagolanu/weldscope/actions/workflows/ci.yml) · [Hosted checks](https://github.com/skkagolanu/weldscope/actions/workflows/deploy.yml) · [Hosting status](docs/hosting.md)

Verified on 13 November 2026 after owner reboot: public `quality-v3`, Python 3.12.16, HF-free application environment and a passing seven-turn live regression. Warm research samples took about 1.1 seconds in the browser; the first research sample took 2.43 seconds. See the dated hosting report for limitations.

## What it does

- Answers greetings and questions about WeldScope and its sources locally, without inference calls or document search. Other unsupported requests receive a brief local scope refusal, including requests to assess real equipment or handle a live incident.
- Searches five bundled research PDFs using BM25S keyword retrieval and local ONNX/NumPy semantic search, then reranks passages with a cross-encoder.
- Uses Groq to generate answers with inspectable source passages. Research output is buffered until citation checks finish, before any answer text is displayed. The prompt asks it to abstain when evidence is insufficient; retrieval has a provisional 50-case development benchmark; semantic answer quality remains unmeasured.
- Keeps up to ten completed exchanges within a bounded context budget. Search queries stay separate from chat history and sensor summaries; failed exchanges are excluded.
- Includes example sensor readings only when requested. The CSV is a static demonstration, not a live feed.
- Reports raw fusion and reranker scores separately. They are not probabilities; unmeasured response confidence is `None`.
- Records retrieval, reranking, generation, first-token and total latency per reply. Token counts remain unknown unless the provider reports them.

The packaged index contains **2,124 current-PDF chunks**. The corpus remains local to the app; no new hosted vector database is required. Rebuild the index from the source PDFs when the corpus changes. Runtime edits to Community Cloud storage are not durable.

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

Dependencies and compatible transitive versions are pinned in `requirements*.txt`, `pyproject.toml`, and `constraints.txt`. Local inference uses bundled ONNX models; the app needs no model downloads or HF token. Deprecated LangChain Community/Classic paths have been removed from the application runtime. RAGAS uses a separately locked offline assessment environment with an explicitly approved compatibility exception. PDF loading uses PyPDF directly; local embeddings and reranking use ONNX Runtime and Microsoft ORT Extensions. HF-related packages are allowed only in the separately locked offline RAGAS environment.

## Configure Groq generation

Use a Groq **Free-plan** account, then set these **root-level TOML values** in Streamlit app settings, outside any `[table]`:

```toml
LLM_PROVIDER = "groq"
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = "your-key-added-privately"
```

For local use, set these variables in the ignored `.env` file instead. Greetings and app/source explanations work without a key. Never commit `.env` or `.streamlit/secrets.toml`.

Groq publishes free-plan limits of 30 requests/minute, 1,000/day, 8,000 tokens/minute and 200,000/day for this model; exact account limits may differ. A payment method is required to upgrade to its paid Developer plan. See [rate limits](https://console.groq.com/docs/rate-limits) and [billing FAQ](https://console.groq.com/docs/billing-faqs).

The application uses Groq directly for generation. Embeddings and reranking run from bundled local ONNX assets. There are no HF runtime packages, model downloads, token requirements or inference calls. RAGAS retains its required HF dependencies only in the isolated evaluator, as explicitly approved.

Other settings include `RETRIEVAL_K`, `RERANK_TOP_K`, `MAX_TOKENS`, `TEMPERATURE`, and `VECTORSTORE_PATH`. See [.env.example](.env.example).

## Rebuild the index

```bash
python build_vectorstore.py --pdf-dir ./data --output-dir ./indexes/fsw-v2
```

New chunks receive deterministic IDs from their source, page and text. Existing stored IDs are retained on load. Metadata filters are applied consistently to lexical and semantic search; the hybrid API supports exact metadata filters.

## Verification

The Python 3.12 suite includes regression tests for corpus-only routing, fast scope refusals, and bounded streaming redraws.

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

Daily and post-CI basic checks detect a sleep page, startup failure, old branding, or broken greeting. The manual hosted research regression separately verifies the current release marker, Python version and HF-package exclusions. They do not guarantee always-on hosting or live inference availability.

## Evaluation and observability

Optional Langfuse tracing connects query, retrieval and generation spans with feedback. Enable it only after configuring the deployment keys. Query content can be sent to Langfuse when enabled.

[RAGAS](evaluation/README.md) is the active offline quality-assessment framework. Native ID-based context precision/recall have been executed on all 160 retrieval question/variant pairs. The custom model judge and custom retrieval scoring implementation were retired; historical reports retain their original labels.

Native semantic assessment covers faithfulness, answer relevance, context precision/recall, factual correctness and a scope rubric. Run the manual **RAGAS Answer Assessment** workflow with the privately configured `GROQ_API_KEY` Actions secret. A completed eight-case panel is now recorded in the dated evaluation report (five answerable cases plus three negative cases); its small, provisional development labels do not establish population accuracy. RAGAS remains outside the Streamlit dependency graph, so evaluation adds no per-chat model calls. Ordinary pytest/AppTest/Playwright checks and runtime citation guards remain in place. See [latest maintenance results](docs/maintenance-2026-11-13.md) and [methodology](docs/evaluation.md).

## Code layout

`weldscope/` contains configuration, ingestion, retrieval, generation, conversation routing, orchestration, monitoring, tracing and optional evaluation. `app.py` is the Streamlit entrypoint. See [architecture](docs/architecture.md), [Streamlit documentation assessment](docs/streamlit-review.md), [dependency maintenance](docs/dependencies.md), [latency measurements](docs/latency.md), and [hosting recovery](docs/hosting.md), [evaluation](docs/evaluation.md) and [operational telemetry](docs/operations.md).

MIT License · Sai Karthik Kagolanu
