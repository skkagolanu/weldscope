# Offline RAGAS assessment

This is a separate Python 3.12 environment. RAGAS and its compatibility dependencies are intentionally absent from Streamlit's runtime requirements and package imports. The owner explicitly approved this narrow offline exception after previously declining it.

```bash
python3.12 -m venv .venv-eval
.venv-eval/bin/pip install -r evaluation/requirements.txt
.venv-eval/bin/python -m unittest discover -s evaluation/tests -v
```

`requirements.lock` pins the tested environment. RAGAS 0.4.3 fails to import with Community 0.4.2 because it imports removed VertexAI modules; Community **0.4.1** was tested successfully and is pinned only here. No packages are installed with dependencies bypassed. This compatibility exception does not establish those dependencies as supported application-runtime choices.

## Native retrieval metrics

Export actual rankings in the application environment, then score in the assessment environment:

```bash
.venv/bin/python scripts/benchmark_retrieval.py --output artifacts/retrieval
.venv-eval/bin/python -m evaluation.run_retrieval --rankings artifacts/retrieval/report.json --thresholds benchmarks/fsw-v1/ragas-thresholds.json --output artifacts/ragas-retrieval
```

Scoring uses upstream `IDBasedContextPrecision` and `IDBasedContextRecall`, not reimplementations. In RAGAS 0.4.3 these ID-based metrics are exposed through its older public metrics API; the newer collections API is used for all model-judged metrics. A future RAGAS major-version migration must revisit that API. Page IDs normalize each source to `filename#page=N`; duplicate chunks on one page count once. Sparse provisional positive labels penalize unlabelled pages, so precision is **known-anchor precision**, not fully adjudicated relevance precision. Historical custom MRR/nDCG reports remain archived, with their original definitions; those scores are not renamed as RAGAS scores.

## Native semantic metrics

Configure `GROQ_API_KEY` privately in the evaluator environment, or as the repository Actions secret, then use the manual **RAGAS Answer Assessment** workflow. It first generates real pipeline responses with the application environment, then runs RAGAS in the isolated environment. The workflow accepts 1–50 cases and an explicit judge model; the default is `openai/gpt-oss-120b`, separate from answer generation's 20B model.

Native metrics are Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness, and DomainSpecificRubrics for scope adherence. Answer relevance uses RAGAS's public embedding interface with the bundled ONNX model. There are no HF model downloads or inference calls; mandatory transitive HF packages remain offline only under the owner-approved exception. The scope rubric scores 1–5; other metrics retain their native scales. No invented combined confidence score is calculated.

Off-corpus/unsupported cases use the scope rubric, with inapplicable retrieval/answer-support metrics explicitly skipped. Reference-dependent metrics are skipped when no reference exists. Failures and non-finite results are recorded and make the run incomplete; failed examples are not silently turned into zero or dropped from coverage. Full JSON/CSV, generated samples and sanitized request telemetry are saved as Actions artifacts. RAGAS analytics is disabled. Model calls are paced; provider quotas may still cause failures. Full 50-case semantic assessment can take tens of minutes and consumes provider usage.

The removed `GroundedAnswerEvaluator` asked one model prompt for four scores. It is no longer an assessment path. Runtime citation guards, pytest/AppTest/Playwright acceptance tests, latency measurement and static checks remain ordinary software checks; RAGAS quality scoring complements them rather than replacing their purpose.

## Groq compatibility checks

RAGAS 0.4.3's generic Groq factory attempts to access `client.messages`, which the Groq SDK does not expose. The runner uses Instructor's supported `from_groq` client and RAGAS's public `InstructorLLM` wrapper directly. It also removes the wrapper's default `max_tokens` so Groq receives only `max_completion_tokens`. A real-SDK/mock-HTTP contract test verifies structured response parsing and the outgoing token-limit fields without making a provider call. This changes transport configuration, not any RAGAS metric implementation.

Long semantic runs write an atomic `progress.json` checkpoint after each assessed case. Repeated metric failures stop further judging after the configured error budget (default five); remaining cases are explicitly marked unassessed rather than disappearing from coverage. The final artifact records input/lock/source hashes, model identifiers, request counts and usage reported by provider responses. Missing usage stays unknown.


## Quota-aware execution and resumption

The judge transport respects Retry-After seconds/HTTP dates and provider quota reset headers, reserves a rolling token budget before each request, refunds reservations to actual reported usage, and caps retries and total tokens. The Groq SDK retry loop is disabled so transport pacing owns 429 handling. Metric checkpoints are written atomically after each native metric and resume only successful scores bound to identical samples, selected IDs, model options, model assets, metric code and dependency lock. Partial/failed scores are never substituted with zero.

Use workflow `samples_run_id` to reuse generated answers for a fresh assessment, or `resume_run_id` to resume a compatible metric checkpoint. A changed index/model/code requires fresh generation. The default eight-case panel spans all six workload categories; it remains a small development panel with provisional labels, not a held-out quality estimate. See [dated execution evidence](../docs/maintenance-2026-11-13.md).
