# Evaluation and quality gates

**Current 13 November status:** see [maintenance results](maintenance-2026-11-13.md) for the rebuilt HF-free runtime, configuration/model comparisons and successful semantic pilot. The older dated runs below preserve their original index and metric definitions.

## Active framework: RAGAS

Following the owner’s explicit request and approval of an offline-only dependency exception, **RAGAS 0.4.3 is now the quality-assessment framework**. The old custom model judge and custom retrieval score implementation were removed. Runtime guards and software acceptance tests remain ordinary application checks. RAGAS is isolated from Streamlit; [setup, metrics and commands](https://github.com/skkagolanu/weldscope/blob/main/evaluation/README.md).

Actual native RAGAS results on 40 answerable development questions, four variants (160 rows), using known physical-PDF-page anchors:

| Variant | Native ID-based precision | Native ID-based recall |
| --- | ---: | ---: |
| BM25S | 0.1350 | 0.4875 |
| Vector | 0.0696 | 0.2500 |
| Hybrid | 0.1296 | 0.4875 |
| Hybrid 20 → rerank 5 | 0.1550 | 0.5750 |

[Native baseline JSON](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/ragas-baseline/report.json), [CSV](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/ragas-baseline/results.csv), [repeat gated run](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/ragas-verified/report.json). These are upstream RAGAS calculations, with zero judge/provider requests. Sparse, provisional positive anchors mean precision is known-anchor precision, not independently adjudicated overall relevance. No semantic scores or held-out quality gains are implied.

The active CI quality gate uses the new native-metric floors in `ragas-thresholds.json`, with a 0.03 absolute regression tolerance. It scores actual fresh rankings in a separate RAGAS job. The native semantic runner uses five standard collections metrics plus a scope rubric, with explicit failures/skips and no combined confidence. The manual credentialed workflow generates real answers, then judges them; execution status is recorded separately below.

## Historical custom benchmark (before RAGAS migration)

## What was executed

The `fsw-v1` development set contains **50 cases**: 10 exact-term, 10 paraphrase, 8 domain-synonym, 6 multihop, 6 follow-up and 10 negative cases. Forty have answer evidence; ten check off-corpus or unavailable-information behaviour. All 50 have expected routing labels. Questions and reference summaries were authored with AI assistance and checked against source pages; they have **not** received independent welding-expert annotation. These are development cases, not a held-out test set. Negative routing tests do not establish that the model abstains correctly on every unsupported question.

[Dataset](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/questions.json) records PDF SHA-256 fingerprints, one-based physical page anchors, reference summaries and follow-up histories. Positive page anchors are provisional and **not exhaustive relevance judgements**. Relevant but unlabelled pages receive no metric credit; the scores must not be presented as fully adjudicated retrieval precision or generalization estimates.

The index contains 2,165 chunks. **42** refer to three files absent from the current bundled PDFs (`doc1_standards.pdf`, `doc2_manual.pdf`, `doc3_procedures.pdf`). Some of those old passages present themselves as standards material. Runtime and benchmark loaders now exclude missing-source/out-of-range-page chunks, and vector search uses the same explicit source allowlist as BM25. The five current PDFs contribute **2,123 eligible chunks**. The shipped index is preserved: the benchmark opens a temporary copy because Chroma can modify a SQLite store on load. Matching filename/page verifies a bounded provenance condition; it does not certify the PDFs' licences, authenticity or every indexed character.

## Retrieval results

First recorded local run, Python 3.12, 40 answerable cases, top five chunks:

| Variant | Page-anchor Recall@5 | MRR@5 | nDCG@5 | Mean query time |
| --- | ---: | ---: | ---: | ---: |
| BM25S | 0.4875 | 0.4238 | 0.4182 | 0.45 ms |
| Vector | 0.2500 | 0.1488 | 0.1656 | 16.80 ms |
| Hybrid | 0.4875 | 0.3446 | 0.3651 | 14.45 ms |
| Hybrid 20 candidates → cross-encoder top 5 | 0.5750 | 0.4479 | 0.4527 | 275.03 ms |

Routing labels matched **50/50**. Recall measures the fraction of labelled evidence pages found; MRR uses the first matching chunk; nDCG gives gain only to the first occurrence of each labelled page, so repeated chunks from one page cannot inflate gain. Questions are averaged equally. [Raw JSON](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/baseline/report.json) includes every ranking, source IDs/pages, package/platform metadata, source-file hashes, initialization time and per-case times. [CSV](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/baseline/results.csv) supports comparison.

The reranked result uses **20 candidates**, while the deployed pipeline retains **5 candidates → 3 passages**. This is an architecture comparison, not an isolated estimate of reranking benefit and not proof the deployed configuration gained 8.75 percentage points. BM25 outranked the current hybrid ordering on these labels, and vector retrieval was weaker; the benchmark does not support claiming hybrid always wins. Results motivate a later held-out evaluation and a controlled candidate-pool ablation before changing production defaults.

Times are local retrieval measurements, exclude initialization recorded separately, and include incidental first-call work where incurred. Variant order is fixed. Observed p95 values in the raw report describe these 40 cases; they are not a service latency SLA or statistically robust tail estimate. This run made **zero model-provider requests**.

## Reproduce and gate

```bash
.venv/bin/python scripts/benchmark_retrieval.py --output artifacts/retrieval
.venv-eval/bin/python -m evaluation.run_retrieval --rankings artifacts/retrieval/report.json --thresholds benchmarks/fsw-v1/ragas-thresholds.json --output artifacts/ragas-retrieval
```

Ranking export uses the Python 3.12 application environment without an inference key; scoring requires the separate offline RAGAS environment. Dataset fingerprints must match the local PDFs. Active native ID precision/recall floors allow an absolute drop of **0.03** from their frozen baselines; historical custom MRR/nDCG floors are archived. Routing expectations remain software checks. Dataset changes require explicit review/versioning. Baselines are never regenerated by a test run. CI runs the actual retrieval models, fails on regressions, and uploads JSON/CSV plus sanitized telemetry artifacts. Hardware-dependent latency is recorded, not used as a universal CI threshold. A [repeat local gated run](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/verified/report.json) passed; this establishes reproducibility under the tested conditions, not independent validation of labels.

Application type checking (`mypy`) and error-level lint (`pylint --errors-only`) are now blocking checks with dependencies installed. Previously advisory type errors were repaired rather than ignored. Formatting, test coverage and security checks remain required.

## Citation and support diagnostics

Research replies are checked for invalid citation numbers/ranges, missing citations, and numerical tokens absent from the retrieved passages. Invalid references cause the **final** answer to be withheld. Missing references or numerical-support gaps produce a review warning. Valid numbers do not prove valid claims: these checks **do not establish semantic entailment, factual correctness or citation correctness**. An abstention recognizer is also heuristic. Research text is now buffered until these structural checks finish; invalid citations and failed-provider partial replies are never delivered to the UI. This pre-output structural check does not prove semantic factual support.

The exact greeting → purpose → FSW → ship/butter-chicken conversation is covered by a Streamlit regression. Domain synonyms such as HAZ, TMAZ, axial force, anvil and shoulder scroll now route to research. Tests also preserve explicit follow-ups and reject unrelated tasks that merely include the word FSW.

## Historical dependency decision (superseded by the explicit RAGAS request)

RAGAS was assessed for separate offline use. The current PyPI release **0.4.3** and upstream dependency metadata directly require `langchain-community`, along with their wider dependency stack. The owner explicitly reaffirmed **no legacy packages anywhere**, including an isolated evaluation environment. At that point RAGAS was **not installed**, no dependency-bypassing/vendor workaround was used, and no RAGAS scores are reported. [Release metadata](https://pypi.org/pypi/ragas/json), [upstream dependencies](https://github.com/vibrantlabsai/ragas/blob/main/pyproject.toml).

The now-retired `GroundedAnswerEvaluator` was a simple model judge, not an equivalent implementation of RAGAS metrics. Its scores are not used in this benchmark. At that earlier point, Groq was configured only in Streamlit, so offline LLM judging was unavailable; the owner later added the private Actions secret, enabling the native pilot documented below. Model-judged faithfulness, answer relevance and reference recall remain unmeasured. An independently reviewed question set and judge calibration are still needed before claiming semantic quality improvements.

## Hosted verification

`python scripts/hosted_regression.py --output artifacts/hosted-regression.json` exercises the real public app, detects its current `quality-v3` diagnostics, runs the exact four-turn conversation, makes live Groq research requests, checks domain synonyms/follow-ups, and records reported token usage, timing, source page identifiers and structural checks. Provider usage is consumed only by research turns. Cold-start state cannot be forced without owner controls; each observed model-load time is retained rather than relabelled as generation latency. The dated observed result is recorded in the hosting report when available.

The manual **Hosted Research Regression** GitHub Actions workflow runs this same browser test and preserves its JSON even on failure. It is not scheduled on every push: live research requests consume the deployed provider quota. The daily basic health check remains provider-free when its local routes behave correctly.

The public seven-turn regression **passed after the owner reboot**, including four real Groq research requests with reported token usage and the exact recipe refusal with zero retrieval/generation. [Hosted evidence](benchmarks/hosted-quality-2026-11-12.json) preserves answers, source/page identifiers, passage hashes and diagnostics. The future-standard request abstained, although the then-serving diagnostic produced a false review warning; the corresponding abstention wording is now covered by a local regression. These observations do not constitute a 50-answer semantic-quality benchmark. [CI for `56b32ce`](https://github.com/skkagolanu/weldscope/actions/runs/43213547802) also passed the actual retrieval gate, blocking lint/type checks and tests.

Final local verification after the abstention-diagnostic correction: **181 tests passed, 97.71% package coverage**, formatting/type checks and the warning-free docs build. The earlier remotely executed `56b32ce` suite had 180 passing tests and 97.70% coverage; the final source changes are not retroactively attributed to that CI run.

## Native RAGAS execution status

The current separately locked evaluator passes **15 native metric/SDK/pacing/checkpoint contract tests**. Current runtime verification passes **179 tests with 97.44% package coverage**. The rebuilt index passes all 160 native four-variant retrieval rows against existing floors; another 1,600 model/chunk comparisons and 120 production-configuration rows are preserved in the dated report. These are provisional development measurements.

The missing-secret preflight and later two-case quota failure are historical. The completed [one-case pilot 43302486289](https://github.com/skkagolanu/weldscope/actions/runs/43302486289) made 14 successful judge requests and reported 17,270 tokens, with all six eligible metrics completed. It establishes actual execution, not enough coverage for representative quality means. Raw pilot scores remain labelled integration evidence.

The [eight-case run 43360885422](https://github.com/skkagolanu/weldscope/actions/runs/43360885422) subsequently completed all planned metrics: 73 successful judge requests, zero failed requests and 96,619 reported tokens. Five answerable cases scored all five support/reference metrics; all eight cases scored the scope rubric. Descriptive means and exact input/source-version limitations are recorded in [latest maintenance](maintenance-2026-11-13.md). Brief provisional references and a tiny development panel prevent population accuracy or calibration claims. The 0.344 native factual-correctness mean motivates quality investigation; a completed run is not a passing quality verdict.


## Latest hosted release verification

After the owner reboot, `quality-v3` passed the public seven-turn regression on 13 November 2026, including the exact historical conversation, live Groq answers, valid domain synonyms/follow-ups and unsupported-standard abstention. The running environment reports Python 3.12.16 and no unexpected HF packages. The exact recipe request performed no retrieval/generation or provider usage. [Current raw hosted evidence](https://github.com/skkagolanu/weldscope/blob/main/benchmarks/fsw-v1/maintenance-2026-11-13/hosted-v3.json). Warm TMAZ/HAZ samples took 1,058/1,145 ms in the browser; the load-incurring first research turn took 2,432 ms browser / 1,493 ms backend. These are dated observations, not universal latency or semantic-quality guarantees. The native semantic panel limitations remain unchanged.
