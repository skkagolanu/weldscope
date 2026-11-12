# Maintenance evidence — 13 November 2026

## Runtime and dependency boundary

The `quality-v3` source removes all HF-related application packages and service calls. Groq supplies generation. The same locally cached MiniLM embedding and cross-encoder weights were exported to bundled FP32 ONNX assets; Microsoft ORT Extensions performs BERT tokenization. Groq has no documented embedding endpoint, so embeddings remain local. There is no model-download fallback or HF token requirement.

The owner explicitly retained **RAGAS offline**, allowing its mandatory HF dependencies only in `evaluation/requirements.lock`. RAGAS itself is never imported by Streamlit. Its embedding adapter now uses the bundled ONNX model too, and the workflow sets `HF_HUB_OFFLINE=1` and disables RAGAS analytics. Optional Langfuse export is separate.

Tokenization matched on 50 development questions plus Unicode/long-input probes. Maximum FP32 output differences were **5.36e-7** for embeddings and **5.72e-6** for reranking. These are numerical parity checks, not semantic quality scores. [Parity artifact](../benchmarks/fsw-v1/maintenance-2026-11-13/onnx-parity.json), [asset provenance/licences](https://github.com/skkagolanu/weldscope/blob/main/models/README.md).

## Clean index and native retrieval gate

`indexes/fsw-v2` was rebuilt directly from the **five current PDFs**, with 500-character chunks, 50-character overlap, deterministic IDs and **2,124 chunks**. JSON passages, NumPy vectors, PDF fingerprints and model hashes are versioned. The obsolete Chroma store was removed from current source, so its 42 stale passages are no longer merely filtered at query time. Local exact squared-L2 search preserves the previous unnormalized embedding geometry. No managed vector service was added.

Native RAGAS scores on the same 40 answerable development cases:

| Configuration | Known-page precision | Known-page recall |
| --- | ---: | ---: |
| BM25S top 5 | 0.13292 | 0.4875 |
| Vector top 5 | 0.06958 | 0.2500 |
| Hybrid top 5 | 0.12958 | 0.4875 |
| Hybrid 20 → rerank 5 | 0.15292 | 0.5750 |

The existing native regression floors passed. Routing matched 50/50 development labels. Recall matches the old index benchmark; small precision differences remain visible rather than being rounded into a claim of exact ranking parity. [Runtime rankings](../benchmarks/fsw-v1/maintenance-2026-11-13/runtime-rankings.json), [native scores](../benchmarks/fsw-v1/maintenance-2026-11-13/runtime-ragas.json).

Labels are provisional positive physical-page anchors, not exhaustive expert relevance judgments. Repeated page IDs count once. These are development comparisons, not held-out generalization or factual-answer measurements.

## Weak-vector investigation

Six stored embeddings sampled across the five PDFs matched freshly encoded current-model vectors at cosine 1.0, including their original norms. A broad embedding-version mismatch was therefore not supported by that sample. Normalizing the old vectors and using exact cosine search did not improve recall on this dataset.

The old MiniLM-L3 encoder truncates at 128 tokens: **18.3%** of cleaned 500-character chunks and **72.9%** of 1,000-character chunks exceeded that limit. Alternatives were compared on identical labels, cleaned PDF text and chunk settings, with the same cross-encoder:

| Encoder / chunk characters | Vector top-5 recall | Hybrid 20 → rerank-5 recall |
| --- | ---: | ---: |
| MiniLM-L3 / 500 | 0.2500 | 0.5750 |
| MiniLM-L3 / 1,000 | 0.2000 | 0.5750 |
| MiniLM-L6 / 500 | 0.2875 | 0.6375 |
| MiniLM-L6 / 1,000 | 0.4500 | 0.6250 |
| BGE-small / 500 | 0.6125 | 0.6500 |
| BGE-small / 1,000 | 0.5375 | 0.7250 |

BGE is a promising candidate, not a promoted production setting or proof of better factual answers. Model, cleaning, normalization and chunking vary across these comparisons; they do not isolate a single causal mechanism. These local experiments completed using the previous cached-model environment before HF runtime retirement. Their original driver is preserved in commit `a96ef47`; it is no longer an active application script. Native score artifacts retain all 1,600 question/configuration rows: [L3](../benchmarks/fsw-v1/maintenance-2026-11-13/l3-ragas.json), [L6](../benchmarks/fsw-v1/maintenance-2026-11-13/l6-ragas.json), [BGE](../benchmarks/fsw-v1/maintenance-2026-11-13/bge-ragas.json), with corresponding `*-rankings.json` files.

Their initial affinity setting happened after numeric-library imports, so those experiment timings/RSS are not certified two-core or isolated model-memory comparisons. Production promotion requires ONNX packaging/parity and resource/quality testing for the new model, independent labels, and hosted measurements. The current release deliberately preserves the existing model.

## Actual production configuration versus larger pool

A separate HF-free runtime profile sets two-core affinity **before** numeric imports and verifies every live thread's affinity. Native RAGAS evaluated all 40 positive cases for each configuration:

| Pool → passages | Known-page precision | Recall | Warm mean retrieval + rerank | Observed warm p95 |
| --- | ---: | ---: | ---: | ---: |
| 5 → 3 (production) | 0.20833 | 0.4625 | 158 ms | 250 ms |
| 20 → 3 | 0.24583 | 0.5500 | 523 ms | 681 ms |
| 20 → 5 | 0.15292 | 0.5750 | 524 ms | 682 ms |

Initialization plus first retrieval/reranking was **474 ms**, excluding Python imports, generation and browser/network time. Python imports took another **434 ms**; combined imports plus cold retrieval/reranking were approximately **908 ms** before generation/UI. Process peak RSS was **414 MiB**. The thread audit initially caught the tokenizer creating automatically affinitized worker threads; explicitly limiting its ORT session to one thread eliminated that behavior. The final saved run verifies all five threads remain within the selected two cores. Warm production statistics exclude the first model-loading case (39 samples); larger variants have 40 each. Fixed order, one local run, and a two-core host slice do not establish Community Cloud limits or a universal two-second SLA. The larger pool costs roughly 366 ms extra retrieval/reranking locally; production remains **5 → 3**. [Profile and thread affinities](../benchmarks/fsw-v1/maintenance-2026-11-13/runtime-profile.json), [native configuration scores](../benchmarks/fsw-v1/maintenance-2026-11-13/runtime-config-ragas.json). Reproduce with `PYTHONPATH=. python scripts/profile_local_retrieval.py`.

## Semantic RAGAS execution and quota recovery

The earlier two-case attempt [43252385554](https://github.com/skkagolanu/weldscope/actions/runs/43252385554) made real provider requests but hit the judge's 8,000-token/minute limit; four native metric calls remained failed. Its partial averages are not published as quality results.

The fix implements rolling token reservations at 6,000/minute, provider reset headers, Retry-After seconds/HTTP dates, bounded 429 retries, actual-usage reconciliation and explicit per-run token/error budgets. Successful native metric checkpoints resume only against matching sample/code/model/options/assets/lock fingerprints. Changed inputs require a new assessment. No key is read into chat or committed.

A fresh **one-case integration pilot completed all six native metrics**, with **14 successful provider requests, zero failed requests and 17,270 reported judge tokens**: [successful workflow](https://github.com/skkagolanu/weldscope/actions/runs/43302486289), [raw pilot artifact](../benchmarks/fsw-v1/maintenance-2026-11-13/semantic-pilot.json). One case is too small for headline quality averages; the artifact is retained as execution evidence. It reused exact saved generation inputs and the prior stack, and does not establish new-release generation quality.

The next assessment uses freshly generated current-release answers on an eight-case panel spanning exact, paraphrase, synonym, multihop, follow-up and negative categories, capped at 100,000 judge tokens. Only completed eligible metrics will be reported, with positive/negative coverage and denominators. A small provisional development panel is not a calibrated population estimate.

## Pre-output validation and observability

Research provider chunks are buffered until structural citation checks complete. Invalid citation numbers are replaced before any research text reaches UI callbacks; provider failures cannot leak partial research answers. `stream_query` uses the same validated path. `first_token_ms` remains a provider diagnostic, while `first_visible_ms` records post-validation server delivery. This adds a deliberate validation boundary and is not a claim of factual entailment or browser-paint timing.

Real in-memory Langfuse/OpenTelemetry tests verify local parent/child correlation. Sanitized bounded JSONL records retain timings, reported tokens and errors locally. **Hosted external Langfuse export and retention remain unverified by the owner's explicit choice.** Community Cloud disk is ephemeral and is not a durable monitoring backend.

## Verification and deployment boundary

The fresh Python 3.12 application environment passed **173 tests, 97.37% package coverage**. Real inference was tested with network connections denied and no HF-related runtime distributions installed. The separate RAGAS environment passed 15 contract/pacing/checkpoint tests. Both installed dependency graphs passed compatibility checks; application mypy and error lint passed before final documentation changes.

`quality-v3` public deployment verification is pending the pushed revision actually serving. Older successful `quality-v1` hosted tests remain dated evidence, not proof of this release. The exact multi-turn hosted regression and a real Groq query must be rerun against current diagnostics before claiming it deployed.
