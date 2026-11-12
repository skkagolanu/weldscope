# FSW development benchmark v1

See [methodology and limitations](../../docs/evaluation.md). `questions.json` contains 50 development cases, 40 with provisional positive physical-PDF-page anchors. None is a held-out or independently expert-labelled evaluation case.

**Active:** `ragas-baseline/`, `ragas-verified/` and `ragas-thresholds.json` contain executed native RAGAS ID-based precision/recall results and the current floors. Export fresh rankings with `python scripts/benchmark_retrieval.py --output artifacts/retrieval`, then score them in the isolated environment using `python -m evaluation.run_retrieval --rankings artifacts/retrieval/report.json --thresholds benchmarks/fsw-v1/ragas-thresholds.json`.

**Historical:** `baseline/`, `verified/` and `thresholds.json` record the previous custom Recall/MRR/nDCG implementation. They are retained with their original definitions, not used by the current gate and not renamed as RAGAS scores. Their source hashes identify the bytes measured when those runs used uncommitted code.

The new RAGAS precision uses unique page-ID sets, differs from ranking-sensitive metrics, and penalizes pages not in the sparse positive labels. No judge credential is needed for these retrieval measurements. Native semantic answer assessment is a separate credentialed job; the availability of a runner is not proof its model scores have been measured.
