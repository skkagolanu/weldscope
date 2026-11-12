# FSW development benchmark v1

See [the methodology and limitations](../../docs/evaluation.md). `questions.json` is the immutable input for the current thresholds: 50 routing cases, 40 with provisional positive physical-PDF-page anchors. All are development cases, not held-out or independently expert-labelled evaluation data.

`baseline/report.json` and `baseline/results.csv` record the first actual four-variant run. `thresholds.json` freezes the regression floors. Reproduce with `python scripts/benchmark_retrieval.py --gate benchmarks/fsw-v1/thresholds.json`.

Results include source code hashes because the first run used uncommitted benchmark implementation alongside the recorded parent revision. This explicitly records the run's bytes rather than attributing it to a commit that did not yet contain them. No provider key, prompt transcript from real users or model-judged quality score is included.
