"""Score exported rankings with native RAGAS ID-based metrics."""

import argparse
import asyncio
import csv
import json
from pathlib import Path
from evaluation.quality import retrieval_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("benchmarks/fsw-v1/questions.json"))
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ragas-retrieval"))
    args = parser.parse_args()
    thresholds = json.loads(args.thresholds.read_text()) if args.thresholds else None
    report = asyncio.run(retrieval_scores(args.rankings, args.dataset, thresholds))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["results"][0]))
        writer.writeheader()
        writer.writerows(report["results"])
    print(json.dumps(dict(summary=report["summary"], gate=report["gate"]), indent=2))
    if not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
