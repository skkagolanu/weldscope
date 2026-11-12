"""Run native RAGAS answer metrics in the isolated, credentialed evaluator."""

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from time import monotonic

from evaluation.quality import answer_scores, digest


class RequestPacer:
    def __init__(self, interval):
        self.interval = interval
        self.last = 0.0
        self.lock = asyncio.Lock()
        self.requests = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    async def request(self, request):
        async with self.lock:
            await asyncio.sleep(max(0, self.interval - (monotonic() - self.last)))
            self.last = monotonic()
            self.requests += 1

    async def response(self, response):
        await response.aread()
        if response.is_success:
            try:
                usage = response.json().get("usage") or {}
                for key in self.usage:
                    value = usage.get(key)
                    if isinstance(value, int):
                        self.usage[key] += value
            except (ValueError, AttributeError):
                pass


async def assess(args):
    if not os.environ.get("GROQ_API_KEY"):
        return dict(
            framework="ragas",
            complete=False,
            status="blocked_missing_judge_credential",
            summary={},
            results=[],
        )
    import httpx
    from groq import AsyncGroq
    from ragas.llms import llm_factory
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings

    payload = json.loads(args.samples.read_text())
    samples = payload["samples"][: args.limit]
    if payload.get("failures") or len(samples) != args.limit:
        raise ValueError("Incomplete generation samples; review failures before scoring")
    pacer = RequestPacer(args.request_interval)
    async with httpx.AsyncClient(
        event_hooks={"request": [pacer.request], "response": [pacer.response]}
    ) as transport:
        async with AsyncGroq(http_client=transport, timeout=60, max_retries=1) as client:
            options = dict(temperature=0, max_completion_tokens=4096)
            if args.judge_model.startswith("openai/gpt-oss-"):
                options.update(reasoning_effort="low", include_reasoning=False)
            llm = llm_factory(args.judge_model, provider="groq", client=client, **options)
            embeddings = HuggingFaceEmbeddings(model=args.embedding_model, device="cpu")
            report = await answer_scores(samples, llm, embeddings)
    report.update(
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        samples_sha256=digest(args.samples),
        provider_requests=pacer.requests,
        reported_token_usage=pacer.usage,
        limitations="Model-judged, provisional development labels; no human calibration or universal factual guarantee. Native scope rubric uses 1-5; other metrics retain their native scales.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ragas-answers"))
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    parser.add_argument("--limit", type=int, choices=range(1, 51), default=50)
    parser.add_argument("--request-interval", type=float, default=2.1)
    args = parser.parse_args()
    if args.request_interval < 0:
        parser.error("Request interval must be non-negative")
    report = asyncio.run(assess(args))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "metric", "score", "error", "skipped"])
        writer.writeheader()
        for row in report["results"]:
            for metric in set(row["scores"]) | set(row["errors"]) | set(row["skipped"]):
                writer.writerow(
                    dict(
                        id=row["id"],
                        metric=metric,
                        score=row["scores"].get(metric),
                        error=row["errors"].get(metric),
                        skipped=row["skipped"].get(metric),
                    )
                )
    print(json.dumps({key: report[key] for key in ["framework", "complete", "summary"]}, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
