"""Run native RAGAS answer metrics in the isolated, credentialed evaluator."""

import argparse
import asyncio
import csv
import json
import os
import platform
from pathlib import Path
from time import monotonic

from evaluation.quality import answer_scores, digest


class RequestPacer:
    def __init__(self, interval):
        self.interval = interval
        self.last = 0.0
        self.lock = asyncio.Lock()
        self.requests = 0
        self.usage_reports = 0
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
                if isinstance(usage.get("total_tokens"), int):
                    self.usage_reports += 1
                for key in self.usage:
                    value = usage.get(key)
                    if isinstance(value, int):
                        self.usage[key] += value
            except (ValueError, AttributeError):
                pass


def make_judge(client, model):
    from instructor import Mode, from_groq
    from ragas.llms import InstructorLLM

    options = dict(temperature=0, max_completion_tokens=4096)
    if model.startswith("openai/gpt-oss-"):
        options.update(reasoning_effort="low", include_reasoning=False)
    # RAGAS 0.4.3's generic Groq factory incorrectly expects client.messages.
    # Use the supported Instructor Groq client and public RAGAS wrapper directly.
    llm = InstructorLLM(
        client=from_groq(client, mode=Mode.JSON), model=model, provider="groq", **options
    )
    llm.model_args.pop("max_tokens", None)
    return llm


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
            llm = make_judge(client, args.judge_model)
            embeddings = HuggingFaceEmbeddings(model=args.embedding_model, device="cpu")

            def checkpoint(rows):
                args.output.mkdir(parents=True, exist_ok=True)
                temporary = args.output / "progress.tmp"
                temporary.write_text(
                    json.dumps(
                        {
                            "framework": "ragas",
                            "complete": False,
                            "status": "in_progress",
                            "requested_samples": len(samples),
                            "completed_samples": len(rows),
                            "results": rows,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                temporary.replace(args.output / "progress.json")

            report = await answer_scores(
                samples, llm, embeddings, on_progress=checkpoint, error_budget=args.error_budget
            )
    report.update(
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        samples_sha256=digest(args.samples),
        provider_requests=pacer.requests,
        reported_token_usage=pacer.usage if pacer.usage_reports else None,
        responses_reporting_usage=pacer.usage_reports,
        request_interval_seconds=args.request_interval,
        error_budget=args.error_budget,
        dataset_sha256=payload.get("dataset_sha256"),
        generation_model=payload.get("generation_model"),
        lock_sha256=digest(Path(__file__).parent / "requirements.lock"),
        runner_sha256=digest(__file__),
        metrics_runner_sha256=digest(Path(__file__).parent / "quality.py"),
        python=platform.python_version(),
        source_commit=os.environ.get("GITHUB_SHA"),
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
    parser.add_argument("--error-budget", type=int, default=5)
    args = parser.parse_args()
    if args.error_budget < 1:
        parser.error("Error budget must be positive")
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
