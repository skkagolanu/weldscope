"""Run native RAGAS answer metrics in the isolated, credentialed evaluator."""

import argparse
import asyncio
import csv
import json
import os
import platform
from pathlib import Path

from evaluation.quality import answer_scores, digest


from evaluation.pacing import RequestPacer, QuotaTransport


def judge_options(model):
    options = dict(temperature=0, max_completion_tokens=4096, max_retries=2)
    if model.startswith("openai/gpt-oss-"):
        options.update(reasoning_effort="low", include_reasoning=False)
    return options


def make_judge(client, model):
    from instructor import Mode, from_groq
    from ragas.llms import InstructorLLM

    options = judge_options(model)
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
    from evaluation.embeddings import LocalRagasEmbeddings

    payload = json.loads(args.samples.read_text())
    samples = payload["samples"][: args.limit]
    selected_ids = [
        value.strip() for value in getattr(args, "case_ids", "").split(",") if value.strip()
    ]
    if selected_ids and selected_ids != [sample["id"] for sample in samples]:
        raise ValueError("Requested case IDs differ from saved samples")
    if payload.get("failures") or len(samples) != args.limit:
        raise ValueError("Incomplete generation samples; review failures before scoring")
    binding = dict(
        samples_sha256=digest(args.samples),
        selected_ids=[s["id"] for s in samples],
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        metrics_runner_sha256=digest(Path(__file__).parent / "quality.py"),
        dependency_lock_sha256=digest(Path(__file__).parent / "requirements.lock"),
        judge_options=judge_options(args.judge_model),
        embedding_artifacts={
            name: digest(Path("models/embedding") / name)
            for name in ["model.onnx", "vocab.txt", "config.json"]
        },
    )
    fingerprint = (
        __import__("hashlib").sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    )
    resume_rows = []
    if args.resume:
        saved = json.loads(args.resume.read_text())
        if saved.get("assessment_fingerprint") != fingerprint:
            raise ValueError("Resume fingerprint differs from inputs, model or metric definitions")
        resume_rows = saved.get("results", [])
    pacer = RequestPacer(args.request_interval, args.tokens_per_minute, args.token_budget)
    async with httpx.AsyncClient(transport=QuotaTransport(pacer)) as transport:
        async with AsyncGroq(http_client=transport, timeout=60, max_retries=0) as client:
            llm = make_judge(client, args.judge_model)
            embeddings = LocalRagasEmbeddings(model=args.embedding_model)

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
                            "rows_with_progress": len(rows),
                            "completed_samples": sum(
                                len(row["scores"]) + len(row["skipped"]) == 6 and not row["errors"]
                                for row in rows
                            ),
                            "assessment_fingerprint": fingerprint,
                            "binding": binding,
                            "results": rows,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                temporary.replace(args.output / "progress.json")

            report = await answer_scores(
                samples,
                llm,
                embeddings,
                on_progress=checkpoint,
                error_budget=args.error_budget,
                resume_rows=resume_rows,
            )
    report.update(
        assessment_fingerprint=fingerprint,
        binding=binding,
        resumed_from_sha256=digest(args.resume) if args.resume else None,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        samples_sha256=digest(args.samples),
        provider_requests=pacer.requests,
        reported_token_usage=pacer.usage if pacer.usage_reports else None,
        responses_reporting_usage=pacer.usage_reports,
        provider_status_counts=pacer.status_counts,
        observed_token_limit_per_minute=pacer.observed_token_limit,
        request_interval_seconds=args.request_interval,
        error_budget=args.error_budget,
        token_budget=args.token_budget,
        tokens_per_minute=args.tokens_per_minute,
        deferred_reason=pacer.deferred_reason,
        dataset_sha256=payload.get("dataset_sha256"),
        generation_model=payload.get("generation_model"),
        lock_sha256=digest(Path(__file__).parent / "requirements.lock"),
        runner_sha256=digest(__file__),
        metrics_runner_sha256=digest(Path(__file__).parent / "quality.py"),
        python=platform.python_version(),
        source_commit=os.environ.get("GITHUB_SHA"),
        selected_cases=len(samples),
        selected_categories=sorted({sample.get("category", "unknown") for sample in samples}),
        coverage_policy="Report descriptive quality only from complete eligible metrics; a pilot is integration evidence, not a population quality estimate.",
        limitations="Model-judged, provisional development labels; no human calibration or universal factual guarantee. Native scope rubric uses 1-5; other metrics retain their native scales.",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ragas-answers"))
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--embedding-model", default="embedding")
    parser.add_argument("--limit", type=int, choices=range(1, 51), default=50)
    parser.add_argument("--request-interval", type=float, default=2.1)
    parser.add_argument("--error-budget", type=int, default=5)
    parser.add_argument("--tokens-per-minute", type=int, default=6000)
    parser.add_argument("--token-budget", type=int, default=100000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--case-ids", default="")
    args = parser.parse_args()
    if args.token_budget < 1 or args.tokens_per_minute < 1:
        parser.error("Quota budgets must be positive")
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
    published = {key: report[key] for key in ["framework", "complete"]}
    published["coverage"] = {
        name: {key: value for key, value in data.items() if key != "mean"}
        for name, data in report["summary"].items()
    }
    if report["complete"] and args.limit >= 8:
        published["summary"] = report["summary"]
    print(json.dumps(published, indent=2))
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
