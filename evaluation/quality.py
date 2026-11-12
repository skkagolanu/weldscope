"""Native RAGAS metrics with explicit input, failure and coverage reporting."""

import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# Evaluation samples are not sent to RAGAS analytics.
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

from ragas import SingleTurnSample  # noqa: E402
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def page_id(source, page):
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise ValueError("Evidence pages must be one-based positive integers")
    return f"{Path(source).name}#page={page}"


async def retrieval_scores(rankings_path, dataset_path, thresholds=None):
    rankings = json.loads(Path(rankings_path).read_text())
    dataset = json.loads(Path(dataset_path).read_text())
    dataset_hash = digest(dataset_path)
    if rankings["provenance"]["dataset_sha256"] != dataset_hash:
        raise ValueError("Ranking and dataset fingerprints differ")
    cases = {case["id"]: case for case in dataset["cases"] if case["answerable"]}
    expected_variants = {"bm25", "vector", "hybrid", "reranked"}
    seen = set()
    rows = []
    metrics = {
        "id_context_precision": IDBasedContextPrecision(),
        "id_context_recall": IDBasedContextRecall(),
    }
    for ranking in rankings["results"]:
        key = (ranking["id"], ranking["variant"])
        if key in seen or key[0] not in cases or key[1] not in expected_variants:
            raise ValueError("Unexpected or duplicate ranking row")
        seen.add(key)
        case = cases[key[0]]
        sample = SingleTurnSample(
            retrieved_context_ids=[
                page_id(item["source"], item["page"]) for item in ranking["retrieved"][:5]
            ],
            reference_context_ids=[
                page_id(item["source"], item["page"]) for item in case["evidence"]
            ],
        )
        row = dict(id=key[0], variant=key[1], category=case["category"])
        for name, metric in metrics.items():
            value = float(await metric.single_turn_ascore(sample))
            if not math.isfinite(value):
                raise ValueError(f"Non-finite RAGAS metric: {key} {name}")
            row[name] = value
        rows.append(row)
    if len(seen) != len(cases) * len(expected_variants):
        raise ValueError("Incomplete variant/question coverage")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = {
        variant: {name: mean(row[name] for row in data) for name in metrics}
        for variant, data in grouped.items()
    }
    failures = []
    if thresholds is not None:
        if thresholds["dataset_sha256"] != dataset_hash:
            failures.append("Dataset fingerprint changed; review/version thresholds")
        for variant, floors in thresholds["minimum"].items():
            for name, floor in floors.items():
                if summary[variant][name] < floor:
                    failures.append(f"{variant}.{name} below {floor}")
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        framework="ragas",
        ragas_version=importlib.metadata.version("ragas"),
        dataset_sha256=dataset_hash,
        rankings_sha256=digest(rankings_path),
        lock_sha256=digest(Path(__file__).parent / "requirements.lock"),
        evaluator_sha256=digest(__file__),
        metric_classes={name: type(metric).__name__ for name, metric in metrics.items()},
        protocol="Native ID-based set precision/recall over unique physical PDF pages in top-five chunks; provisional positive anchors, not exhaustive relevance labels. No MRR/nDCG or semantic scores implied.",
        model_provider_calls=0,
        answerable_cases=len(cases),
        results=rows,
        summary=summary,
        gate=dict(evaluated=thresholds is not None, passed=not failures, failures=failures),
    )


async def answer_scores(samples, llm, embeddings, *, minimum_interval=0):
    """Use native collections metrics; skipped references and failed metrics stay explicit."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        DomainSpecificRubrics,
        FactualCorrectness,
    )

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm),
        "scope_adherence": DomainSpecificRubrics(
            llm=llm,
            rubrics={
                "score1_description": "The assistant follows an unrelated request, invents unavailable facts, or provides real-equipment/emergency advice.",
                "score2_description": "The response substantially drifts outside the FSW corpus or fails to refuse an unsupported task.",
                "score3_description": "The response mostly respects scope but includes unsupported extra advice or an unclear refusal.",
                "score4_description": "The response stays grounded in the corpus and appropriately states important evidence gaps.",
                "score5_description": "The response directly handles the request within the FSW corpus or clearly refuses an off-corpus/unsupported task, without unrelated, operational or emergency advice.",
            },
        ),
    }
    rows = []
    for sample in samples:
        if (
            not sample.get("question")
            or not sample.get("response")
            or (sample.get("answerable", True) and not sample.get("contexts"))
        ):
            raise ValueError("Semantic samples require question, response and retrieved contexts")
        arguments = {
            "faithfulness": dict(
                user_input=sample["question"],
                response=sample["response"],
                retrieved_contexts=sample["contexts"],
            ),
            "answer_relevancy": dict(user_input=sample["question"], response=sample["response"]),
            "context_precision": dict(
                user_input=sample["question"],
                reference=sample.get("reference"),
                retrieved_contexts=sample["contexts"],
            ),
            "context_recall": dict(
                user_input=sample["question"],
                reference=sample.get("reference"),
                retrieved_contexts=sample["contexts"],
            ),
            "factual_correctness": dict(
                response=sample["response"], reference=sample.get("reference")
            ),
            "scope_adherence": dict(
                user_input=sample["question"],
                response=sample["response"],
                retrieved_contexts=sample.get("contexts") or None,
            ),
        }
        row = dict(id=sample["id"], scores={}, errors={}, skipped={})
        for name, metric in metrics.items():
            if name != "scope_adherence" and not sample.get("answerable", True):
                row["skipped"][name] = "unsupported_case_use_scope_rubric"
                continue
            if name in {
                "context_precision",
                "context_recall",
                "factual_correctness",
            } and not sample.get("reference"):
                row["skipped"][name] = "no_reference"
                continue
            try:
                result = await metric.ascore(**arguments[name])
                value = float(result.value)
                if not math.isfinite(value):
                    raise ValueError("Non-finite score")
                lower, upper = metric.allowed_values
                if not lower <= value <= upper:
                    raise ValueError("Score outside native metric range")
                row["scores"][name] = value
            except Exception as error:
                # Raw provider errors may contain request data; retain only the class.
                row["errors"][name] = type(error).__name__
            if minimum_interval:
                await asyncio.sleep(minimum_interval)
        rows.append(row)
    summary = {}
    for name in metrics:
        values = [row["scores"][name] for row in rows if name in row["scores"]]
        summary[name] = dict(
            mean=mean(values) if values else None,
            scored=len(values),
            failed=sum(name in row["errors"] for row in rows),
            skipped=sum(name in row["skipped"] for row in rows),
        )
    return dict(
        framework="ragas",
        ragas_version=importlib.metadata.version("ragas"),
        complete=bool(rows) and not any(row["errors"] for row in rows),
        metric_classes={name: type(metric).__name__ for name, metric in metrics.items()},
        summary=summary,
        results=rows,
    )
