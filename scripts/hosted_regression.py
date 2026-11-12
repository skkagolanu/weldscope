"""Verify current hosted inference, synonyms and the exact reported conversation."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from langchain_core.documents import Document
from weldscope.generation.validation import check_answer
from check_hosted_app import find_chat_frame
from profile_hosted_latency import profile

EXACT_RECIPE = "i am in a ship that has fsw for welds. unfortunately i think there were lots of faults and now the joints that had defects are coming off. i would be drowning soon but i want to know how to cook butter chicken one last time before sinkign"


def inspect_response(app):
    assistant = app.locator('[data-testid="stChatMessage"]').last
    timing = assistant.locator('[data-testid="stExpander"]').filter(has_text="Latency breakdown")
    timing.locator("summary").click()
    code = timing.locator('[data-testid="stCode"] code')
    expect(code).to_be_visible(timeout=10000)
    diagnostics = json.loads(code.inner_text())
    assert diagnostics["app_revision"] == "quality-v2", "Updated application not serving"
    sources = []
    documents = []
    for expander in assistant.locator('[data-testid="stExpander"]').all():
        label = expander.locator("summary").inner_text()
        match = re.search(r"Source (\d+): (.+)", label)
        if match:
            expander.locator("summary").click()
            text = expander.locator('[data-testid="stText"]').inner_text()
            documents.append(Document(page_content=text))
            page = re.search(r"Page (\d+)", expander.inner_text())
            sources.append(
                dict(
                    number=int(match[1]),
                    source=match[2],
                    page=int(page[1]) if page else None,
                    passage_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
            )
    return diagnostics, sources, documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://weldscope.streamlit.app/")
    parser.add_argument("--browser-path")
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hosted-regression.json"))
    args = parser.parse_args()
    results = []
    report = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        url=args.url,
        passed=False,
        results=results,
        limitations="Public browser observations; no owner reboot or cold-start control. Structural checks are not semantic entailment or RAGAS scores.",
    )
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.browser_path)
            page = browser.new_page()
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            app = find_chat_frame(page, args.startup_timeout)
            questions = [
                "hi",
                "what do you do",
                "what is fsw",
                EXACT_RECIPE,
                "What is the TMAZ?",
                "How does it differ from the HAZ?",
                "Summarize the latest 2035 changes to the FSW standard.",
            ]
            for index, question in enumerate(questions):
                result = profile(page, app, question, 90, clear=index == 0)
                results.append(result)
                assert result["status"] == "complete", result
                diagnostics, sources, documents = inspect_response(app)
                result.update(diagnostics=diagnostics, sources=sources)
                if diagnostics["route"] == "research":
                    result["independent_structural_check"] = check_answer(
                        result["answer"], documents
                    )
                    assert not result["independent_structural_check"]["invalid_citations"]
                    assert sources, "No source passages for research query"
                    assert all(
                        Path(source["source"]).name.startswith("fsw_doc") for source in sources
                    )
                if index == 2:
                    assert diagnostics["route"] == "research"
                    assert (
                        diagnostics["token_usage"]
                        and diagnostics["token_usage"]["total_tokens"] > 0
                    )
                if index == 3:
                    assert "Recipes and cooking instructions are outside" in result["answer"]
                    assert diagnostics["route"] == "out_of_scope" and not sources
                    assert diagnostics["timings"]["generation_ms"] == 0
                    assert diagnostics["timings"]["retrieval_ms"] == 0
                    assert not diagnostics["token_usage"]
                if index in {4, 5}:
                    assert diagnostics["route"] == "research", "Valid domain synonym refused"
                print(
                    json.dumps(
                        dict(
                            question=question,
                            backend_total_ms=result["backend_total_ms"],
                            route=diagnostics["route"],
                        )
                    ),
                    flush=True,
                )
            browser.close()
            report["passed"] = True
    except Exception as error:
        report["failure"] = f"{type(error).__name__}: {str(error)[:400]}"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(report["failure"])


if __name__ == "__main__":
    main()
