"""Measure fresh hosted turns; distinguish browser-visible and backend timings."""

import argparse
import json
import re
from pathlib import Path
from time import perf_counter
from playwright.sync_api import Error, expect, sync_playwright
from check_hosted_app import find_chat_frame


def profile(page, app, question, timeout, *, clear=True):
    messages = app.locator('[data-testid="stChatMessage"]')
    if clear:
        app.get_by_role("button", name="Clear Conversation").click()
        expect(messages).to_have_count(0, timeout=15000)
    expected_count = messages.count() + 2
    box = app.locator('[data-testid="stChatInput"] textarea')
    box.fill(question)
    started = perf_counter()
    box.press("Enter")
    first_visible = None
    answer = ""
    caption = ""
    while perf_counter() - started < timeout:
        try:
            if messages.count() < expected_count:
                page.wait_for_timeout(50)
                continue
            assistant = messages.last
            markdown = assistant.locator('[data-testid="stMarkdownContainer"]')
            if markdown.count():
                answer = markdown.first.inner_text().strip()
                if answer and answer not in {"Thinking...", "▌"} and first_visible is None:
                    first_visible = perf_counter() - started
            text = assistant.inner_text()
            total = re.search(r"Total ([\d,]+) ms", text)
            if total:
                caption = next(
                    line for line in text.splitlines() if "Total " in line and " ms" in line
                )
                return dict(
                    question=question,
                    status="complete",
                    browser_total_ms=round((perf_counter() - started) * 1000, 1),
                    browser_first_visible_ms=(
                        round(first_visible * 1000, 1) if first_visible else None
                    ),
                    backend_total_ms=int(total[1].replace(",", "")),
                    timing_caption=caption,
                    answer_words=len(answer.split()),
                    answer=answer,
                )
            if assistant.locator('[data-testid="stAlert"]').count():
                return dict(question=question, status="error", error=text)
        except Error:
            pass  # Streamlit can replace elements while a turn reruns.
        page.wait_for_timeout(50)
    return dict(
        question=question, status="timeout", elapsed_ms=round((perf_counter() - started) * 1000, 1)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://weldscope.streamlit.app/")
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument(
        "--conversation", action="store_true", help="Keep earlier turns as session context"
    )
    parser.add_argument("--browser-path")
    parser.add_argument("--output", default="artifacts/latency-profile.json")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser_path)
        page = browser.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        app = find_chat_frame(page, 60)
        results = []
        for index, question in enumerate(args.question):
            result = profile(
                page, app, question, args.timeout, clear=not args.conversation or index == 0
            )
            results.append(result)
            print(json.dumps(result), flush=True)
        browser.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(url=args.url, results=results), indent=2))


if __name__ == "__main__":
    main()
