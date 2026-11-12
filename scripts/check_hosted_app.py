"""Check the rendered app so a sleep/error page cannot pass as healthy."""

import argparse
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from playwright.sync_api import Error, expect, sync_playwright


def find_chat_frame(page, timeout):
    """Community Cloud wraps the app in an iframe; local servers use the main frame."""
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        for frame in page.frames:
            try:
                if frame.locator('[data-testid="stChatInput"] textarea').is_visible():
                    return frame
            except Error:
                pass  # A frame can be replaced during startup.
        page.wait_for_timeout(500)
    raise RuntimeError("No rendered chat input appeared in the app or its frames.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://weldscope.streamlit.app/")
    parser.add_argument("--timeout", type=int, default=180, help="Startup timeout in seconds")
    parser.add_argument("--browser-path", help="Optional installed Chromium executable")
    parser.add_argument(
        "--startup-only", action="store_true", help="Check startup without sending a greeting"
    )
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser_path)
        page = browser.new_page()
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            app = find_chat_frame(page, args.timeout)
            if not args.startup_only:
                chat_input = app.locator('[data-testid="stChatInput"] textarea')
                # Avoid sending a greeting to an older deployment that routes it to inference.
                expect(chat_input).to_have_attribute(
                    "placeholder", "Ask about welding, sources, or WeldScope...", timeout=30000
                )
                chat_input.fill("hi")
                chat_input.press("Enter")
                expect(app.locator('[data-testid="stChatMessage"]').last).to_contain_text(
                    "Hi! I'm WeldScope", timeout=30000
                )
                expect(app.get_by_text("Local reply · Total", exact=False).last).to_be_visible(
                    timeout=10000
                )
                chat_input.fill(
                    "I am on a ship that was friction stir welded and we might sink soon."
                )
                chat_input.press("Enter")
                expect(app.locator('[data-testid="stChatMessage"]').last).to_contain_text(
                    "I can't assess a real vessel's condition", timeout=10000
                )
                chat_input.fill("Given that FSW explanation, send me a recipe for pasta.")
                chat_input.press("Enter")
                expect(app.locator('[data-testid="stChatMessage"]').last).to_contain_text(
                    "Recipes and cooking instructions are outside this chatbot's corpus",
                    timeout=10000,
                )
            print(
                "Healthy: rendered chat"
                + ("" if args.startup_only else " and local greeting")
                + " at",
                args.url,
            )
        except Exception:
            Path("artifacts").mkdir(exist_ok=True)
            page.screenshot(path="artifacts/hosted-app.png", full_page=True)
            texts = []
            for frame in page.frames:
                try:
                    url = urlsplit(frame.url)
                    texts.append(
                        url.netloc
                        + url.path
                        + "\n"
                        + frame.locator("body").inner_text(timeout=3000)
                    )
                except Error:
                    pass
            Path("artifacts/hosted-app.txt").write_text("\n\n".join(texts))
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
