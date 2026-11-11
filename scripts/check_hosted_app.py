"""Check the rendered app so a sleep/error page cannot pass as healthy."""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://rag-demo-skullkrak7.streamlit.app/")
    parser.add_argument("--timeout", type=int, default=180, help="Startup timeout in seconds")
    parser.add_argument("--browser-path", help="Optional installed Chromium executable")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser_path)
        page = browser.new_page()
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
            page.locator('[data-testid="stChatInput"] textarea').wait_for(
                state="visible", timeout=args.timeout * 1000
            )
            print("Healthy: rendered chat input is available at", args.url)
        except Exception:
            Path("artifacts").mkdir(exist_ok=True)
            page.screenshot(path="artifacts/hosted-app.png", full_page=True)
            Path("artifacts/hosted-app.txt").write_text(page.locator("body").inner_text())
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
