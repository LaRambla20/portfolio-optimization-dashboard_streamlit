"""
Playwright test for the Efficient Frontier dashboard.
Run with: .venv/Scripts/python test_dashboard.py
Requires the Streamlit app to be running on localhost:8501.
Screenshots are saved to test_screenshots/.
"""

import os
from playwright.sync_api import sync_playwright, expect

BASE_URL = "http://localhost:8501"
SCREENSHOT_DIR = "test_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Set HEADLESS=1 for automated/CI runs (no visible window, more reliable);
# default is a visible browser with slow_mo so you can watch it locally.
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

SECTIONS = [
    "1. Load ETF Data",
    "2. Per-ETF Analytics",
    "3. ETF Prices",
    "3b. Rolling Returns",
    "4. Returns & Statistics",
    "5. Input Portfolio Analysis",
    "6. Monte Carlo Efficient Frontier",
    "7. Scipy Efficient Frontier",
    "8. Value at Risk",
]


def shot(page, name):
    path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
    page.screenshot(path=path, full_page=False)
    print(f"  [screenshot] {path}")


def wait_for_streamlit(page):
    """Wait until Streamlit is no longer in a running state."""
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]') || "
        "document.querySelector('[data-testid=\"stStatusWidget\"]').innerText === ''",
        timeout=120_000,
    )


def test_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, slow_mo=0 if HEADLESS else 100)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})

        # ── 1. Load the app ──────────────────────────────────────────────────
        print("Loading app...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30_000)
        page.wait_for_selector("h1", timeout=15_000)
        shot(page, "01_initial_load")
        print("  OK: page loaded, title visible")

        # ── 2. Click Run Analysis ────────────────────────────────────────────
        print("Clicking Run Analysis...")
        run_btn = page.get_by_role("button", name="Run Analysis")
        run_btn.click()
        print("  Waiting for full analysis to complete (Monte Carlo takes ~30-60s)...")
        # Wait for the final success message rendered by render_var_analysis
        page.wait_for_selector("text=Analysis complete", timeout=180_000)
        wait_for_streamlit(page)
        shot(page, "02_after_run")
        print("  OK: analysis complete")

        # ── 3. Verify all section headers rendered ───────────────────────────
        print("Checking section headers...")
        for section in SECTIONS:
            locator = page.locator(f"text={section}").first
            expect(locator).to_be_visible(timeout=15_000)
            print(f"  OK: {section}")

        # ── 4. Check data availability gauge rendered (SVG) ──────────────────
        # The gauge is an inline SVG rendered via st.markdown (no longer an iframe,
        # since components.html was dropped). Locate it by the gradient id it defines.
        print("Checking data availability gauge...")
        expect(page.locator("svg:has(linearGradient#gaugeGrad)").first).to_be_attached(timeout=10_000)
        print("  OK: gauge SVG present")

        # ── 5. Scroll through sections and screenshot ────────────────────────
        print("Scrolling and screenshotting sections...")
        for i, section in enumerate(SECTIONS, start=3):
            header = page.locator(f"text={section}").first
            header.scroll_into_view_if_needed()
            page.wait_for_timeout(500)
            shot(page, f"{i:02d}_section_{section[:20].replace(' ', '_').replace('.', '')}")
            print(f"  OK: {section}")

        # ── 6. Check portfolio cards render in Monte Carlo ───────────────────
        # NB: target cards by their SUMMARY text. The "How to read this section"
        # help expanders also contain "My Portfolio" in their body, so a plain
        # text= / has_text= selector would wrongly match a (collapsed) help expander.
        print("Checking portfolio cards...")
        my_card_summary = page.locator("summary", has_text="My Portfolio").first
        my_card_summary.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        expect(my_card_summary).to_be_visible()
        expect(page.locator("summary", has_text="Max Sharpe Portfolio").first).to_be_visible()
        expect(page.locator("summary", has_text="Min Volatility Portfolio").first).to_be_visible()
        print("  OK: portfolio cards visible")

        # ── 7. Expand a portfolio card and check metrics ─────────────────────
        print("Expanding My Portfolio card...")
        card = page.locator("details").filter(
            has=page.locator("summary", has_text="My Portfolio")
        ).first
        card.scroll_into_view_if_needed()
        card.locator("summary").click()
        page.wait_for_timeout(1000)
        # Metric labels inside st.columns() have overflow:hidden parents so
        # Playwright marks them "hidden" even when rendered — use to_be_attached.
        expect(page.locator("text=Average annual return").first).to_be_attached(timeout=10_000)
        expect(page.locator("text=Sharpe Ratio").first).to_be_attached(timeout=5_000)
        expect(page.locator("text=Max Drawdown").first).to_be_attached(timeout=5_000)
        expect(page.locator("text=CVaR (95%)").first).to_be_attached(timeout=5_000)
        shot(page, "10_portfolio_card_expanded")
        print("  OK: card expanded with metrics")

        # ── 8. Final full-page screenshot ────────────────────────────────────
        page.keyboard.press("End")
        page.wait_for_timeout(500)
        shot(page, "11_bottom_of_page")

        print("\nAll tests passed.")
        browser.close()


if __name__ == "__main__":
    test_dashboard()
