# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **No build system or linter.** A Playwright end-to-end test exists — see **Testing**.

**Setup (first time):**
```bash
python -m venv .venv
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org streamlit pandas numpy scipy matplotlib seaborn yfinance
```
> Note: `--trusted-host` flags are needed on Windows machines with corporate/root SSL cert issues.

**Run app:**
```bash
.venv\Scripts\streamlit run efficient_frontier_app/efficient_frontier_app.py
```

**Workflow:** Pre-loaded CSVs for EM57.MI, VWCE.MI, SGLD.MI, BTC-EUR (daily + monthly) are included in `individual_indices_data/` — click **Run Analysis** immediately. To add or refresh tickers, use the sidebar "Download ETF Data" panel.

## Testing

**Requires the app to be running first** (`Run app` command above), then:

```bash
.venv\Scripts\python test_dashboard.py
```

For automated/non-interactive runs use `HEADLESS=1 .venv\Scripts\python test_dashboard.py` (plain `headless=False` hangs when not driven by a human).

> **Gotcha:** Streamlit caches imported modules in memory, so a still-running `streamlit.exe` serves **stale code** after you edit `ui_components.py`/etc. After edits, kill all `streamlit.exe`, confirm port 8501 has no LISTENING socket, then restart before re-testing.

Playwright drives a real Chromium browser, clicks Run Analysis, and verifies all 9 section headers render plus portfolio card metrics. Screenshots are saved to `test_screenshots/` (gitignored).

**Install Playwright** (already in venv; only needed once on a fresh clone):
```bash
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org playwright
# Chromium download requires disabling Node SSL verification on this machine:
$env:NODE_TLS_REJECT_UNAUTHORIZED = "0"; .venv\Scripts\playwright install chromium
```

## Git

Remote: https://github.com/LaRambla20/portfolio-optimization-dashboard_streamlit.git
Default branch: `master`

**Push requires Windows native cert store** (same SSL issue as pip):
```bash
git -c http.sslBackend=schannel push
```

## Architecture

Modular Streamlit dashboard split into 5 files inside `efficient_frontier_app/`.
- **`efficient_frontier_app/efficient_frontier_app.py`** — main entry point: sidebar inputs, derived parameters, orchestrates data loading and UI rendering.
- **`efficient_frontier_app/portfolio_calculations.py`** — pure math: portfolio performance, optimization (SciPy), Monte Carlo simulation, VaR calculations.
- **`efficient_frontier_app/data_handling.py`** — data operations: CSV loading, validation, price spike checks, merged dataframe building, return computations. Uses `@st.cache_data`.
- **`efficient_frontier_app/ui_components.py`** — UI renderers: 9 section functions (`render_load_etf_data`, `render_per_etf_analytics`, `render_etf_prices`, `render_rolling_returns`, `render_returns_statistics`, `render_input_portfolio_analysis`, `render_monte_carlo`, `render_scipy_ef`, `render_var_analysis`) plus shared helpers (`collect_portfolio_info`, `display_portfolio_cards`). All data passed as explicit parameters (C-like style).
- **`efficient_frontier_app/descriptions.py`** — single source of truth for the per-section "How to read this section" expanders (concept → markdown with KaTeX formulas); rendered via `render_section_help`.
(No legacy files currently on disk — the repo contains only the 5-file modular app.)

### Critical Return Type Split

The code maintains two separate return series:
- `returns` — user-selected type (`return_type` toggle: "logarithmic" or "simple") used **only for single-asset display metrics**
- `portfolio_returns_simple` — always simple returns, used for **all portfolio optimization** (Monte Carlo, SciPy, VaR)

This split fixes the mathematical error where `sum(weights * log_returns)` incorrectly computes portfolio log returns (log returns aren't additive across assets).

Key variables for optimization:
- `portfolio_returns_simple` — simple returns dataframe
- `portfolio_mean_returns` — mean of simple returns
- `portfolio_cov_matrix` — covariance of simple returns

**Annualisation (textbook MPT, linear):** annual return = `mean × N`, annual volatility = `std × √N` (N = periods/year: 252/52/12). This is an *arithmetic/expected* return, distinct from the geometric CAGR shown in §2.

### Section Structure

1. **Load ETF Data** — validates CSVs, checks price spikes, shows data availability gauge
2. **Per-ETF Analytics** — CAGR, simple/calendar-year returns, "Annualised Metrics by Look-back Period", **Max Drawdown** (always simple returns for consistency)
3. **ETF Prices** — raw and normalized (base = 1000) price charts (merged df is built upstream via inner join)
3b. **Rolling Returns** — moving-window returns (1/5/10 years) for **individual assets only** (the portfolio rolling-returns chart now lives in §5)
4. **Returns & Statistics** — covariance/correlation matrix **tables** (from simple returns), daily returns plot. The correlation **heatmap** moved to §5.
5. **Input Portfolio Analysis** — the user's allocation as a **buy-and-hold** (un-rebalanced) portfolio: cumulative-return chart (portfolio + per-asset overlays), annotated underwater curve, deepest-drawdown recovery + longest-underwater-stretch durations (calendar days, "≥ N days, ongoing" if unrecovered), headline metrics (avg annual return, annual vol, Sharpe, Sortino, max DD, historical VaR/CVaR), portfolio rolling-returns chart, and the asset correlation heatmap. See **Buy-and-hold basis** below.
6. **Monte Carlo Efficient Frontier** — random portfolio simulation, Sortino ratios
7. **SciPy Efficient Frontier** — optimization via `scipy.optimize.minimize`, efficient frontier line
8. **VaR Analysis** — per-period parametric (drift-free `σ·z`) and historical VaR/CVaR shown side by side, plus skew/excess-kurtosis fat-tail diagnostics. Historical figures guarded by a data-sufficiency check (`n_tail = round((1-alpha)*n)`). **No VaR frontier** (dropped: drift-free VaR ∝ volatility, so it was redundant with §7).

### Buy-and-hold basis (§5 only)

§5 is the **one place** in the app that does **not** rebalance. Every other portfolio figure (the "My Portfolio" cards on the frontier, §8 VaR) uses the constant-weight series `portfolio_returns_simple.dot(weights)` (rebalanced to target weights each period). §5 instead builds a *drifting* value series via `buy_and_hold_value_series(merged_df, tickers, weights)`: each asset rebased to its first price, `V_t = Σ wᵢ·Pᵢₜ/Pᵢ₀`, `V₀ = 1` (scale-invariant — only weight proportions matter). All §5 metrics derive from this single series, so the **same portfolio's return/risk/drawdown legitimately differ between §5 and §6–§8** — a `st.info` banner in §5 flags this. Annualisation still follows the textbook-linear convention (`mean × N`, `std × √N`).

### Data Format

CSV files in `individual_indices_data/` named `{ticker}_data_{period}.csv` (period: daily/weekly/monthly) with columns: `date`, `adj close`.

### VaR / CVaR Gotchas

- **`cvar(returns, alpha)` expects the tail quantile** (e.g., `alpha=0.05` for 95% CVaR), NOT the confidence level (`0.95`). The sidebar `alpha` is the confidence level — pass `1 - alpha` to `cvar()`.
- **Sign convention (loss = positive)**: VaR and `cvar()` (CVaR) report losses as **positive** numbers. §8 shows per-period parametric (drift-free `σ·z`) and historical VaR/CVaR side by side; the latter are guarded by a data-sufficiency check (`n_tail = round((1-alpha)*n)`). §5 reuses the same `n_tail≥5` guard for its (historical-only) buy-and-hold VaR/CVaR. CVaR for the portfolio cards is computed once, historically, at display time in `display_portfolio_cards(portfolios, alpha)`, so the whole app uses a single CVaR definition and sign.
- **Monte Carlo weights** use `np.random.dirichlet(np.ones(n))` for *uniform* simplex coverage — do **not** revert to normalising `np.random.random(n)` (biases toward equal weights). `np.random.seed(777)` (in the entry point) makes runs deterministic.

### Keeping descriptions in sync

When you change *how* a metric is computed, update its matching entry in `descriptions.py` — the per-section "How to read this section" text states the formula/assumptions and will otherwise silently drift from the code.

### Rendering gotchas (silent failures — verify in a browser, not just the source)

Requires **Streamlit ≥ 1.50** (uses `width="stretch"`; developed against 1.58).

- **Inline SVG (the data-availability gauge in `render_load_etf_data`) must use `st.markdown(html, unsafe_allow_html=True)`.** Do **not** use `st.html` — it runs input through DOMPurify (default config), which strips `<svg>` entirely, so the gauge renders as *nothing* with no error. `st.components.v1.html` (the old approach) is removed after 2026-06-01. `st.iframe` only embeds a URL, not an HTML/SVG string. After any change here, confirm the SVG is actually in the DOM (e.g. Playwright `svg:has(linearGradient#gaugeGrad)`); the deprecation-warning-gone check alone is not enough.
- **KaTeX `$$…$$` blocks in `descriptions.py` must sit on a single source line.** A hard line break inside the block makes Streamlit's markdown treat it as a paragraph break, splitting the delimiters so the formula fails to parse (renders as raw red error text). The test's gauge check locates the inline SVG (not an iframe) for this reason.

### Key Functions

**data_handling.py:**
- `compute_returns(merged_df, return_type)` — return series (simple or logarithmic)
- `compute_portfolio_returns_simple(merged_df)` — simple returns for portfolio optimization
- `compute_rolling_returns(merged_df, window_periods, return_type)` — rolling returns over moving window
- `build_merged_dataframe(tickers, folder_path, filename_suffix, filter_date)` — inner join of asset prices
- `check_price_spikes(tickers, folder_path, filename_suffix, filter_date)` — detects >60% price moves

**portfolio_calculations.py:**
- `portfolio_annualised_performance(weights, mean_returns, cov_matrix, ...)` — portfolio return/volatility
- `cvar(returns, alpha=0.05)` — historical CVaR; `alpha` is tail quantile
- `max_drawdown(returns)` — maximum drawdown
- `portfolio_cvar(weights, returns, alpha)` / `portfolio_max_drawdown(weights, returns)` — portfolio-level CVaR and Max DD
- `random_portfolios(num, mean_returns, cov_matrix, ...)` — Monte Carlo simulation
- `random_portfolios_sortino(num, ...)` — returns 5 rows (std, ret, sharpe, sortino, downside_dev); used in sections 6 & 7
- **§5 buy-and-hold helpers:** `buy_and_hold_value_series(merged_df, tickers, weights)` — drifting (un-rebalanced) value series, `V₀=1`; `underwater_episodes(value)` — list of peak/trough/recovery episodes; `deepest_drawdown_episode(value)` / `longest_underwater_episode(value)` (→ `(episode, days, ongoing)`); `downside_deviation_series(returns, N, rf)` — Sortino downside dev on a precomputed return series

**ui_components.py:**
- `render_rolling_returns(rolling_returns, tickers, rolling_window_years, return_type)` — renders 3b (individual assets only; no portfolio chart, no portfolio arg)
- `render_input_portfolio_analysis(merged_df, portfolio_returns_simple, tickers, my_portfolio_allocation, annualisation_factor, risk_free_rate, alpha, window_periods, rolling_window_years)` — renders §5 (buy-and-hold); `_fmt_period(days, ongoing)` helper formats underwater durations
- `collect_portfolio_info()` — returns dict with max_dd and port_returns (CVaR is computed later from port_returns)
- `display_portfolio_cards(portfolios, alpha)` — shows 6 metrics incl. Max Drawdown and CVaR at the user's confidence level (label updates with α)
