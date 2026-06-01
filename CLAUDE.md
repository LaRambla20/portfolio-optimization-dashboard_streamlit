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

**Unit tests (no running app, no network needed):** `.venv\Scripts\python test_total_return_synthesis.py` covers the total-return reconstruction math (yield recovery, splice continuity, FX caveat, CSV save/read round-trip). It also runs one live `^GSPC` vs `^SP500TR` check that SKIPs gracefully if the network/SSL is unavailable.

> **yfinance SSL gotcha:** yfinance uses `curl_cffi`, so the pip `--trusted-host` and git `schannel` workarounds do **not** apply to it. The app's `run_download`/`run_total_return_reconstruction` use a bare `yf.Ticker` (fine on the real machine). To drive yfinance for local validation on an SSL-intercepting machine, pass a relaxed session: `yf.Ticker(sym, session=curl_cffi.requests.Session(impersonate="chrome", verify=False))`.

**Boot-check (fast path):** for sidebar-only / non-render changes, launch the app headless and grep the boot log for `error|traceback` instead of the full Playwright run — e.g. `streamlit run ... --server.headless true > boot.log 2>&1 &` then inspect `boot.log`.

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

1. **Load ETF Data** — validates CSVs, checks price spikes, shows data availability gauge, a 💱 warning if any series isn't EUR, and (for `{ETF}_EXT` files) a 🧬 caption flagging reconstructed total-return rows
2. **Per-ETF Analytics** — CAGR, simple/calendar-year returns, "Annualised Metrics by Look-back Period", **Max Drawdown** (always simple returns for consistency)
3. **ETF Prices** — raw and normalized (base = 1000) price charts (merged df is built upstream via inner join)
3b. **Rolling Returns** — moving-window returns (1/5/10 years) for **individual assets only** (the portfolio rolling-returns chart now lives in §5)
4. **Returns & Statistics** — covariance/correlation matrix **tables** (from simple returns), daily returns plot. The correlation **heatmap** moved to §5.
5. **Input Portfolio Analysis** — the user's allocation as a **buy-and-hold** (un-rebalanced) portfolio: opens with an allocation pie ("Your Allocation", non-zero weights only) + normalised-weights table, then a cumulative-return chart (portfolio + per-asset overlays), annotated underwater curve, deepest-drawdown recovery + longest-underwater-stretch durations (calendar days, "≥ N days, ongoing" if unrecovered), headline metrics (avg annual return, annual vol, Sharpe, Sortino, max DD, historical VaR/CVaR), portfolio rolling-returns chart, and the asset correlation heatmap. See **Buy-and-hold basis** below.
6. **Monte Carlo Efficient Frontier** — random portfolio simulation, Sortino ratios
7. **SciPy Efficient Frontier** — optimization via `scipy.optimize.minimize`, efficient frontier line
8. **VaR Analysis** — per-period parametric (drift-free `σ·z`) and historical VaR/CVaR shown side by side, plus skew/excess-kurtosis fat-tail diagnostics. Historical figures guarded by a data-sufficiency check (`n_tail = round((1-alpha)*n)`). **No VaR frontier** (dropped: drift-free VaR ∝ volatility, so it was redundant with §7).

### Buy-and-hold basis (§5 only)

§5 is the **one place** in the app that does **not** rebalance. Every other portfolio figure (the "My Portfolio" cards on the frontier, §8 VaR) uses the constant-weight series `portfolio_returns_simple.dot(weights)` (rebalanced to target weights each period). §5 instead builds a *drifting* value series via `buy_and_hold_value_series(merged_df, tickers, weights)`: each asset rebased to its first price, `V_t = Σ wᵢ·Pᵢₜ/Pᵢ₀`, `V₀ = 1` (scale-invariant — only weight proportions matter). All §5 metrics derive from this single series, so the **same portfolio's return/risk/drawdown legitimately differ between §5 and §6–§8** — a `st.info` banner in §5 flags this. Annualisation still follows the textbook-linear convention (`mean × N`, `std × √N`).

### Currency handling (single base currency = EUR)

**The app does no FX conversion anywhere except where noted here — every `adj close` is assumed already in EUR** (price-chart axes are hard-labeled `[EUR]`, portfolio weights derive from EUR market values). Feeding in a USD/GBP-priced series silently treats those prices as EUR, so its returns/vol and especially **correlations** (every same-currency asset shares a hidden FX factor) come out in the wrong numeraire, biasing the covariance matrix → frontier and VaR. No error is raised. Verified live: US-listed `IVV` shows 14.57%/yr in USD but 13.99%/yr once converted to EUR (matching the EUR-listed `SXR8.DE` at 13.84%).

Two guards:
- **Download auto-converts to EUR by default** (`run_download(..., convert_to_eur=True)`): detects the ticker's currency (`detect_currency`, via `fast_info`/`info`), and if not EUR scales every price column by Yahoo's `{CCY}EUR=X` rate (`fetch_eur_multiplier` → `apply_eur_conversion`; handles GBp/pence by `/100`). The sidebar **"Keep native currency (don't convert to EUR)"** checkbox (default off) is the escape hatch that stores raw prices instead. Conversion needs two network calls (the sniff + the FX series); if either fails the native series is stored and §1 warns — so the check below is the safety net, not redundant. Prefer the EUR-listed share class (`.MI`/`.DE`/`.AS`) when one exists — it's the same thing.
- **§1 currency check** (`read_currency_info` → `render_load_etf_data`): warns if any loaded series isn't EUR. Every download writes a `currency` column (the stored currency: `EUR` when native or converted, else e.g. `USD`), so the check is offline; legacy CSVs lacking the column fall back to a cached network sniff.

The on-disk `currency` column is a constant per file and, like the reconstruction tag columns, is ignored by the 2-column/`usecols` readers.

### Total-return reconstruction (extending ETFs with index history)

Accumulating ETFs (VWCE, EM57, …) only span a few years; the **price-return** indices they track (most Yahoo `^` tickers, all Stooq indices) run for decades but **exclude reinvested dividends**, so splicing one straight in front of an ETF biases every long-run figure low. The sidebar **🧬 Total-return reconstruction** panel (inside *Download settings*) fixes this:

1. For each job — *(index ticker, calibrating ETF, optional FX ticker)* — it downloads all three at the selected interval(s).
2. `synthesize_total_return` converts the index to the ETF's currency (`index / eurusd` — **FX first**, or `q_hat` absorbs FX drift), then calibrates the missing yield geometrically from the overlap: annual gross-up `f = (1+g_etf)/(1+g_index)`, reported `q_hat = f − 1` (this also absorbs TER/tracking — intentional, so the synthetic tail meets the real ETF seamlessly).
3. Older index history is grossed up **on returns** (`f**(1/N)` per period) and **chained** to the real ETF (scaled to meet it at `join_date` — never raw-level concatenation), then saved as `{ETF}_EXT_data_{period}.csv` with the `synthetic` / `recon_yield` tag columns.

Add the new `{ETF}_EXT` ticker to **My Portfolio** to analyze it. §1 shows a 🧬 caption flagging which rows are reconstructed (estimate, not measured).

**The index must track the same underlying as the ETF** (e.g. `^GSPC` with an S&P 500 ETF). The calibration makes the index match the ETF over the overlap, so pairing mismatched underlyings (S&P 500 vs FTSE All-World) folds their *performance gap* into `q_hat` — `^GSPC`→`VWCE.MI` yields a nonsensical **−1.2%/yr** because US stocks outran the world 2020–2025. The runner emits a `⚠️` log line when `q_hat` lands outside ~0–6%/yr (also catches data quirks: splits, wrong FX, distributing-vs-accumulating share classes — e.g. `SXR8.DE` returned an implausibly low +0.1%). Tests: `test_total_return_synthesis.py` (deterministic math/splice/FX + save/read round-trip, plus a live `^GSPC` vs `^SP500TR` check that recovers q≈1.9%). Note: the live test fetches via a `curl_cffi` session with `verify=False` to bypass this machine's SSL interception — `run_total_return_reconstruction` itself uses a bare `yf.Ticker` like `run_download`, matching existing behavior.

### Data Format

CSV files in `individual_indices_data/` named `{ticker}_data_{period}.csv` (period: daily/weekly/monthly) with columns: `date`, `adj close`.

**Reconstructed (total-return) files** — `{ETF}_EXT_data_{period}.csv` — carry two extra columns: `synthetic` (bool: `True` for rows reconstructed from the index *before* the ETF's first date, `False` for real ETF rows) and `recon_yield` (the calibrated annual gross-up `q_hat`, filled only on synthetic rows). Files downloaded by `run_download` also carry a `currency` column (see **Currency handling**). All other readers slice the first two columns (`iloc[:, :2]`) or pass `usecols`, so they ignore the extras — only `read_synthetic_info` (§1 badge) and `read_currency_info` (§1 currency check) read them. See **Total-return reconstruction** and **Currency handling** below.

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
- **Total-return reconstruction:** `synthesize_total_return(price_index, etf, eurusd, periods_per_year)` — calibrate `q_hat` from the ETF overlap + return the spliced EUR series; `build_reconstructed_frame(index_prices, etf_prices, fx_prices, periods_per_year)` — shape it into the `date,adj close,synthetic,recon_yield` CSV frame; `run_total_return_reconstruction(jobs, intervals, output_dir, log_queue)` — threaded download+splice+save (mirrors `run_download`); `read_synthetic_info(...)` — per-ticker tag metadata for the §1 badge
- **Currency:** `run_download(..., convert_to_eur=True)` — downloads, auto FX-converts non-EUR to EUR (unless overridden), writes a `currency` column; `detect_currency(symbol, yf)` / `fetch_eur_multiplier(currency, yf_interval, end_date, yf)` / `apply_eur_conversion(data, eur_multiplier)` — the conversion helpers (last one is pure, unit-tested); `read_currency_info(tickers, folder_path, filename_suffix)` — resolves each ticker's currency (stored column, else cached network sniff) for the §1 check

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
