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

**Workflow:** Pre-loaded CSVs for EM57.MI, VWCE.MI, SGLD.MI, IMIE.MI, DBMF, BTC-EUR (daily + monthly) are included in `individual_indices_data/` — click **Run Analysis** immediately. To add or refresh tickers, use the sidebar "Download ETF Data" panel.

## Testing

**Requires the app to be running first** (`Run app` command above), then:

```bash
.venv\Scripts\python test_dashboard.py
```

For automated/non-interactive runs use `HEADLESS=1 .venv\Scripts\python test_dashboard.py` (plain `headless=False` hangs when not driven by a human).

> **Gotcha:** Streamlit caches imported modules in memory, so a still-running `streamlit.exe` serves **stale code** after you edit `ui_components.py`/etc. After edits, kill all `streamlit.exe`, confirm port 8501 has no LISTENING socket, then restart before re-testing.

Playwright drives a real Chromium browser, clicks Run Analysis, and verifies all 8 section headers render plus portfolio card metrics. Screenshots are saved to `test_screenshots/` (gitignored).

> **Gotcha:** the test's "done" signal is `st.success(" Analysis complete!")`, rendered **last** in `efficient_frontier_app.py` after the final section. Remove or relocate it (e.g. when merging/reordering sections) and `test_dashboard.py` hangs 180s then fails on `wait_for_selector("text=Analysis complete")` — keep it as the final render. The test's `SECTIONS` list must also match the rendered `st.header` strings exactly (incl. the section count).

**Unit tests (no running app, no network needed):** `.venv\Scripts\python test_total_return_synthesis.py` covers the total-return reconstruction math (yield recovery, splice continuity, FX caveat, CSV save/read round-trip; plus one live `^GSPC` vs `^SP500TR` check that SKIPs gracefully offline). `.venv\Scripts\python test_rebalancing.py` covers `rebalanced_value_series` (Never == buy-and-hold, Every-period == constant-weight compounded, periodic-reset continuity).

> **Adding a unit test:** plain assert-based script — `sys.path.insert(0, "efficient_frontier_app")`, functions that `assert` and `print("... OK")`, driven by an `if __name__ == "__main__"` runner. No pytest. Run with `.venv\Scripts\python test_X.py`.

> **yfinance SSL gotcha:** yfinance uses `curl_cffi`, so the pip `--trusted-host` and git `schannel` workarounds do **not** apply to it. The app's `run_download`/`run_total_return_reconstruction` use a bare `yf.Ticker` (fine on the real machine). To drive yfinance for local validation on an SSL-intercepting machine, pass a relaxed session: `yf.Ticker(sym, session=curl_cffi.requests.Session(impersonate="chrome", verify=False))`.

**Boot-check (fast path):** for sidebar-only / non-render changes, launch the app headless and grep the boot log for `error|traceback` instead of the full Playwright run — e.g. `streamlit run ... --server.headless true > boot.log 2>&1 &` then inspect `boot.log`.

> **PowerShell boot-check caveat:** that bash redirection misleads on PowerShell — Streamlit's normal Uvicorn startup line surfaces in the log as a `NativeCommandError`/`RemoteException` (not a real error), `boot.log` is written UTF-16 (garbles grep), and a background run reports **exit 255** when force-killed (also not a failure). Treat **port 8501 LISTENING** as the success signal, not a clean log. Stop the app with `Get-Process streamlit | Stop-Process -Force` and confirm the port is freed.

> **Clean PowerShell boot-check:** `Start-Process .venv\Scripts\streamlit.exe -ArgumentList 'run','efficient_frontier_app/efficient_frontier_app.py','--server.headless','true' -WindowStyle Hidden`, then poll `Get-NetTCPConnection -LocalPort 8501 -State Listen` in a short loop (a returned row = booted). Avoids the log-file garbling entirely.

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
- **`efficient_frontier_app/data_handling.py`** — data operations: CSV loading, validation, stock-split detection + price-anomaly checks, merged dataframe building, return computations. Uses `@st.cache_data`.
- **`efficient_frontier_app/ui_components.py`** — UI renderers: 8 section functions (`render_load_etf_data`, `render_per_etf_analytics`, `render_etf_prices`, `render_rolling_returns`, `render_returns_statistics`, `render_input_portfolio_analysis`, `render_monte_carlo`, `render_scipy_ef`) plus shared helpers (`render_tail_risk` — §5's tail-risk subsection, `collect_portfolio_info`, `display_portfolio_cards`). All data passed as explicit parameters (C-like style).
- **`efficient_frontier_app/descriptions.py`** — single source of truth for the per-section "How to read this section" expanders (concept → markdown with KaTeX formulas); rendered via `render_section_help`.
(No legacy files currently on disk — the repo contains only the 5-file modular app.)

### Simple returns everywhere (no return-type toggle)

**The app uses simple (arithmetic) returns throughout** — there is no log/simple toggle. This was a deliberate simplification for interpretability: every figure means "actual realised percentage change," and the dashboard never asks the user to reason about which return type a section is in. A single series, `portfolio_returns_simple` (= `merged_df` prices `.pct_change().dropna()`), backs §4's distribution stats/plot **and** all portfolio optimization (Monte Carlo, SciPy, VaR); §3b rolling returns are simple cumulative window returns.

Using simple returns for portfolio math is also the *correct* choice, not just the simple one: `sum(weights * log_returns)` does not equal the portfolio log return (log returns aren't additive across assets), so optimization must use simple returns regardless. (Log returns have a textbook edge only for single-asset distribution-shape analysis, and at daily/monthly frequency they barely differ from simple — not worth a toggle here.)

Key variables for optimization:
- `portfolio_returns_simple` — simple returns dataframe
- `portfolio_mean_returns` — mean of simple returns
- `portfolio_cov_matrix` — covariance of simple returns

**Annualisation (textbook MPT, linear):** annual return = `mean × N`, annual volatility = `std × √N` (N = periods/year: 252/52/12). This is an *arithmetic/expected* return, distinct from the geometric CAGR shown in §2.

### Section Structure

1. **Load ETF Data** — validates CSVs, reports recorded stock splits (📐, from the `stock splits` column — informational, since Adj Close is already split-adjusted) and flags statistically anomalous price moves (⚠️, robust MAD z-score), shows data availability gauge, a 💱 warning if any series isn't EUR, and (for `{ETF}_EXT` files) a 🧬 caption flagging reconstructed total-return rows
2. **Per-ETF Analytics** — CAGR, simple/calendar-year returns, "Annualised Metrics by Look-back Period", **Max Drawdown** (always simple returns for consistency)
3. **ETF Prices** — raw and normalized (base = 1000) price charts (merged df is built upstream via inner join)
3b. **Rolling Returns** — moving-window returns (1/2/3/5/7/10 years) for **individual assets only** (the portfolio rolling-returns chart now lives in §5)
4. **Per-Asset Returns & Statistics** — per-asset min/max/mean/**median**/std summary (per-period simple returns; the mean−median gap reads skew), per-asset Sortino, covariance/correlation matrix **tables** (from simple returns), per-period returns plot. **Not subject to the rebalancing selector** — purely per-asset, no portfolio aggregate. The correlation **heatmap** moved to §5. (Per-period **median** is not annualised — quantiles aren't additive, so no `×N`; the *portfolio* median lives in §5's Tail Risk & Return Distribution subsection, which *is* rebalancing-subject.)
5. **Input Portfolio Analysis** — the user's allocation held at the **sidebar-selected rebalancing cadence** (default **Never** = buy-and-hold): opens with an allocation pie ("Your Allocation", non-zero weights only) + normalised-weights table, then a cumulative-return chart (portfolio + per-asset overlays), annotated underwater curve, deepest-drawdown recovery + longest-underwater-stretch durations (calendar days, "≥ N days, ongoing" if unrecovered), headline growth/drawdown metrics (3+3 grid: geometric **CAGR**, arithmetic avg annual return, annual vol; Sharpe, Sortino, max DD), portfolio rolling-returns chart, the asset correlation heatmap, and finally a **Tail Risk & Return Distribution** subsection (the former §8, merged in — see below). See **Rebalancing basis** below.
6. **Monte Carlo Efficient Frontier** — random portfolio simulation, Sortino ratios
7. **SciPy Efficient Frontier** — optimization via `scipy.optimize.minimize`, efficient frontier line

**§5 Tail Risk & Return Distribution subsection** (was a standalone §8 until merged): per-period parametric (drift-free `σ·z`) and historical VaR/CVaR shown side by side, the mean/**median**/vol/skew/excess-kurtosis profile, and the return-distribution histogram vs a fitted normal. Rendered by `render_tail_risk(returns, alpha)`, reusing the per-period series §5 already built (computed once). Historical figures guarded by a data-sufficiency check (`n_tail = round((1-alpha)*n)`, `hist_ok` at `n_tail≥5`). There is **no separate VaR section and no VaR frontier** (the latter dropped earlier: drift-free VaR ∝ volatility, so it was redundant with §7).

### Rebalancing basis (selectable; §5 follows it, §6/§7 do not)

A sidebar **Rebalancing frequency** selector (`Never` default · `Every 6 months` · `Yearly` · `Every period`) governs **§5 only** (including its Tail Risk & Return Distribution subsection). §5 builds a single value series via `rebalanced_value_series(merged_df, tickers, weights, rebalance_every_periods)`: held at target weights, reset every **K** rows (count-based, `K = round(N/f)`; `Never` → `K=None`), drifting between resets — `V_t = V_r·Σ wᵢ·Pᵢₜ/Pᵢᵣ`, `V₀=1`, continuous across resets. `Never` reproduces the old buy-and-hold exactly; `Every period` reproduces the old constant-weight `portfolio_returns_simple.dot(weights)` exactly (so both historical bases remain reachable). The tail-risk subsection derives its per-period return distribution from the **same** series (`.pct_change().dropna()`, the `bh_ret` §5 already computed — `render_tail_risk` takes it as an argument rather than rebuilding it).

**§6/§7 stay per-period regardless of the selector** — the closed-form MPT frontier (annualised `mean × N`, `√(wᵀΣw) × √N`) requires `r_p = Σ wᵢrᵢ` each period, which *is* per-period rebalancing; less-frequent rebalancing has no closed-form mean/cov, so the frontier can't be re-derived for it. §5/§6/§7 each print a one-line `st.caption` declaring its basis, so the **same portfolio's return/risk/drawdown legitimately differ between §5 and §6/§7** unless the cadence is `Every period`. `rebalance_label` (the human string) is threaded from the entry point into the §5 banner. Annualisation still follows the textbook-linear convention (`mean × N`, `std × √N`). `buy_and_hold_value_series` remains as a thin wrapper (`K=None`).

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
- **Sign convention (loss = positive)**: VaR and `cvar()` (CVaR) report losses as **positive** numbers. §5's Tail Risk & Return Distribution subsection (`render_tail_risk`) shows per-period parametric (drift-free `σ·z`) and historical VaR/CVaR side by side; the historical pair is guarded by a data-sufficiency check (`n_tail = round((1-alpha)*n)`, shown only when `n_tail≥5`). CVaR for the portfolio cards is computed once, historically, at display time in `display_portfolio_cards(portfolios, alpha)`, so the whole app uses a single CVaR definition and sign.
- **Monte Carlo weights** use `np.random.dirichlet(np.ones(n))` for *uniform* simplex coverage — do **not** revert to normalising `np.random.random(n)` (biases toward equal weights). `np.random.seed(777)` (in the entry point) makes runs deterministic.

### Keeping descriptions in sync

When you change *how* a metric is computed, update its matching entry in `descriptions.py` — the per-section "How to read this section" text states the formula/assumptions and will otherwise silently drift from the code.

**Per-period terminology:** user-facing labels/tooltips/descriptions say **per-period** (the data interval may be daily/weekly/monthly). Reserve "daily" for genuine daily-interval or calendar references only — the annualisation legend (`252/52/12`), the mixed-calendar weekend-folding caveat, and `.days` durations.

### Rendering gotchas (silent failures — verify in a browser, not just the source)

Requires **Streamlit ≥ 1.50** (uses `width="stretch"`; developed against 1.58).

- **`st.data_editor` with `key=`: never reassign its returned frame back into the session_state variable you pass as its input.** The "My Portfolio" editor seeds `st.session_state.portfolio_df` *once*; the widget then persists edits as a diff against that stable seed. Writing the returned (edited) frame back moves the baseline underneath the widget, dropping the pending value of a freshly-added row until it's typed twice. Read current state from the editor's *return value* (`edited_portfolio`), not the session_state key.
- **`st.data_editor` newly-added rows yield `None`/`NaN` cells.** A row where the user typed a ticker but not yet a Market Value comes through with value `None`; coerce (`0.0 if v is None or pd.isna(v) else float(v)`) before `sum()`/division, or it raises `TypeError: float + NoneType`.
- **Inline SVG (the data-availability gauge in `render_load_etf_data`) must use `st.markdown(html, unsafe_allow_html=True)`.** Do **not** use `st.html` — it runs input through DOMPurify (default config), which strips `<svg>` entirely, so the gauge renders as *nothing* with no error. `st.components.v1.html` (the old approach) is removed after 2026-06-01. `st.iframe` only embeds a URL, not an HTML/SVG string. After any change here, confirm the SVG is actually in the DOM (e.g. Playwright `svg:has(linearGradient#gaugeGrad)`); the deprecation-warning-gone check alone is not enough.
- **KaTeX `$$…$$` blocks in `descriptions.py` must sit on a single source line.** A hard line break inside the block makes Streamlit's markdown treat it as a paragraph break, splitting the delimiters so the formula fails to parse (renders as raw red error text). The test's gauge check locates the inline SVG (not an iframe) for this reason.

### Key Functions

**data_handling.py:**
- `compute_portfolio_returns_simple(merged_df)` — the single simple-returns series (`.pct_change().dropna()`) used by §4 stats and all optimization; also returns its mean and covariance
- `compute_rolling_returns(merged_df, window_periods)` — rolling simple returns over a moving window
- `build_merged_dataframe(tickers, folder_path, filename_suffix, filter_date)` — inner join of asset prices
- `detect_stock_splits(tickers, folder_path, filename_suffix, filter_date)` — ground-truth splits from the yfinance `stock splits` column (date + ratio); skips files without the column. `check_price_anomalies(tickers, folder_path, filename_suffix, filter_date, z_threshold=8.0, min_abs_move=0.45)` — flags moves that are robust-MAD-z-score outliers *and* exceed a 45% floor (so genuine crypto swings and interval scaling don't false-trigger), cross-referenced against split dates. **Note: Adj Close is already split-adjusted, so a real split is *not* a price jump — splits come from the column, the anomaly check catches glitches / unadjusted splits.**
- **Total-return reconstruction:** `synthesize_total_return(price_index, etf, eurusd, periods_per_year)` — calibrate `q_hat` from the ETF overlap + return the spliced EUR series; `build_reconstructed_frame(index_prices, etf_prices, fx_prices, periods_per_year)` — shape it into the `date,adj close,synthetic,recon_yield` CSV frame; `run_total_return_reconstruction(jobs, intervals, output_dir, log_queue)` — threaded download+splice+save (mirrors `run_download`); `read_synthetic_info(...)` — per-ticker tag metadata for the §1 badge
- **Currency:** `run_download(..., convert_to_eur=True)` — downloads, auto FX-converts non-EUR to EUR (unless overridden), writes a `currency` column; `detect_currency(symbol, yf)` / `fetch_eur_multiplier(currency, yf_interval, end_date, yf)` / `apply_eur_conversion(data, eur_multiplier)` — the conversion helpers (last one is pure, unit-tested); `read_currency_info(tickers, folder_path, filename_suffix)` — resolves each ticker's currency (stored column, else cached network sniff) for the §1 check

**portfolio_calculations.py:**
- `portfolio_annualised_performance(weights, mean_returns, cov_matrix, ...)` — portfolio return/volatility
- `cvar(returns, alpha=0.05)` — historical CVaR; `alpha` is tail quantile
- `max_drawdown(returns)` — maximum drawdown
- `portfolio_cvar(weights, returns, alpha)` / `portfolio_max_drawdown(weights, returns)` — portfolio-level CVaR and Max DD
- `random_portfolios(num, mean_returns, cov_matrix, ...)` — Monte Carlo simulation
- `random_portfolios_sortino(num, ...)` — returns 5 rows (std, ret, sharpe, sortino, downside_dev); used in sections 6 & 7
- **§5 rebalancing helpers:** `rebalanced_value_series(merged_df, tickers, weights, rebalance_every_periods=None)` — value series held at target weights, reset every K rows (`None`=buy-and-hold, `1`=per-period rebalanced), `V₀=1`; `buy_and_hold_value_series(merged_df, tickers, weights)` — thin wrapper delegating with `K=None`; `underwater_episodes(value)` — list of peak/trough/recovery episodes; `deepest_drawdown_episode(value)` / `longest_underwater_episode(value)` (→ `(episode, days, ongoing)`); `downside_deviation_series(returns, N, rf)` — Sortino downside dev on a precomputed return series

**ui_components.py:**
- `render_rolling_returns(rolling_returns, tickers, rolling_window_years)` — renders 3b (individual assets only; no portfolio chart, no portfolio arg)
- `render_input_portfolio_analysis(merged_df, portfolio_returns_simple, tickers, my_portfolio_allocation, annualisation_factor, risk_free_rate, alpha, window_periods, rolling_window_years, rebalance_every_periods, rebalance_label)` — renders §5 at the selected rebalancing cadence (see **Rebalancing basis**); `_fmt_period(days, ongoing)` helper formats underwater durations. Ends by calling `render_tail_risk` for the merged tail-risk subsection
- `render_tail_risk(my_portfolio_returns, alpha)` — §5's Tail Risk & Return Distribution subsection (formerly the standalone §8): parametric + historical VaR/CVaR, mean/median/vol/skew/kurtosis, distribution histogram. Header-less; takes §5's already-computed per-period series (no recompute)
- `collect_portfolio_info()` — returns dict with max_dd and port_returns (CVaR is computed later from port_returns)
- `display_portfolio_cards(portfolios, alpha)` — shows 6 metrics incl. Max Drawdown and CVaR at the user's confidence level (label updates with α)
