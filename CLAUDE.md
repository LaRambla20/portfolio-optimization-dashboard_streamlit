# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

> Commands below are **Linux/macOS**. On Windows see **[WINDOWS.md](WINDOWS.md)** — `.venv\Scripts\…` paths, PowerShell boot-checks and the corporate-SSL workarounds. Everything outside this *Commands*/*Testing*/*Git* syntax applies to both.

- **No build system or linter.** A Playwright end-to-end test exists — see **Testing**.

**Setup (first time):**
```bash
python3 -m venv .venv
.venv/bin/pip install streamlit pandas numpy scipy matplotlib seaborn yfinance playwright
.venv/bin/playwright install chromium      # ~115 MB, only on a fresh clone
```
> **Debian/Ubuntu ship `ensurepip` separately**, so a bare `python3 -m venv .venv` dies with
> `ModuleNotFoundError: No module named 'ensurepip'`. Either `sudo apt install python3-venv`, or
> build it pip-less and bootstrap: `python3 -m venv --without-pip .venv` then
> `curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python -`.

**Run app:**
```bash
.venv/bin/streamlit run efficient_frontier_app/efficient_frontier_app.py
```

**Workflow:** Pre-loaded CSVs for EM57.MI, VWCE.MI, SGLD.MI, IMIE.MI, DBMF, BTC-EUR, ZPRV.DE (daily + monthly) are included in `individual_indices_data/` — click **Run Analysis** immediately. To add or refresh tickers, use the sidebar **📥 Get Data** panel.

## Testing

**Requires the app to be running first** (`Run app` command above), then:

```bash
HEADLESS=1 .venv/bin/python tests/test_dashboard.py
```

`HEADLESS=1` is effectively mandatory on Linux — `headless=False` needs an X/Wayland display and hangs without one.

> **Gotcha:** Streamlit caches imported modules in memory, so a still-running server serves **stale code** after you edit `ui_components.py`/etc. After edits, kill it and restart before re-testing — and verify the restart actually bound (see *Boot-check*), because a stale process holding the port makes a failed restart look successful.

Playwright drives a real Chromium browser, clicks Run Analysis, and verifies all 8 section headers render plus portfolio card metrics. Screenshots are saved to `test_screenshots/` (gitignored).

> **Gotcha:** the test's "done" signal is `st.success(" Analysis complete!")`, rendered **last** in `efficient_frontier_app.py` after the final section. Remove or relocate it (e.g. when merging/reordering sections) and `test_dashboard.py` hangs 180s then fails on `wait_for_selector("text=Analysis complete")` — keep it as the final render. The test's `SECTIONS` list must also match the rendered `st.header` strings exactly (incl. the section count).

**Unit tests (no running app, no network needed) — all live in `tests/`.** Run one with
`.venv/bin/python tests/test_X.py`, or all of them:

```bash
for t in tests/test_*.py; do [ "$t" = tests/test_dashboard.py ] || .venv/bin/python "$t"; done
```

| Test | Covers |
|---|---|
| `test_total_return_synthesis.py` | Total-return reconstruction math: yield recovery, splice continuity, FX caveat, CSV save/read round-trip, `convert_series_to_eur` alignment, and the `currency` column being appended last and the price reader ignoring it. Plus one live `^GSPC` vs `^SP500TR` check that SKIPs gracefully offline. |
| `test_rebalancing.py` | `rebalanced_value_series` — Never == buy-and-hold, Every-period == constant-weight compounded, periodic-reset continuity. |
| `test_aftertax.py` | `rebalanced_value_aftertax` (the §6 capital-gains-tax overlay): `tax_rate=0` reproduces `rebalanced_value_series` exactly and `netliq==with_tax`; `netliq ≤ with_tax ≤ no-tax` monotonicity; single-asset pays only final-liquidation tax; within-period loss-netting beats taxing gross gains. |
| `test_real_terms.py` | `real_deflator`/`to_real` — 0%/yr is an exact no-op, constant-rate deflation shifts the mean by ~π but leaves volatility ~unchanged, and deepens drawdowns. |
| `test_lookback_windows.py` | §2 per-asset look-back machinery — `evaluate_simple_return`/`evaluate_CAGR` empty/zero-span guards, `load_asset_series` (own-history + `usecols` ignores `_EXT`/currency extras), `_lookback_year_windows` (drops windows longer than history, keeps fitting ones + a true `Full` row), `_full_history_label` (formatting + 12-month rollover). |
| `test_geometric_return.py` | `_geometric_annual_return` — zero-vol → geo above arithmetic via intra-period compounding; high-vol → geo below by ≈σ²/2 drag; NaN guards. |
| `test_extend_wizard.py` | The guided wizard's pure helpers — `suggest_fx_ticker` incl. the EUR/USD divisor direction and same-currency `None`, `index_query_from_name`'s issuer/wrapper stripping, `default_q_regime`'s classification, and `q_hat_verdict` across **both** bands (the 11 real measured pairings are asserted directly, plus that a regime-less call still behaves as the original 0–6% gate). |
| `test_optimizers.py` | The §7/§8 SLSQP frontier solvers (all six delegate to the shared `_optimize` helper) — weights stay on the long-only simplex, min-vol is the volatility floor, `efficient_return`/`efficient_volatility` hit their targets exactly, and max-Sortino beats equal-weight. |

> **Adding a unit test:** put it in `tests/`. Plain assert-based script — anchor the import path to the file (`sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))`, so it runs from any CWD), functions that `assert` and `print("... OK")`, driven by an `if __name__ == "__main__"` runner. No pytest. Run with `.venv/bin/python tests/test_X.py`.

> **yfinance uses `curl_cffi`, not `requests`** — so pip/git SSL workarounds never apply to it. A bare `yf.Ticker` works here; on an SSL-intercepting machine see [WINDOWS.md](WINDOWS.md).

**Boot-check (fast path):** for sidebar-only / non-render changes, launch headless and read the log instead of running Playwright:
```bash
nohup .venv/bin/streamlit run efficient_frontier_app/efficient_frontier_app.py \
  --server.headless true > /tmp/boot.log 2>&1 &
sleep 8; grep -q "Uvicorn server started" /tmp/boot.log && echo BOUND || cat /tmp/boot.log
```
> **A `LISTEN` row on 8501 is not proof of a good boot.** If an older process still holds the port, the new one logs `Port 8501 is not available` — which a `grep -Ei 'error|traceback'` does **not** match — and the stale process keeps serving old code. `Uvicorn server started` in the **new** process's log is the only reliable signal.

> **Stopping it kills your own shell unless you bracket the pattern.** `pkill -f "streamlit run …"`
> matches the shell running the pkill and kills it (exit 144 — and any heredoc in that same command is
> silently never written). Two rules, both needed:
> 1. Bracket the first letter so the pattern can't match itself: `pkill -f "[s]treamlit run effici"`.
> 2. **Never put the kill and the relaunch in the same command.** The bracket trick doesn't save you
>    there — the `nohup .venv/bin/streamlit run efficient_frontier_app…` line later in the *same*
>    command line is an unbracketed match, so pkill still finds and kills the shell. Kill in one call,
>    launch in the next. (`| head -1` doesn't help either: pgrep can list the shell first.)

> **Driving `build_merged_dataframe` offline:** it filters `date <= filter_date_string`, so passing `None` raises `TypeError: Invalid comparison between datetime64 and NoneType`. Pass a real date string (e.g. a far-future `"2100-01-01"` to keep all rows) when calling it outside the app for a smoke test.

## Git

Remote: https://github.com/LaRambla20/portfolio-optimization-dashboard_streamlit.git
Default branch: `master`

Plain `git push` works here. On Windows the corporate cert store is needed — see [WINDOWS.md](WINDOWS.md).

## Architecture

Modular Streamlit dashboard split into 5 files inside `efficient_frontier_app/`.
- **`efficient_frontier_app/efficient_frontier_app.py`** — main entry point: sidebar inputs, derived parameters, orchestrates data loading and UI rendering.
- **`efficient_frontier_app/portfolio_calculations.py`** — pure math: portfolio performance, optimization (SciPy), Monte Carlo simulation, VaR calculations.
- **`efficient_frontier_app/data_handling.py`** — data operations: CSV loading, validation, stock-split detection + price-anomaly checks, merged dataframe building, return computations. Uses `@st.cache_data`.
- **`efficient_frontier_app/ui_components.py`** — UI renderers: 8 section functions (`render_load_etf_data`, `render_per_etf_analytics`, `render_etf_prices`, `render_rolling_returns`, `render_returns_statistics`, `render_input_portfolio_analysis`, `render_monte_carlo`, `render_scipy_ef`) plus shared helpers (`render_tail_risk` — §6's tail-risk subsection, `collect_portfolio_info`, `display_portfolio_cards`). All data passed as explicit parameters (C-like style).
- **`efficient_frontier_app/descriptions.py`** — single source of truth for the per-section "How to read this section" expanders (concept → markdown with KaTeX formulas); rendered via `render_section_help`.
(No legacy files currently on disk — the repo contains only the 5-file modular app.)

### Simple returns everywhere (no return-type toggle)

**The app uses simple (arithmetic) returns throughout** — there is no log/simple toggle. This was a deliberate simplification for interpretability: every figure means "actual realised percentage change," and the dashboard never asks the user to reason about which return type a section is in. A single series, `portfolio_returns_simple` (= `merged_df` prices `.pct_change().dropna()`), backs §5's distribution stats/plot **and** all portfolio optimization (Monte Carlo, SciPy, VaR); §4 rolling returns are simple cumulative window returns.

Using simple returns for portfolio math is also the *correct* choice, not just the simple one: `sum(weights * log_returns)` does not equal the portfolio log return (log returns aren't additive across assets), so optimization must use simple returns regardless. (Log returns have a textbook edge only for single-asset distribution-shape analysis, and at daily/monthly frequency they barely differ from simple — not worth a toggle here.)

Key variables for optimization:
- `portfolio_returns_simple` — simple returns dataframe
- `portfolio_mean_returns` — mean of simple returns
- `portfolio_cov_matrix` — covariance of simple returns

**Annualisation (textbook MPT, linear):** annual return = `mean × N`, annual volatility = `std × √N` (N = periods/year: 252/52/12). This is an *arithmetic/expected* return, distinct from the geometric CAGR shown in §2.

### Real (inflation-adjusted) terms toggle (nominal by default; §2 + §6 only)

A sidebar **"Show real (inflation-adjusted) returns"** checkbox (default off) deflates the **performance** sections by a constant assumed inflation rate. The **"Assumed annual inflation (%)"** number_input (default 2.0) is rendered **only when the checkbox is on** — wrapped in `if real_terms:` in the entry point so the dependency is visually obvious; `annual_inflation` is set to `0.0` on the else branch (a hidden Streamlit widget loses its value, so toggling off→on resets it to 2.0 — accepted, no `key=`). Both `real_terms` (bool) and `annual_inflation` (decimal) are threaded into `render_per_etf_analytics` and `render_input_portfolio_analysis`; `real_terms` alone is threaded into `render_monte_carlo`/`render_scipy_ef` (for a nominal-basis caption only).

**Implementation = deflate the level series once, not rewrite formulas.** `real_deflator(date_index, annual_inflation)` returns `D_t = (1+π)^(years since start)` (calendar-time based, robust to irregular spacing); `to_real(level, π)` = `level / real_deflator(...)`. §2 deflates each asset's price series right after loading it via `load_asset_series`; §6 deflates the `rebalanced_value_series` output once — so every downstream figure (returns, CAGR, drawdown, underwater episodes, rolling returns, **and** the `render_tail_risk` distribution, which takes §6's already-deflated `bh_ret`) becomes real with no per-figure edits. §6 also deflates the per-asset cumulative overlays and the risk-free rate (`rf_used = (1+rf)/(1+π)−1`) so Sharpe/Sortino stay consistent.

**Scope rationale (why §5/§7/§8 stay nominal).** A *constant* rate shifts means but not variances: volatility, covariance/correlation, skew/kurtosis are **unchanged** (§6's parametric VaR/CVaR are drift-*inclusive*, so under §6's deflation they shift only by the tiny per-period drift), and because the real risk premium `r_real − rf_real ≈ r_nom − rf_nom` is inflation-invariant, **Sharpe, Sortino and the efficient-frontier weights don't move** — so deflating §7/§8 would only relabel the return axis. §7/§8 print a one-line nominal-basis caption when the toggle is on; §6's correlation heatmap (built from nominal `portfolio_returns_simple`) carries a note that it's invariant. The `real_returns` entry in `descriptions.py` documents the Fisher math + invariances (appended to §2/§6 help lists when on). When the toggle is off, `annual_inflation=0` makes `real_deflator` all-ones, so the nominal path is a guaranteed no-op.

### Capital-gains tax / net-liquidation overlay (§6 only; off by default)

A sidebar **"Capital-gains tax on realized gains (%)"** number_input (default **0.0**, always shown — 0% is a meaningful "off" value, no conditional widget) threads `cgt_rate` (decimal) into `render_input_portfolio_analysis`. When `cgt_rate > 0`, §6 builds an after-tax series via `rebalanced_value_aftertax(merged_df, tickers, weights, rebalance_every_periods, cgt_rate)` → `(with_tax, netliq)` and overlays the **net-liq** line (red dashed) on the cumulative-return chart with a tax-drag caption (net-liq CAGR vs pre-tax CAGR in pp/yr). **Off by default and `cgt_rate=0` is an *exact* no-op** — `rebalanced_value_aftertax` at `tax_rate=0` reproduces `rebalanced_value_series` exactly and `netliq == with_tax` (unit-asserted in `tests/test_aftertax.py`), so the default render is byte-identical to before.

**Tax model:** at each rebalance, overweight legs are trimmed to target, realizing gains/losses; **realized losses net against realized gains within that same rebalance** (`taxable = max(0, Σ realized)`) and the positive remainder is taxed. **No loss carry-forward** (the Italian *zainetto* is intentionally not modelled — the source export script's bug was taxing *gross* gains; this fixes it by netting). The net-liq line additionally subtracts tax owed on *unrealized* gains: `netliq_t = V_t − τ·max(0, V_t − book_t)`. **Composes with both** the rebalancing cadence (tax realized only at resets; `Never`/`K=None` → no interim tax, only final liquidation tax) **and** real terms (tax computed on nominal levels, then the net-liq level is deflated once by `real_deflator`, same "deflate the level once" rule). The `capital_gains_tax` entry in `descriptions.py` documents the formula (appended to the §6 help list when on). Threaded only into §6 — §7/§8 are pre-tax (MPT frontier weights are tax-agnostic here).

### Section Structure

1. **Load Data** — validates CSVs, reports recorded stock splits (📐, from the `stock splits` column — informational, since Adj Close is already split-adjusted) and flags statistically anomalous price moves (⚠️, robust MAD z-score), shows data availability gauge, a 💱 warning if any series isn't EUR, a 📅 mixed-calendar caveat (`mixed_calendar`/`seven_day_tickers` from `compute_data_availability` — basket mixes 7-day crypto + ~5-day equity), and (for `{ETF}_EXT` files) a 🧬 caption flagging reconstructed total-return rows. The 📅 caveat is **also propagated to where the covariance is consumed** — §5 (cov/corr tables), §6 (heatmap), §7/§8 (frontier+VaR) — via `_mixed_calendar_note` (Issue 5, disclosure only; numbers unchanged, weekly/monthly remains the remedy). The annualisation descriptions (`avg_annual_return`/`annual_volatility`) note the `×N`/`√N` i.i.d. (no-serial-correlation) assumption.
2. **Per-Asset Analytics** — CAGR, simple/calendar-year returns, "Annualised Metrics by Look-back Period", **Max Drawdown** (always simple returns for consistency; optionally deflated to **real terms** — see the toggle above). Each asset is loaded over its **own full history** via `load_asset_series` (not the inner-joined common window used by the portfolio sections), so a long-history asset isn't truncated to the shortest holding's start. Look-back windows (simple-return, CAGR, both annualised-metrics tables) only show the 1/3/5-year horizons that **fully fit** within the asset's history, plus a true `Full (Ny Mm)` row (`_lookback_year_windows`); windows longer than the asset's life are omitted rather than silently truncated and mislabelled (Issue 3). Calendar-year rows skip years the asset wasn't trading, and the "as of last full year-end" view is omitted when there's no prior full year.
3. **Per-Asset Prices** — raw and normalized (base = 1000) price charts (merged df is built upstream via inner join)
4. **Per-Asset Rolling Returns** — moving-window **cumulative** returns (1/2/3/5/7/10 years; `P_t/P_{t−w}−1`, the whole-window total, *not* annualised — labelled "Cumulative N-Year Return" with a not-comparable-across-window-lengths caption, Issue 4) for **individual assets only** (the portfolio rolling-returns chart now lives in §6)
5. **Per-Asset Returns & Statistics** — per-asset min/max/mean/**median**/std summary (per-period simple returns; the mean−median gap reads skew), per-asset Sortino, covariance/correlation matrix **tables** (from simple returns), per-period returns plot. **Not subject to the rebalancing selector** — purely per-asset, no portfolio aggregate. The correlation **heatmap** moved to §6. (Per-period **median** is not annualised — quantiles aren't additive, so no `×N`; the *portfolio* median lives in §6's Tail Risk & Return Distribution subsection, which *is* rebalancing-subject.)
6. **Input Portfolio Analysis** — the user's allocation held at the **sidebar-selected rebalancing cadence** (default **Never** = buy-and-hold): opens with an allocation pie ("Your Allocation", non-zero weights only) + normalised-weights table, then a cumulative-return chart (portfolio + per-asset overlays), annotated underwater curve, deepest-drawdown recovery + longest-underwater-stretch durations (calendar days, "≥ N days, ongoing" if unrecovered), headline growth/drawdown metrics (3+3 grid: geometric **CAGR**, arithmetic avg annual return, annual vol; Sharpe, Sortino, max DD), portfolio rolling-returns chart (**cumulative** N-year return, relabelled per Issue 4), the asset correlation heatmap, and finally a **Tail Risk & Return Distribution** subsection (the former standalone VaR section, merged in — see below). Optionally deflated to **real terms** (whole section incl. tail risk; see the toggle above), and optionally overlaid with an **after-tax net-liquidation** line (see the capital-gains-tax toggle below). See **Rebalancing basis** below.
7. **Monte Carlo Efficient Frontier Portfolio Optimization** — random portfolio simulation, Sortino ratios
8. **Scipy Efficient Frontier Portfolio Optimization** — optimization via `scipy.optimize.minimize`, efficient frontier line

**§6 Tail Risk & Return Distribution subsection** (was a standalone VaR section until merged): per-period parametric (drift-inclusive, VaR `= σ·|z| − μ`, normal-ES CVaR `= σ·φ(z)/τ − μ`) and historical VaR/CVaR shown side by side (same drift basis, so the comparison isolates tail shape), the mean/**median**/vol/skew/excess-kurtosis profile, and the return-distribution histogram vs a fitted normal. Rendered by `render_tail_risk(returns, alpha)`, reusing the per-period series §6 already built (computed once). Historical figures guarded by a data-sufficiency check (`n_tail = round((1-alpha)*n)`, `hist_ok` at `n_tail≥5`). There is **no separate VaR section and no VaR frontier** (the latter dropped earlier: drift-free VaR ∝ volatility, so it was redundant with §8).

### Rebalancing basis (selectable; §6 follows it, §7/§8 do not)

A sidebar **Rebalancing frequency** selector (`Never` default · `Every 6 months` · `Yearly` · `Every period`) governs **§6 only** (including its Tail Risk & Return Distribution subsection). §6 builds a single value series via `rebalanced_value_series(merged_df, tickers, weights, rebalance_every_periods)`: held at target weights, reset every **K** rows (count-based, `K = round(N/f)`; `Never` → `K=None`), drifting between resets — `V_t = V_r·Σ wᵢ·Pᵢₜ/Pᵢᵣ`, `V₀=1`, continuous across resets. `Never` reproduces the old buy-and-hold exactly; `Every period` reproduces the old constant-weight `portfolio_returns_simple.dot(weights)` exactly (so both historical bases remain reachable). The tail-risk subsection derives its per-period return distribution from the **same** series (`.pct_change().dropna()`, the `bh_ret` §6 already computed — `render_tail_risk` takes it as an argument rather than rebuilding it).

**§7/§8 stay per-period regardless of the selector** — the closed-form MPT frontier (annualised `mean × N`, `√(wᵀΣw) × √N`) requires `r_p = Σ wᵢrᵢ` each period, which *is* per-period rebalancing; less-frequent rebalancing has no closed-form mean/cov, so the frontier can't be re-derived for it. §6/§7/§8 each print a one-line `st.caption` declaring its basis, so the **same portfolio's return/risk/drawdown legitimately differ between §6 and §7/§8** unless the cadence is `Every period`. `rebalance_label` (the human string) is threaded from the entry point into the §6 banner. Annualisation still follows the textbook-linear convention (`mean × N`, `std × √N`). Buy-and-hold is just `rebalanced_value_series(..., rebalance_every_periods=None)` — the §6 default.

### Get Data panel (one sidebar panel for both download jobs)

**📥 Get Data** replaced the old *Download settings* panel and the 🧬 reconstruction panel nested
inside it. Both jobs download from Yahoo on their own — nothing has to be downloaded first — and a
`st.radio` picks between them, rendering only that job's inputs:

- `Download ETF's price history as is` → a `text_area` of tickers → `run_download`
- `Download & extend ETF's price history` → **"🧬 Start guided setup"** → the guided wizard below
  (the old 3-column `data_editor` is gone; the wizard handles one fund at a time)

The two radio labels are module constants (`DL_MODE_PLAIN` / `DL_MODE_EXTEND`) so the branch checks
can't drift from the displayed strings.

**No autofill, by explicit user request** — retyping over pre-filled examples every time was the
complaint. Examples live where they cost nothing to dismiss: `placeholder=` (greyed, never submitted)
on every ticker input. **Do not "helpfully" re-add default values.** A *computed* suggestion is the
exception (the wizard's "Use this" / "Use EURUSD=X" buttons) — it fills the field on click, so there
is still nothing to erase.

Shared below the branch: one *Intervals to fetch* `multiselect` defaulting to `[data_period]`, and a
live `st.warning` when `data_period not in dl_intervals` — the default only binds on first render, so
the warning (not the default) is what actually prevents the download-daily/analyze-monthly missing-file
trap. `⚙️ Advanced` holds *Columns to drop*, *Output folder* and *Keep native currency*; a blank output
folder resolves to `folder_path` (`dl_output_dir.strip() or folder_path`), so the two can no longer
silently disagree. `📖 How to find tickers` renders `DESCRIPTIONS["finding_tickers"]`; the extend branch
additionally shows `DESCRIPTIONS["extending_history"]`. Both are plain `st.markdown` (not
`render_section_help`, which is for main-area sections) and are deliberately KaTeX-free.

`run_job_with_progress(title, target, args, kwargs, total_tasks, done_message)` runs either backend on a
thread and drains its log queue. Both backends share one contract: **the log queue is the last positional
arg**, `✅`/`❌`-prefixed lines each mark one finished task, and a final `__DONE__done/total` carries the
count. `run_download` previously emitted bare `Saved …`/`Error …`, so its progress bar sat at 0 until it
jumped to 100% — the prefixes are now required, don't drop them. `cell_text()` coerces `data_editor`
cells (a fresh row yields `None`, and `str(None)` is the truthy `"None"` — that's why it isn't a plain
`str()`).

### Guided extend wizard (§ Get Data → modal dialog)

The *Download & extend* mode opens a four-step `st.dialog` (`render_extend_wizard` in
`ui_components.py`) instead of a job table: **Fund → Index → Currency → Confirm**. Each step probes
Yahoo and only enables **Continue** once its gate passes, so a dead ticker is caught before any run.

| Step | Gate |
|---|---|
| 1 · Fund | `probe_ticker` returns `ok` (real downloadable history) |
| 2 · Index | `ok`, **>=2 overlapping dates** with the fund, and the index **starts before** the fund. Also *previews* q̂ per candidate |
| 3 · Currency | radio: *choose a pair* (pre-filled with the derived pair) vs *proceed without* — the latter auto-selected when both currencies match |
| 3 · Currency | `ok`; auto-skipped when both currencies match; **warns** when the FX series starts *after* the index |
| 4 · Confirm | `q_hat_verdict(q, regime)` — q̂ inside the band for the chosen regime — else **Reconstruct stays disabled** |

**Two bands, because "how big should the gap be" depends on the pairing** (`Q_BANDS`):

| Regime | Band | When |
|---|---|---|
| `price_index` | 0% … +6% | index is price-only, fund collects the dividends → gap *is* the yield |
| `same_income` | −0.5% … +0.5% | index already carries the income, or the asset has none (gold) → only fees separate them |

A single 0–6% band **rejected correct pairings**: `GC=F`→`SGLD.MI` recovers **−0.24%/yr** (gold pays
nothing; that's SGLD's fee) and was blocked. `default_q_regime(index_probe)` picks the band from the
*index* leg — pays dividends → `same_income`; FUTURE/CURRENCY → `same_income`; INDEX without
dividends → `price_index`; anything else (a fund used as the index, whose Adj Close is total-return)
→ `same_income`. Validated on 11 real pairings: all 6 good pass, all 5 bad block.

**The fund leg cannot decide this** — an accumulating equity fund and a gold ETC both report
`Dividends = 0`. Hence the step-4 override radio; never make the regime purely automatic.

**Accumulating vs distributing changes nothing here.** The app reads **Adj Close**, which adds
distributions back, so a distributing fund is already total-return (measured: `Adj Close/Close`
falls to ~0.78 for VT, stays exactly 1.0000 for accumulating funds and no-income assets). It is
*displayed* per leg because it explains the classification, not because it gates.

**`run_total_return_reconstruction`'s log warning uses the same `q_hat_verdict`**, with `regime`
carried on the job dict. It previously hard-coded 0–6% and so announced "outside the usual 0–4%
dividend band" immediately after the wizard had passed the very same gold pairing. Keep them on one
helper or they will disagree again.

**q̂ is the only gate; correlation was deliberately rejected.** Measured: the correct pairing
`^GSPC`→`SXR8.DE` correlates **0.74**, while the nonsensical `^GSPC`→`VWCE.MI` correlates **0.89** —
*higher*. All broad equity indices correlate ~0.85+, and cross-currency pairs are penalised by FX
noise, so correlation ranks a bad pair above a good one. q̂ separates them (that bad pair recovers a
**negative** yield). Don't reintroduce a correlation check; it looks meaningful and isn't.

**Step 2 previews q̂ for every candidate** via `preview_candidate_fit`, so you pick the best index
up front instead of discovering at step 4 that it fails. Nearly free — probing already cached each
candidate's series, so all six previews take ~0.35 s of pure pandas.

**The preview must use an auto-derived FX pair**, because the user hasn't chosen one yet and q̂
*without* conversion is badly wrong: a JPY-quoted gold ETF reads **−1.75%/yr** unconverted versus
**+0.14%** converted — the difference between "rejected" and "fine". Each row therefore names the
pair it used (`via EURUSD=X`), and step 3 pre-fills that same pair so the number the user saw is the
number that gets used. Rows are ranked plausible-first, and candidates that **start after the fund**
are demoted with a reason — that is what makes EM57.MI read as "nothing here extends this" at a
glance instead of four separate checks.

**Index suggestions are probed, never trusted.** `suggest_index_candidates` merges
`CURATED_INDEX_HINTS` with `yf.Search`, then runs **every** candidate — from both sources — through
`probe_ticker`, listing them OK/BAD with spans. It accepts `SEARCH_QUOTE_TYPES` =
INDEX/ETF/FUTURE/CURRENCY/MUTUALFUND — **`FUTURE` is what makes `GC=F` (gold, back to 2000) reachable
at all**, and `EQUITY` is excluded because it floods the list with individual stocks. The fund's own
symbol is filtered out (searching a fund's name returns the fund).

**Query order in `_search_queries` is load-bearing, not cosmetic**: `"Gold index"` returns S&P/TSX
*Global Gold* — gold **miners**, a different asset from bullion — while plain `"Gold"` returns
`GC=F`. So the full cleaned name is tried first and leading words are dropped only as a fallback,
stopping at the first query that yields candidates. The curated map exists because Search alone returns
only quote-only `*.FGI` tickers for the common European ETFs (searching "FTSE All-World" puts the
history-less `AW01.FGI` first). Because hints are probed like everything else, a stale entry shows as
BAD rather than misleading. Search finding nothing is fine — the step falls back to manual entry.

`suggest_fx_ticker(etf_ccy, index_ccy)` → `{etf}{index}=X`. **Direction is load-bearing**:
`synthesize_total_return` divides (`index / fx`) and Yahoo quotes `ABCDEF=X` as DEF per 1 ABC, so an
EUR fund with a USD index needs `EURUSD=X`. Reversing it silently inverts the conversion.

**Streamlit constraints (verified in 1.62 — violating these breaks the wizard):**
- `st.dialog` is a **fragment**: widgets inside rerun only the dialog. Use `st.rerun(scope="fragment")`
  to step, and `st.rerun(scope="app")` to close it.
- **`st.sidebar` cannot be called inside a dialog.** Config is snapshotted into
  `st.session_state["wiz_cfg"]` when the wizard opens.
- **Elements created outside the dialog are additive across its reruns.** The reconstruction therefore
  queues into `session_state["recon_job"]` and is run by the *main script* via `run_job_with_progress`
  — never draw the progress UI from inside the modal.
- **Dismissing with the ✕ does *not* clear your own open-flag.** The close is client-side only, so
  `wiz_open` stays True and the **next full script rerun** — editing the My Portfolio table, nudging
  any sidebar widget — re-renders the dialog and it pops back up unwanted. `on_dismiss=wiz_dismissed`
  is the only hook that fires on the ✕; without it the wizard is unclosable in practice. Any future
  `st.dialog` gated on a session_state flag needs the same callback.
- **A keyed widget ignores its `value=` argument on every rerun after the first.** The "Use this" /
  "Use EURUSD=X" buttons must write `st.session_state["wiz_idx_in"]` / `["wiz_fx_in"]` directly;
  passing `value=` does nothing (this was a real bug — the buttons silently failed to fill the field).
  `_wiz_seed` seeds each input once; `wiz_reset()` clears the state dict **and** the widget keys,
  which outlive it.
- **A new index leg invalidates the suggested rate and the detected regime.** `Check index` clears
  `wiz_fx_in` / `wiz_fx_mode` / `wiz_regime_in` — the suggested pair depends on the *index's*
  currency, so a stale one silently converts with the wrong rate.
- **`st.metric(delta=…)` draws a direction arrow.** A non-numeric label ("implausible") always renders
  an *up* arrow, which reads as good next to a negative yield — the q̂ metric passes no `delta`.

**Not every fund can be extended, and no search change fixes that.** EM57.MI (Amundi Euro Govt Bond
5-7Y) starts 2008-01 and *every* EUR government-bond series on Yahoo starts 2008-01 or later
(`EXHC.DE`, `EXHB.DE`, `IBGL.AS`, `IBGX.AS`, `IBCI.DE`, `EUNH.DE`, `X03G.DE` all checked). The step-2
"index must start before the fund" gate reports this correctly. Extending it needs a different data
provider, not a wider search.

### Currency handling (single base currency = EUR)

**The app does no FX conversion anywhere except where noted here — every `adj close` is assumed already in EUR** (price-chart axes are hard-labeled `[EUR]`, portfolio weights derive from EUR market values). Feeding in a USD/GBP-priced series silently treats those prices as EUR, so its returns/vol and especially **correlations** (every same-currency asset shares a hidden FX factor) come out in the wrong numeraire, biasing the covariance matrix → frontier and VaR. No error is raised. Verified live: US-listed `IVV` shows 14.57%/yr in USD but 13.99%/yr once converted to EUR (matching the EUR-listed `SXR8.DE` at 13.84%).

Two guards:
- **Both download jobs auto-convert to EUR by default** (`run_download(..., convert_to_eur=True)` and `run_total_return_reconstruction(..., convert_to_eur=True)`; the reconstruction converts **the ETF leg only, before splicing** — the older synthetic rows are already put into the ETF's currency by the job's FX ticker, so converting the spliced output would double-convert them): detects the ticker's currency (`detect_currency`, via `fast_info`/`info`), and if not EUR scales every price column by Yahoo's `{CCY}EUR=X` rate (`fetch_eur_multiplier` → `apply_eur_conversion`; handles GBp/pence by `/100`). The sidebar **"Keep native currency (don't convert to EUR)"** checkbox (default off) is the escape hatch that stores raw prices instead. Conversion needs two network calls (the sniff + the FX series); if either fails the native series is stored and §1 warns — so the check below is the safety net, not redundant. Prefer the EUR-listed share class (`.MI`/`.DE`/`.AS`) when one exists — it's the same thing.
- **§1 currency check** (`read_currency_info` → `render_load_etf_data`): warns if any loaded series isn't EUR. Every download writes a `currency` column (the stored currency: `EUR` when native or converted, else e.g. `USD`), so the check is offline; legacy CSVs lacking the column fall back to a cached network sniff.

The on-disk `currency` column is a constant per file and, like the reconstruction tag columns, is ignored by the `usecols` readers.

> **A mid-month download leaves a partial last bar.** A monthly fetch on the 21st writes that month's
> *running* value — often equal to the prior month's close, which reads as a stale duplicate row. It
> settles at month end; don't treat it as a closed month or "correct" it.

> **Yahoo revises history between downloads.** Two fetches minutes apart are identical; hours apart can
> differ by ~1e-7 relative. Before suspecting a code change, run **both code paths against one fetch** —
> that isolates upstream drift from a real behaviour change.

### Total-return reconstruction (extending ETFs with index history)

Accumulating ETFs (VWCE, EM57, …) only span a few years; the **price-return** indices they track (most Yahoo `^` tickers, all Stooq indices) run for decades but **exclude reinvested dividends**, so splicing one straight in front of an ETF biases every long-run figure low. The **📥 Get Data** panel's *Download & extend ETF's price history* mode fixes this:

1. For each job — *(index ticker, calibrating ETF, optional FX ticker)* — it downloads all three at the selected interval(s).
2. `synthesize_total_return` converts the index to the ETF's currency (`index / eurusd` — **FX first**, or `q_hat` absorbs FX drift), then calibrates the missing yield geometrically from the overlap: annual gross-up `f = (1+g_etf)/(1+g_index)`, reported `q_hat = f − 1` (this also absorbs TER/tracking — intentional, so the synthetic tail meets the real ETF seamlessly).
3. Older index history is grossed up **on returns** (`f**(1/N)` per period) and **chained** to the real ETF (scaled to meet it at `join_date` — never raw-level concatenation), then saved as `{ETF}_EXT_data_{period}.csv` with the `synthetic` / `recon_yield` tag columns.

Add the new `{ETF}_EXT` ticker to **My Portfolio** to analyze it. §1 shows a 🧬 caption flagging which rows are reconstructed (estimate, not measured).

**The index must track the same underlying as the ETF** (e.g. `^GSPC` with an S&P 500 ETF). The calibration makes the index match the ETF over the overlap, so pairing mismatched underlyings (S&P 500 vs FTSE All-World) folds their *performance gap* into `q_hat` — `^GSPC`→`VWCE.MI` yields a nonsensical **−1.2%/yr** because US stocks outran the world 2020–2025. The runner emits a `⚠️` log line when `q_hat` lands outside ~0–6%/yr (also catches data quirks: splits, wrong FX, distributing-vs-accumulating share classes — e.g. `SXR8.DE` returned an implausibly low +0.1%). Tests: `tests/test_total_return_synthesis.py` (deterministic math/splice/FX + save/read round-trip, plus a live `^GSPC` vs `^SP500TR` check that recovers q≈1.9%). Note: the live test fetches via a `curl_cffi` session with `verify=False` to bypass this machine's SSL interception — `run_total_return_reconstruction` itself uses a bare `yf.Ticker` like `run_download`, matching existing behavior.

### Data Format

CSV files in `individual_indices_data/` named `{ticker}_data_{period}.csv` (period: daily/weekly/monthly) with columns: `date`, `adj close`.

**Reconstructed (total-return) files** — `{ETF}_EXT_data_{period}.csv` — carry two extra columns: `synthetic` (bool: `True` for rows reconstructed from the index *before* the ETF's first date, `False` for real ETF rows) and `recon_yield` (the calibrated annual gross-up `q_hat`, filled only on synthetic rows), plus a `currency` column appended **last**. Files downloaded by `run_download` also carry a `currency` column (see **Currency handling**). Every price reader passes `usecols=["date", "adj close"]`, so column order on disk doesn't matter and the extras are ignored — only `read_synthetic_info` (§1 badge) and `read_currency_info` (§1 currency check) read them. See **Total-return reconstruction** and **Currency handling** below.

> **`_EXT` files are gitignored** (`individual_indices_data/*_EXT_data_*.csv`), as are `^`-prefixed
> index downloads. Only the plain ETF sample CSVs are tracked — a fresh clone has no reconstructed
> series, so re-run the wizard to get one. Local `_EXT` files in your tree never show up in a diff.

### VaR / CVaR Gotchas

- **`cvar(returns, alpha)` expects the tail quantile** (e.g., `alpha=0.05` for 95% CVaR), NOT the confidence level (`0.95`). The sidebar `alpha` is the confidence level — pass `1 - alpha` to `cvar()`.
- **Sign convention (loss = positive)**: VaR and `cvar()` (CVaR) report losses as **positive** numbers. §6's Tail Risk & Return Distribution subsection (`render_tail_risk`) shows per-period parametric (drift-inclusive `σ·|z| − μ`) and historical VaR/CVaR side by side; the historical pair is guarded by a data-sufficiency check (`n_tail = round((1-alpha)*n)`, shown only when `n_tail≥5`). CVaR for the portfolio cards is computed once, historically, at display time in `display_portfolio_cards(portfolios, alpha)`, so the whole app uses a single CVaR definition and sign.
- **Monte Carlo weights** use `np.random.dirichlet(np.ones(n))` for *uniform* simplex coverage — do **not** revert to normalising `np.random.random(n)` (biases toward equal weights). `np.random.seed(777)` (in the entry point) makes runs deterministic.

### Keeping descriptions in sync

When you change *how* a metric is computed, update its matching entry in `descriptions.py` — the per-section "How to read this section" text states the formula/assumptions and will otherwise silently drift from the code.

**Per-period terminology:** user-facing labels/tooltips/descriptions say **per-period** (the data interval may be daily/weekly/monthly). Reserve "daily" for genuine daily-interval or calendar references only — the annualisation legend (`252/52/12`), the mixed-calendar weekend-folding caveat, and `.days` durations.

**Asset terminology:** user-facing text says **asset** (section headers, chart titles, tooltips, descriptions), not "ETF" — the portfolio can hold crypto/indices too. "ETF" is intentionally kept in only two places: (a) the **total-return reconstruction** feature, which genuinely targets *accumulating ETFs*; and (b) render **function identifiers** (`render_load_etf_data`, `render_per_etf_analytics`, `render_etf_prices`) — these were *not* renamed with the headers, so don't "tidy" them.

### Rendering gotchas (silent failures — verify in a browser, not just the source)

Requires **Streamlit ≥ 1.50** (uses `width="stretch"`). Last verified end-to-end on Python 3.14.4 · streamlit 1.62 · pandas 3.0.5 · numpy 2.5.2 · scipy 1.18 · yfinance 1.6 · playwright 1.62. **pandas 3.x** is a major bump past the 2.x this was written against — everything passes today, but it is the first suspect for an unexplained data-shaped failure.

- **`st.data_editor` with `key=`: never reassign its returned frame back into the session_state variable you pass as its input.** The "My Portfolio" editor seeds `st.session_state.portfolio_df` *once*; the widget then persists edits as a diff against that stable seed. Writing the returned (edited) frame back moves the baseline underneath the widget, dropping the pending value of a freshly-added row until it's typed twice. Read current state from the editor's *return value* (`edited_portfolio`), not the session_state key.
- **`st.data_editor` newly-added rows yield `None`/`NaN` cells.** A row where the user typed a ticker but not yet a Market Value comes through with value `None`; coerce (`0.0 if v is None or pd.isna(v) else float(v)`) before `sum()`/division, or it raises `TypeError: float + NoneType`.
- **`st.data_editor` renders its grid to a `<canvas>` (glide-data-grid), so cell and column text is *not* in the DOM.** Playwright assertions like `"Index" in inner_text()` always fail, and synthetic clicks/keystrokes do not reliably enter a cell's edit mode (Streamlit's own `AppTest` has no `data_editor` driver either). Assert on ARIA instead — `aria-colcount`/`aria-rowcount` on `[role='grid']`. The **My Portfolio** editor is still one of these; the extend wizard replaced its table with text inputs precisely because those *are* drivable.
- **Streamlit reruns shift the layout, so cached click coordinates go stale.** Measuring an element's position while a `st.spinner` is showing and clicking after it clears lands on the wrong element (a silent no-op, not an error). Re-screenshot or re-measure immediately before clicking.
- **Inline SVG (the data-availability gauge in `render_load_etf_data`) must use `st.markdown(html, unsafe_allow_html=True)`.** Do **not** use `st.html` — it runs input through DOMPurify (default config), which strips `<svg>` entirely, so the gauge renders as *nothing* with no error. `st.components.v1.html` (the old approach the gauge was migrated off of) is no longer used here, and that Streamlit API was itself removed after 2026-06-01 — don't reintroduce it. `st.iframe` only embeds a URL, not an HTML/SVG string. After any change here, confirm the SVG is actually in the DOM (e.g. Playwright `svg:has(linearGradient#gaugeGrad)`); the deprecation-warning-gone check alone is not enough.
- **KaTeX `$$…$$` blocks in `descriptions.py` must sit on a single source line.** A hard line break inside the block makes Streamlit's markdown treat it as a paragraph break, splitting the delimiters so the formula fails to parse (renders as raw red error text). The test's gauge check locates the inline SVG (not an iframe) for this reason. **Static pre-check:** scan for any source line with an odd count of `$$` — a balanced display block has two per line, so an odd count flags a split block (`python -c "import re;[print(i) for i,l in enumerate(open('efficient_frontier_app/descriptions.py',encoding='utf-8'),1) if l.count('\$\$')%2]"`). **Browser check:** expand every "📖 How to read this section" panel and assert the DOM has `>0` `.katex` nodes, **zero** `.katex-error` nodes, and **zero** stray `$$` in `document.body.innerText` (a broken block leaves the unparsed LaTeX as visible text). The static scan catches the common case offline; the browser check is the ground truth.

### Key Functions

**data_handling.py:**
- `compute_portfolio_returns_simple(merged_df)` — the single simple-returns series (`.pct_change().dropna()`) used by §5 stats and all optimization; also returns its mean and covariance
- `compute_rolling_returns(merged_df, window_periods)` — rolling **cumulative** simple returns over a moving window (`P_t/P_{t−w}−1`, whole-window total, not annualised — Issue 4)
- `build_merged_dataframe(tickers, folder_path, filename_suffix, filter_date)` — inner join of asset prices (common window; used by the portfolio sections)
- `load_asset_series(folder_path, ticker, filename_suffix, filter_date)` — single asset's `adj close` over its *own* full history (no inner join); used by §2 per-asset analytics. `usecols` ignores `_EXT`/`currency` extras
- `detect_stock_splits(tickers, folder_path, filename_suffix, filter_date)` — ground-truth splits from the yfinance `stock splits` column (date + ratio); skips files without the column. `check_price_anomalies(tickers, folder_path, filename_suffix, filter_date, z_threshold=8.0, min_abs_move=0.45)` — flags moves that are robust-MAD-z-score outliers *and* exceed a 45% floor (so genuine crypto swings and interval scaling don't false-trigger), cross-referenced against split dates. **Note: Adj Close is already split-adjusted, so a real split is *not* a price jump — splits come from the column, the anomaly check catches glitches / unadjusted splits.**
- **Total-return reconstruction:** `synthesize_total_return(price_index, etf, eurusd, periods_per_year)` — calibrate `q_hat` from the ETF overlap + return the spliced EUR series; `build_reconstructed_frame(index_prices, etf_prices, fx_prices, periods_per_year, currency=None)` — shape it into the `date,adj close,synthetic,recon_yield[,currency]` CSV frame; `run_total_return_reconstruction(jobs, intervals, output_dir, log_queue, convert_to_eur=True)` — threaded download+splice+save (mirrors `run_download`, including its EUR-conversion contract); `convert_series_to_eur(prices, eur_multiplier)` — Series-level wrapper over `apply_eur_conversion` used for the ETF leg; `read_synthetic_info(...)` — per-ticker tag metadata for the §1 badge
- **Currency:** `run_download(..., convert_to_eur=True)` — downloads, auto FX-converts non-EUR to EUR (unless overridden), writes a `currency` column; `detect_currency(symbol, yf)` / `fetch_eur_multiplier(currency, yf_interval, end_date, yf)` / `apply_eur_conversion(data, eur_multiplier)` — the conversion helpers (last one is pure, unit-tested); `read_currency_info(tickers, folder_path, filename_suffix)` — resolves each ticker's currency (stored column, else cached network sniff) for the §1 check

**portfolio_calculations.py:**
- `portfolio_annualised_performance(weights, mean_returns, cov_matrix, ...)` — portfolio return/volatility
- `cvar(returns, alpha=0.05)` — historical CVaR; `alpha` is tail quantile (portfolio CVaR/Max DD are computed directly from these base functions at display time — there are no `portfolio_cvar`/`portfolio_max_drawdown` wrappers)
- `max_drawdown(returns)` — maximum drawdown
- `random_portfolios(num, mean_returns, cov_matrix, ...)` — Monte Carlo simulation
- `random_portfolios_sortino(num, ...)` — returns 5 rows (std, ret, sharpe, sortino, downside_dev); used in sections 7 & 8
- **§6 rebalancing helpers:** `rebalanced_value_series(merged_df, tickers, weights, rebalance_every_periods=None)` — value series held at target weights, reset every K rows (`None`=buy-and-hold, `1`=per-period rebalanced), `V₀=1`; `rebalanced_value_aftertax(merged_df, tickers, weights, rebalance_every_periods=None, tax_rate=0.0)` — after-tax twin returning `(with_tax, netliq)`: tracks per-asset cost basis, taxes **net** realized gains at each reset (losses net within-period, no carry-forward), `netliq = with_tax − τ·max(0, with_tax − book)`; `tax_rate=0` reproduces `rebalanced_value_series` exactly (the §6 capital-gains-tax overlay, see toggle above); `underwater_episodes(value)` — list of peak/trough/recovery episodes; `deepest_drawdown_episode(value)` / `longest_underwater_episode(value)` (→ `(episode, days, ongoing)`); `downside_deviation_series(returns, N, rf)` — Sortino downside dev on a precomputed return series
- **Real-terms helpers:** `real_deflator(date_index, annual_inflation)` — cumulative deflator `(1+π)^(years since start)` indexed by date (all-ones at `π=0`); `to_real(level_series, annual_inflation)` — `level / real_deflator(...)`. Apply to *levels* (prices, value series), not returns. See **Real (inflation-adjusted) terms toggle**.

**ui_components.py:**
- `render_rolling_returns(rolling_returns, tickers, rolling_window_years)` — renders §4 (individual assets only; no portfolio chart, no portfolio arg)
- `render_input_portfolio_analysis(merged_df, portfolio_returns_simple, tickers, my_portfolio_allocation, annualisation_factor, risk_free_rate, alpha, window_periods, rolling_window_years, rebalance_every_periods, rebalance_label, real_terms=False, annual_inflation=0.0, mixed_calendar=False, seven_day_tickers=None, cgt_rate=0.0)` — renders §6 at the selected rebalancing cadence (see **Rebalancing basis**), deflated to real terms when `real_terms` (see **Real (inflation-adjusted) terms toggle**), and overlaid with an after-tax net-liq line when `cgt_rate > 0` (see **Capital-gains tax / net-liquidation overlay**); `_fmt_period(days, ongoing)` helper formats underwater durations. Ends by calling `render_tail_risk` for the merged tail-risk subsection
- `render_per_etf_analytics(tickers, folder_path, filename_suffix, filter_date_string, annualisation_factor, real_terms=False, annual_inflation=0.0)` loads each asset's own full history via `load_asset_series` (no `merged_df` arg — §2 is purely per-asset) and deflates its price series when `real_terms`; `render_monte_carlo(..., real_terms=False, mixed_calendar=False, seven_day_tickers=None)` / `render_scipy_ef(...)` take `real_terms` (nominal-basis caption) and `mixed_calendar`/`seven_day_tickers` (mixed-calendar caveat); they never deflate — the frontier weights are inflation-invariant
- `render_tail_risk(my_portfolio_returns, alpha)` — §6's Tail Risk & Return Distribution subsection (formerly the standalone VaR section): parametric + historical VaR/CVaR, mean/median/vol/skew/kurtosis, distribution histogram. Header-less; takes §6's already-computed per-period series (no recompute)
- `collect_portfolio_info(name, weights, mean_returns, cov_matrix, risk_free_rate, ...)` — returns dict with `ret` (arithmetic mean×N), `geo_ret` (compound CAGR of the constant-weight return series, via `_geometric_annual_return`), max_dd and port_returns (CVaR is computed later from port_returns); `collect_portfolio_info_mtc(name, index, results, weights_list, ...)` is the Monte Carlo variant reading from the simulation results array
- `display_portfolio_cards(portfolios, alpha)` — shows 7 metrics in a 4+3 grid: arithmetic **Average annual return** + **Compound return (CAGR)** beside it (Issue 6 — arithmetic is the optimizer/frontier input; realized CAGR differs by volatility drag down ≈σ²/2 vs intra-period compounding up, usually lower for volatile portfolios), Ann. Volatility, Sharpe, Sortino, Max Drawdown, and CVaR at the user's confidence level (label updates with α). A visible caption above the cards declares that **Max DD / CVaR use the per-period-rebalanced basis** and that buy-and-hold drawdown (per §6's default cadence) is usually larger (Issue 7, disclosure only — numbers unchanged)
