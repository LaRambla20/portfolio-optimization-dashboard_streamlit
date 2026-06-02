# Portfolio Optimization Dashboard

A Streamlit dashboard for **Modern Portfolio Theory** analysis. Load ETF price data, visualize the efficient frontier, run Monte Carlo simulations, and evaluate risk via VaR and CVaR.

## Features

- **Input portfolio analysis** — your allocation as a **buy-and-hold** (un-rebalanced) portfolio: allocation pie, cumulative-return chart with per-asset overlays, annotated underwater curve, deepest-drawdown recovery and longest-underwater-stretch durations, plus headline return/risk metrics (geometric CAGR, annualised return/volatility, Sharpe, Sortino, max drawdown, historical VaR/CVaR)
- **Efficient Frontier** — Monte Carlo simulation (uniform-simplex sampling) and SciPy optimization (SLSQP)
- **Risk metrics** — Sharpe & Sortino ratios, Max Drawdown, and both parametric (normal) and historical VaR/CVaR with fat-tail (skew/kurtosis) diagnostics
- **Per-ETF analytics** — CAGR, simple/calendar-year returns, look-back-period metrics, cumulative-return charts
- **Simple returns throughout** — one consistent return definition (actual realised % change) across every section and all optimization, so there's no return-type toggle to reason about
- **Rolling returns** — 1/5/10-year moving windows for individual assets (the portfolio's rolling returns live in the Input Portfolio Analysis section)
- **Built-in guidance** — every section has a "How to read this section" panel with plain-language explanations and formulas
- **Data download** — built-in yfinance downloader with progress streaming, **auto-converting non-EUR tickers to EUR** so the whole portfolio shares one currency (toggleable)
- **Total-return reconstruction** — extend a short-lived accumulating ETF backward with the longer **price-return** index it tracks: the missing dividend yield is calibrated from the ETF overlap, the older history is grossed up and spliced on, and reconstructed rows are flagged in section 1
- **Data-quality checks** — section 1 reports any **recorded stock splits** (read straight from yfinance's `stock splits` column — informational, since Adj Close is already split-adjusted) and flags **statistically anomalous price moves** with a robust, self-calibrating outlier test that adapts per asset and per interval (so genuine crypto swings aren't false-flagged)
- **Currency safety** — section 1 warns if any loaded series isn't in EUR (the app otherwise assumes a single base currency)
- **Flexible inputs** — configurable portfolio weights, confidence level, date filter

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org streamlit pandas numpy scipy matplotlib seaborn yfinance
```

> The `--trusted-host` flags work around SSL certificate issues common on Windows.

> **Requires Streamlit ≥ 1.50.** The app uses APIs from recent Streamlit (`width="stretch"`, and `st.markdown(..., unsafe_allow_html=True)` for the SVG data-availability gauge in place of the now-removed `st.components.v1.html`). Developed against Streamlit 1.58.

## Usage

1. **Run analysis** — pre-loaded CSVs for EM57.MI, VWCE.MI, SGLD.MI, DBMF, and BTC-EUR are included in `individual_indices_data/`. Configure your portfolio and parameters in the sidebar, then click **Run Analysis**.

2. **Add or refresh tickers** — expand the "Download ETF Data" panel in the sidebar, enter tickers in yfinance format (e.g. `IWDA.AS`, `BTC-EUR`), and click Download. Non-EUR tickers are auto-converted to EUR by default; tick **Keep native currency** to store raw prices instead.

3. **Extend an ETF with index history** *(optional)* — in the same panel, use **Total-return reconstruction** to pair a long price-return index (e.g. `^GSPC`) with the accumulating ETF that tracks it and an FX ticker (e.g. `EURUSD=X`). The result is saved as `{ETF}_EXT`; add it to your portfolio to analyze the extended history. The index must track the **same underlying** as the ETF (a recovered dividend yield outside ~0–4%/yr is the tell that it doesn't).

```bash
.venv\Scripts\streamlit run efficient_frontier_app/efficient_frontier_app.py
```

## Testing

A Playwright end-to-end test drives the full dashboard in a real browser. Start the app first, then run:

```bash
.venv\Scripts\python test_dashboard.py
```

This clicks **Run Analysis**, waits for all computations to finish, verifies all 9 section headers, checks portfolio cards, and saves screenshots to `test_screenshots/`.

For automated/CI runs, set `HEADLESS=1` to run without a visible browser window:

```bash
HEADLESS=1 .venv\Scripts\python test_dashboard.py
```

Unit tests for the total-return reconstruction and EUR-conversion logic run standalone (no app or network required):

```bash
.venv\Scripts\python test_total_return_synthesis.py
```

## Project Structure

```
efficient_frontier_app/
├── efficient_frontier_app.py   # Entry point — sidebar, orchestration
├── portfolio_calculations.py   # Pure math: optimization, Monte Carlo, VaR/CVaR
├── data_handling.py            # CSV loading, merging, return computation
├── ui_components.py            # Section renderers (9 sections)
└── descriptions.py             # Per-section "How to read this section" guides

individual_indices_data/        # ETF CSVs (pre-loaded samples included; downloader writes here)
```

## Data Format

CSVs must be named `{ticker}_data_{period}.csv` (e.g. `IWDA.AS_data_daily.csv`) with columns `date` and `adj close`. The built-in downloader produces this format automatically.

Downloaded files also carry a `currency` column, and reconstructed `{ticker}_EXT_data_{period}.csv` files add `synthetic` / `recon_yield` columns marking the reconstructed history. These extra columns are optional metadata — readers only need `date` and `adj close`.

## Dashboard Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Load ETF Data | Recorded stock-split report, price-anomaly detection, data availability gauge, non-EUR currency warning, reconstructed-history flag |
| 2 | Per-ETF Analytics | CAGR, returns, drawdown per asset |
| 3 | ETF Prices | Raw and normalized price charts |
| 3b | Rolling Returns | Moving-window returns for individual assets |
| 4 | Returns & Statistics | Covariance/correlation matrices, return distributions |
| 5 | Input Portfolio Analysis | Your allocation as a buy-and-hold portfolio: allocation pie, cumulative returns, annotated underwater curve, drawdown/recovery durations, headline metrics (geometric CAGR, arithmetic avg annual return, volatility, Sharpe, Sortino, max drawdown), historical VaR/CVaR, correlation heatmap |
| 6 | Monte Carlo EF | Random portfolio simulation (Sharpe & Sortino) |
| 7 | SciPy EF | Optimized efficient frontier via SLSQP |
| 8 | VaR Analysis | Per-period parametric & historical VaR/CVaR side by side, with skew/kurtosis fat-tail diagnostics |

## Data-quality checks (section 1)

Section 1 runs two **independent** checks on the raw prices, because a stock split and a bad price are different things:

**Recorded stock splits (📐).** Read directly from yfinance's `stock splits` column — the exact split ratio (e.g. `2.0` = 2-for-1, `0.1` = 1-for-10 reverse) on the exact ex-date, identical across daily/weekly/monthly because it's recorded data, not inferred from a price jump. Since the app analyses **Adj Close**, which yfinance has *already split-adjusted*, a split produces **no jump** in the series and is purely informational. Files without the column (legacy downloads, `_EXT` reconstructions) are simply skipped.

**Price-anomaly check (⚠️).** A fixed "flag any move > 60%" rule can't serve every asset and interval at once — a monthly bar compounds ~21 daily moves, and Bitcoin routinely swings further in a month than an equity ETF does in a year. Instead, each step-to-step return is standardised against the asset's *own* history using a fat-tail-resistant robust z-score:

```
z = (r − median(r)) / (1.4826 · MAD(r))
```

where `MAD` is the median absolute deviation. A move is flagged only when `|z| > 8` **and** the move exceeds a 45% floor. The floor sits above the largest genuine single-bar swings (even crypto rarely moves more than ~40% in a day), since real glitches and unadjusted splits move price by roughly half or double. Because the scale adapts per asset and per interval, normal high-volatility swings pass while a fat-finger tick, a currency mix-up, or an unadjusted split stands out.

Each flagged move is cross-referenced against the recorded split dates, shown in the **"On split date?"** column:

- **`yes`** — the anomalous move lands on a recorded split ex-date, so it's almost certainly just a split not reflected in this particular series (harmless).
- **`—`** — the move doesn't coincide with any split; this is the one to investigate as a possible data glitch.
