# Portfolio Optimization Dashboard

A Streamlit dashboard for **Modern Portfolio Theory** analysis. Load ETF price data, visualize the efficient frontier, run Monte Carlo simulations, and evaluate risk via VaR and CVaR.

## Features

- **Efficient Frontier** — Monte Carlo simulation (uniform-simplex sampling) and SciPy optimization (SLSQP)
- **Risk metrics** — Sharpe & Sortino ratios, Max Drawdown, and both parametric (normal) and historical VaR/CVaR with fat-tail (skew/kurtosis) diagnostics
- **Per-ETF analytics** — CAGR, simple/calendar-year returns, look-back-period metrics, cumulative-return charts
- **Rolling returns** — 1/5/10-year moving windows for individual assets and the portfolio
- **Built-in guidance** — every section has a "How to read this section" panel with plain-language explanations and formulas
- **Data download** — built-in yfinance downloader with progress streaming
- **Flexible inputs** — configurable portfolio weights, return type, confidence level, date filter

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org streamlit pandas numpy scipy matplotlib seaborn yfinance
```

> The `--trusted-host` flags work around SSL certificate issues common on Windows.

> **Requires Streamlit ≥ 1.50.** The app uses APIs from recent Streamlit (`width="stretch"`, and `st.markdown(..., unsafe_allow_html=True)` for the SVG data-availability gauge in place of the now-removed `st.components.v1.html`). Developed against Streamlit 1.58.

## Usage

1. **Run analysis** — pre-loaded CSVs for EM57.MI, VWCE.MI, SGLD.MI, and BTC-EUR are included in `individual_indices_data/`. Configure your portfolio and parameters in the sidebar, then click **Run Analysis**.

2. **Add or refresh tickers** — expand the "Download ETF Data" panel in the sidebar, enter tickers in yfinance format (e.g. `IWDA.AS`, `BTC-EUR`), and click Download.

```bash
.venv\Scripts\streamlit run efficient_frontier_app/efficient_frontier_app.py
```

## Testing

A Playwright end-to-end test drives the full dashboard in a real browser. Start the app first, then run:

```bash
.venv\Scripts\python test_dashboard.py
```

This clicks **Run Analysis**, waits for all computations to finish, verifies all 8 section headers, checks portfolio cards, and saves screenshots to `test_screenshots/`.

For automated/CI runs, set `HEADLESS=1` to run without a visible browser window:

```bash
HEADLESS=1 .venv\Scripts\python test_dashboard.py
```

## Project Structure

```
efficient_frontier_app/
├── efficient_frontier_app.py   # Entry point — sidebar, orchestration
├── portfolio_calculations.py   # Pure math: optimization, Monte Carlo, VaR/CVaR
├── data_handling.py            # CSV loading, merging, return computation
├── ui_components.py            # Section renderers (8 sections)
└── descriptions.py             # Per-section "How to read this section" guides

individual_indices_data/        # ETF CSVs (pre-loaded samples included; downloader writes here)
```

## Data Format

CSVs must be named `{ticker}_data_{period}.csv` (e.g. `IWDA.AS_data_daily.csv`) with columns `date` and `adj close`. The built-in downloader produces this format automatically.

## Dashboard Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Load ETF Data | Spike detection, data availability gauge |
| 2 | Per-ETF Analytics | CAGR, returns, drawdown per asset |
| 3 | ETF Prices | Raw and normalized price charts |
| 3b | Rolling Returns | Moving-window returns for assets and portfolio |
| 4 | Returns & Statistics | Covariance/correlation matrices, return distributions |
| 5 | Monte Carlo EF | Random portfolio simulation (Sharpe & Sortino) |
| 6 | SciPy EF | Optimized efficient frontier via SLSQP |
| 7 | VaR Analysis | Per-period parametric & historical VaR/CVaR side by side, with skew/kurtosis fat-tail diagnostics |
