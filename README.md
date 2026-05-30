# Portfolio Optimization Dashboard

A Streamlit dashboard for **Modern Portfolio Theory** analysis. Load ETF price data, visualize the efficient frontier, run Monte Carlo simulations, and evaluate risk via VaR and CVaR.

## Features

- **Efficient Frontier** — Monte Carlo simulation and SciPy optimization (SLSQP)
- **Risk metrics** — Sharpe ratio, Sortino ratio, Max Drawdown, VaR, CVaR
- **Per-ETF analytics** — CAGR, simple/log returns, rolling metrics, cumulative return charts
- **Rolling returns** — 1/5/10-year moving windows for individual assets and portfolio
- **Data download** — Built-in yfinance downloader with progress streaming
- **Flexible inputs** — Configurable portfolio weights, return type, confidence level, date filter

## Setup

```bash
python -m venv .venv
.venv\Scripts\pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org streamlit pandas numpy scipy matplotlib seaborn yfinance
```

> The `--trusted-host` flags work around SSL certificate issues common on Windows.

## Usage

1. **Download data** — expand the "Download ETF Data" panel in the sidebar, enter tickers (yfinance format, e.g. `IWDA.AS`, `BTC-EUR`), and click Download. This creates `individual_indices_data/` with CSV files.

2. **Run analysis** — configure your portfolio and parameters in the sidebar, then click **Run Analysis**.

```bash
.venv\Scripts\streamlit run efficient_frontier_app/efficient_frontier_app.py
```

## Project Structure

```
efficient_frontier_app/
├── efficient_frontier_app.py   # Entry point — sidebar, orchestration
├── portfolio_calculations.py   # Pure math: optimization, Monte Carlo, VaR/CVaR
├── data_handling.py            # CSV loading, merging, return computation
└── ui_components.py            # Section renderers (8 sections)

individual_indices_data/        # Downloaded CSVs (gitignored, created on first download)
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
| 7 | VaR Analysis | Parametric VaR, CVaR, Monte Carlo VaR frontier |
