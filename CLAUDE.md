# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **No tests, build system, or linter** are configured.

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

## Git

Remote: https://github.com/LaRambla20/portfolio-optimization-dashboard_streamlit.git
Default branch: `master`

**Push requires Windows native cert store** (same SSL issue as pip):
```bash
git -c http.sslBackend=schannel push
```

## Architecture

Modular Streamlit dashboard split into 4 files inside `efficient_frontier_app/`.
- **`efficient_frontier_app/efficient_frontier_app.py`** — main entry point: sidebar inputs, derived parameters, orchestrates data loading and UI rendering.
- **`efficient_frontier_app/portfolio_calculations.py`** — pure math: portfolio performance, optimization (SciPy), Monte Carlo simulation, VaR calculations.
- **`efficient_frontier_app/data_handling.py`** — data operations: CSV loading, validation, price spike checks, merged dataframe building, return computations. Uses `@st.cache_data`.
- **`efficient_frontier_app/ui_components.py`** — UI renderers: 8 section functions (`render_load_etf_data`, `render_per_etf_analytics`, `render_etf_prices`, `render_returns_statistics`, `render_monte_carlo`, `render_scipy_ef`, `render_var_analysis`, `render_rolling_returns`) plus shared helpers (`collect_portfolio_info`, `display_portfolio_cards`). All data passed as explicit parameters (C-like style).
(No legacy files currently on disk — the repo contains only the 4-file modular app.)

### Critical Return Type Split

The code maintains two separate return series:
- `returns` — user-selected type (`return_type` toggle: "logarithmic" or "simple") used **only for single-asset display metrics**
- `portfolio_returns_simple` — always simple returns, used for **all portfolio optimization** (Monte Carlo, SciPy, VaR)

This split fixes the mathematical error where `sum(weights * log_returns)` incorrectly computes portfolio log returns (log returns aren't additive across assets).

Key variables for optimization:
- `portfolio_returns_simple` — simple returns dataframe
- `portfolio_mean_returns` — mean of simple returns
- `portfolio_cov_matrix` — covariance of simple returns

### Section Structure

1. **Load ETF Data** — validates CSVs, checks price spikes, shows data availability gauge
2. **Per-ETF Analytics** — CAGR, simple/log returns, rolling metrics, **Max Drawdown** (always displays simple returns for consistency)
3. **Build Merged Dataframe** — inner join of asset prices, normalized charts
3b. **Rolling Returns** — moving-window returns (1/5/10 years) for individual assets and portfolio allocation
4. **Returns & Statistics** — covariance/correlation matrices (from simple returns), daily returns plot
5. **Monte Carlo Efficient Frontier** — random portfolio simulation, Sortino ratios
6. **SciPy Efficient Frontier** — optimization via `scipy.optimize.minimize`, efficient frontier line
7. **VaR Analysis** — parametric VaR, **CVaR (Conditional Value at Risk)**, Monte Carlo VaR frontier

### Data Format

CSV files in `individual_indices_data/` named `{ticker}_data_{period}.csv` (period: daily/weekly/monthly) with columns: `date`, `adj close`.

### VaR / CVaR Gotchas

- **`cvar(returns, alpha)` expects the tail quantile** (e.g., `alpha=0.05` for 95% CVaR), NOT the confidence level (`0.95`). The sidebar `alpha` is the confidence level — pass `1 - alpha` to `cvar()`.
- **Sign conventions differ**: `portfolio_annualised_performance_VaR()` returns VaR as a **positive loss amount**. `cvar()` returns CVaR as a **negative return** (e.g., -0.15 = 15% expected shortfall). Keep this in mind when displaying both metrics side-by-side.

### Key Functions

**data_handling.py:**
- `compute_returns(merged_df, return_type)` — return series (simple or logarithmic)
- `compute_portfolio_returns_simple(merged_df)` — simple returns for portfolio optimization
- `compute_rolling_returns(merged_df, window_periods, return_type)` — rolling returns over moving window
- `build_merged_dataframe(tickers, folder_path, filename_suffix, filter_date)` — inner join of asset prices
- `check_price_spikes(tickers, folder_path, filename_suffix, filter_date)` — detects >60% price moves

**portfolio_calculations.py:**
- `portfolio_annualised_performance(weights, mean_returns, cov_matrix, ...)` — portfolio return/volatility
- `portfolio_annualised_performance_VaR(weights, mean_returns, cov_matrix, alpha, ...)` — also returns CVaR
- `compute_portfolio_rolling_returns(weights, returns_simple, window_periods)` — portfolio rolling returns
- `cvar(returns, alpha=0.05)` — historical CVaR; `alpha` is tail quantile
- `max_drawdown(returns)` — maximum drawdown
- `portfolio_cvar(weights, returns, alpha)` / `portfolio_max_drawdown(weights, returns)` — portfolio-level CVaR and Max DD
- `random_portfolios(num, mean_returns, cov_matrix, ...)` — Monte Carlo simulation
- `random_portfolios_sortino(num, ...)` — returns 5 rows (std, ret, sharpe, sortino, downside_dev); used in sections 5 & 6
- `random_portfolios_VaR(num, ...)` — returns 5 rows (std, return, sharpe, VaR, CVaR)

**ui_components.py:**
- `render_rolling_returns()` — renders 3b. Rolling Returns section
- `collect_portfolio_info()` — returns dict with max_dd and cvar fields
- `display_portfolio_cards()` — shows 6 metrics including Max Drawdown and CVaR
