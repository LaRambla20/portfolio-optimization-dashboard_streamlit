"""
Main entry point for Efficient Frontier dashboard.
C-like procedural flow: sidebar inputs → data loading → compute → render UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import queue
import threading
import os

from data_handling import (
    check_price_spikes,
    compute_data_availability,
    build_merged_dataframe,
    compute_returns,
    compute_portfolio_returns_simple,
    compute_rolling_returns,
    run_download,
)

from ui_components import (
    render_load_etf_data,
    render_per_etf_analytics,
    render_etf_prices,
    render_returns_statistics,
    render_monte_carlo,
    render_scipy_ef,
    render_var_analysis,
    render_rolling_returns,
)

from portfolio_calculations import compute_portfolio_rolling_returns

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Efficient Frontier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Efficient Frontier — Modern Portfolio Theory")

# ─────────────────────────────────────────────────────────
# SIDEBAR — ALL USER INPUTS
# ─────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")

# --- Data source ---
st.sidebar.subheader("Data Source")
folder_path = st.sidebar.text_input(
    "CSV folder path",
    value="individual_indices_data",
    help="Path to the folder containing the downloaded ETF CSVs.",
)
data_period = st.sidebar.selectbox(
    "Data period",
    options=["daily", "weekly", "monthly"],
    index=0,
)

# --- Download ---
st.sidebar.subheader("📥 Download ETF Data")
with st.sidebar.expander("Download settings", expanded=False):
    dl_tickers_raw = st.text_area(
        "Tickers to download (one per line)",
        value="EM57.MI\nVWCE.MI\nSGLD.MI\nBTC-EUR",
        height=120,
        help="These are the tickers passed to yfinance. They don't have to match your portfolio.",
    )
    dl_intervals = st.multiselect(
        "Intervals to download",
        options=["daily", "weekly", "monthly"],
        default=["daily"],
    )
    dl_columns_to_drop = st.multiselect(
        "Columns to drop",
        options=["Volume", "Capital Gains", "High", "Low", "Open", "Dividends", "Stock Splits"],
        default=["Volume", "Capital Gains", "High", "Low", "Open"],
    )
    dl_output_dir = st.text_input(
        "Download output folder",
        value="individual_indices_data",
        help="Must match the CSV folder path used for analysis.",
    )
    dl_button = st.button("⬇️  Download now", use_container_width=True)

# --- Portfolio definition ---
st.sidebar.subheader("My Portfolio")
st.sidebar.markdown(
    "Add tickers and their **current market value (EUR)**. "
    "Use the same ticker symbols as the CSV filenames."
)

default_portfolio = {
    "EM57.MI": 2900.34,
    "VWCE.MI": 5226.69,
    "SGLD.MI": 1994.88,
    "BTC-EUR": 0.0,
}

if "portfolio_df" not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(
        list(default_portfolio.items()), columns=["Ticker", "Market Value (EUR)"]
    )

edited_portfolio = st.sidebar.data_editor(
    st.session_state.portfolio_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", required=True),
        "Market Value (EUR)": st.column_config.NumberColumn(
            "Market Value (EUR)", min_value=0.0, format="%.2f"
        ),
    },
    key="portfolio_editor",
)
st.session_state.portfolio_df = edited_portfolio

raw_portfolio = dict(zip(edited_portfolio["Ticker"], edited_portfolio["Market Value (EUR)"]))
raw_portfolio = {k: v for k, v in raw_portfolio.items() if k and k.strip()}

# --- Simulation parameters ---
st.sidebar.subheader("Simulation Parameters")
num_portfolios = st.sidebar.number_input(
    "Number of random portfolios", min_value=1000, max_value=100000, value=25000, step=1000
)
num_eff_portfolios = st.sidebar.number_input(
    "Points on efficient frontier", min_value=10, max_value=500, value=50, step=10
)
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (€STR)", min_value=0.0, max_value=1.0, value=0.01932, step=0.001, format="%.5f"
)
return_type = st.sidebar.radio(
    "Return type", options=["logarithmic", "simple"], index=0
)

rolling_window_years = st.sidebar.selectbox(
    "Rolling return window (years)",
    options=[1, 5, 10],
    index=0,
    help="Window length for rolling return calculations",
)

alpha = st.sidebar.slider(
    "VaR confidence level (α)", min_value=0.80, max_value=0.99, value=0.95, step=0.01
)
eps = st.sidebar.number_input(
    "Search epsilon (ε)", min_value=0.0001, max_value=0.05, value=0.001, step=0.0001, format="%.4f"
)

# --- Date filter ---
st.sidebar.subheader("Date Filter")
filter_date = st.sidebar.date_input(
    "Filter data up to date",
    value=datetime.date.today(),
    max_value=datetime.date.today(),
)
filter_date_string = pd.to_datetime(filter_date)

# --- Custom portfolios ---
st.sidebar.subheader("Custom Portfolio Targets")
custom_target_ret_input = st.sidebar.text_input(
    "Custom target return (blank = skip)", value="", placeholder="e.g. 0.12"
)
custom_target_vol_input = st.sidebar.text_input(
    "Custom target volatility (blank = skip)", value="", placeholder="e.g. 0.08"
)
custom_target_VaR_input = st.sidebar.text_input(
    "Custom target VaR (blank = skip)", value="0.05", placeholder="e.g. 0.05"
)

def parse_optional_float(s):
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

custom_target_ret = parse_optional_float(custom_target_ret_input)
custom_target_vol = parse_optional_float(custom_target_vol_input)
custom_target_VaR = parse_optional_float(custom_target_VaR_input)

np.random.seed(777)

# ─────────────────────────────────────────────────────────
# DERIVED PARAMETERS
# ─────────────────────────────────────────────────────────

if data_period == "daily":
    annualisation_factor = 252
    filename_suffix = "_data_daily.csv"
elif data_period == "weekly":
    annualisation_factor = 52
    filename_suffix = "_data_weekly.csv"
elif data_period == "monthly":
    annualisation_factor = 12
    filename_suffix = "_data_monthly.csv"

window_periods = rolling_window_years * annualisation_factor

tickers = list(raw_portfolio.keys())
total_invested = sum(raw_portfolio.values())
if total_invested > 0:
    my_portfolio_allocation = {k: v / total_invested for k, v in raw_portfolio.items()}
else:
    my_portfolio_allocation = {k: 1.0 / len(raw_portfolio) for k in raw_portfolio}

# ─────────────────────────────────────────────────────────
# DOWNLOAD PANEL
# ─────────────────────────────────────────────────────────

if dl_button:
    dl_tickers = [t.strip() for t in dl_tickers_raw.strip().splitlines() if t.strip()]
    if not dl_tickers:
        st.warning("No tickers entered — nothing to download.")
    elif not dl_intervals:
        st.warning("Select at least one interval to download.")
    else:
        st.subheader("📥 Downloading ETF Data")
        log_box  = st.empty()
        prog_bar = st.progress(0)
        status   = st.empty()

        log_lines = []
        log_q = queue.Queue()
        total_tasks = len(dl_tickers) * len(dl_intervals)

        t = threading.Thread(
            target=run_download,
            args=(dl_tickers, dl_intervals, dl_columns_to_drop, dl_output_dir, log_q),
            daemon=True,
        )
        t.start()

        completed = 0
        while t.is_alive() or not log_q.empty():
            try:
                msg = log_q.get(timeout=0.2)
                if msg.startswith("__DONE__"):
                    completed = int(msg.replace("__DONE__", "").split("/")[0])
                    prog_bar.progress(1.0)
                else:
                    log_lines.append(msg)
                    if any(msg.startswith(p) for p in ["✅", "❌"]):
                        completed += 1
                        prog_bar.progress(min(completed / total_tasks, 1.0))
                    log_box.code("\n".join(log_lines[-30:]))
            except queue.Empty:
                pass

        t.join()
        status.success(f"Download complete — {completed}/{total_tasks} files processed.")
        st.divider()

# ─────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────

run = st.button("▶  Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Configure parameters in the sidebar, then click **Run Analysis**.")
    st.stop()

# ─────────────────────────────────────────────────────────
# CHECK MISSING FILES
# ─────────────────────────────────────────────────────────

if not tickers:
    st.error("No tickers defined in the portfolio. Please add at least one.")
    st.stop()

missing_files = []
for ticker in tickers:
    fp = os.path.join(folder_path, ticker + filename_suffix)
    if not os.path.exists(fp):
        missing_files.append(fp)

if missing_files:
    st.error("The following CSV files were not found:\n\n" + "\n".join(f"- `{f}`" for f in missing_files))
    st.info(
        "Run `download_etf_data_yahoofinance.py` first to download the data, "
        "then point the **CSV folder path** to the output folder."
    )
    st.stop()

# ─────────────────────────────────────────────────────────
# LOAD DATA & COMPUTE
# ─────────────────────────────────────────────────────────

with st.spinner("Loading data and computing..."):
    spike_warnings = check_price_spikes(tickers, folder_path, filename_suffix, filter_date_string)
    data_availability = compute_data_availability(tickers, folder_path, filename_suffix, filter_date_string)
    merged_df = build_merged_dataframe(tickers, folder_path, filename_suffix, filter_date_string)
    returns = compute_returns(merged_df, return_type)
    portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix = compute_portfolio_returns_simple(merged_df)
    rolling_returns = compute_rolling_returns(merged_df, window_periods, return_type)
    portfolio_rolling_returns = compute_portfolio_rolling_returns(
        np.array(list(my_portfolio_allocation.values())),
        portfolio_returns_simple,
        window_periods
    )

if len(merged_df) < window_periods:
    st.warning(
        f"Rolling window ({window_periods} periods) exceeds data length ({len(merged_df)}). "
        f"Rolling returns will have insufficient data."
    )

# ─────────────────────────────────────────────────────────
# RENDER SECTIONS
# ─────────────────────────────────────────────────────────

render_load_etf_data(tickers, spike_warnings, data_availability)
render_per_etf_analytics(merged_df, tickers, folder_path, filename_suffix, filter_date_string, return_type, annualisation_factor)
render_etf_prices(merged_df, tickers)
render_rolling_returns(rolling_returns, portfolio_rolling_returns, tickers,
                        rolling_window_years, return_type)
render_returns_statistics(returns, portfolio_returns_simple, portfolio_mean_returns,
                           portfolio_cov_matrix, tickers, return_type, annualisation_factor,
                           risk_free_rate)
render_monte_carlo(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                    tickers, annualisation_factor, risk_free_rate, num_portfolios, eps,
                    custom_target_ret, custom_target_vol, my_portfolio_allocation, alpha)
render_scipy_ef(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                  tickers, annualisation_factor, risk_free_rate, num_portfolios,
                  num_eff_portfolios, eps, custom_target_ret, custom_target_vol,
                  my_portfolio_allocation, alpha)
render_var_analysis(portfolio_returns_simple, alpha, my_portfolio_allocation)
