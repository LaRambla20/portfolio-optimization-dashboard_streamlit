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
    check_price_anomalies,
    detect_stock_splits,
    compute_data_availability,
    build_merged_dataframe,
    compute_portfolio_returns_simple,
    compute_rolling_returns,
    run_download,
    run_total_return_reconstruction,
    read_synthetic_info,
    read_currency_info,
)

from ui_components import (
    render_load_etf_data,
    render_per_etf_analytics,
    render_etf_prices,
    render_returns_statistics,
    render_input_portfolio_analysis,
    render_monte_carlo,
    render_scipy_ef,
    render_var_analysis,
    render_rolling_returns,
)

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
    index=2,
)
st.sidebar.caption(
    "💡 For long-term investors (≥ 10 years), monthly data is more meaningful: "
    "it smooths out short-term noise and better reflects the holding horizon."
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
    dl_keep_native = st.checkbox(
        "Keep native currency (don't convert to EUR)",
        value=False,
        help="By default, any ticker not trading in EUR is detected and its prices are "
             "scaled to EUR via the matching `{CCY}EUR=X` rate before saving, so the whole "
             "portfolio shares one currency (the app does no FX conversion elsewhere). Tick "
             "this to store the raw native-currency prices instead — §1 will then warn if "
             "they aren't EUR.",
    )
    dl_button = st.button("⬇️  Download now", width="stretch")

    st.markdown("---")
    st.markdown("**🧬 Total-return reconstruction** *(optional)*")
    st.caption(
        "Extend an accumulating ETF backward with a longer **price-return** index "
        "(most `^` / Stooq indices exclude dividends). The missing dividend yield is "
        "calibrated from the ETF overlap, the older index history is grossed up and "
        "spliced on, and the result is saved as `{ETF}_EXT`. Add an **FX ticker** "
        "(e.g. `EURUSD=X`) when the index is quoted in a different currency than the "
        "ETF, or the yield will absorb FX drift. Uses the *Intervals* selected above.\n\n"
        "⚠️ The index must track the **same underlying** as the ETF (e.g. an S&P 500 "
        "index with an S&P 500 ETF). Pairing mismatched underlyings (S&P 500 vs "
        "All-World) folds their performance gap into a meaningless yield — a recovered "
        "`q` outside ~0–4%/yr is the tell."
    )
    if "recon_jobs_df" not in st.session_state:
        st.session_state.recon_jobs_df = pd.DataFrame(
            [{"Index ticker": "^GSPC", "Calibrate vs ETF": "SXR8.DE", "FX ticker": "EURUSD=X"}]
        )
    recon_jobs_edited = st.data_editor(
        st.session_state.recon_jobs_df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Index ticker": st.column_config.TextColumn(
                "Index ticker", help="Long-history price-return index (e.g. ^GSPC)."),
            "Calibrate vs ETF": st.column_config.TextColumn(
                "Calibrate vs ETF", help="Accumulating ETF to calibrate the yield against."),
            "FX ticker": st.column_config.TextColumn(
                "FX ticker", help="USD-per-EUR pair (e.g. EURUSD=X). Leave blank if same currency."),
        },
        key="recon_editor",
    )
    st.session_state.recon_jobs_df = recon_jobs_edited
    recon_button = st.button("🧬  Reconstruct now", width="stretch")

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
    width="stretch",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", required=True),
        "Market Value (EUR)": st.column_config.NumberColumn(
            "Market Value (EUR)", min_value=0.0, format="%.2f"
        ),
    },
    key="portfolio_editor",
)
# NOTE: do *not* write `edited_portfolio` back into st.session_state.portfolio_df.
# With key= set, the editor already persists edits as a diff against this stable
# seed. Reassigning the returned (edited) frame moves that baseline underneath the
# widget, which drops the pending value of a freshly-added row until it's typed
# twice. The seed is set once above; the editor manages its own state thereafter.

raw_portfolio = dict(zip(edited_portfolio["Ticker"], edited_portfolio["Market Value (EUR)"]))
# A freshly-added row may have a ticker but a blank Market Value, which comes
# through as None/NaN — coerce to 0.0 so sum()/division below don't choke.
raw_portfolio = {
    k: (0.0 if v is None or pd.isna(v) else float(v))
    for k, v in raw_portfolio.items()
    if k and k.strip()
}

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
            kwargs={"convert_to_eur": not dl_keep_native},
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
# RECONSTRUCTION PANEL
# ─────────────────────────────────────────────────────────

if recon_button:
    recon_jobs = [
        {
            "index": str(r["Index ticker"]).strip(),
            "etf": str(r["Calibrate vs ETF"]).strip(),
            "fx": str(r.get("FX ticker") or "").strip(),
        }
        for _, r in recon_jobs_edited.iterrows()
        if str(r["Index ticker"]).strip() and str(r["Calibrate vs ETF"]).strip()
    ]
    if not recon_jobs:
        st.warning("Add at least one row with both an index ticker and a calibrating ETF.")
    elif not dl_intervals:
        st.warning("Select at least one interval (in the Download settings above).")
    else:
        st.subheader("🧬 Reconstructing Total-Return History")
        rec_log_box  = st.empty()
        rec_prog_bar = st.progress(0)
        rec_status   = st.empty()

        rec_log_lines = []
        rec_log_q = queue.Queue()
        rec_total = len(recon_jobs) * len(dl_intervals)

        rt = threading.Thread(
            target=run_total_return_reconstruction,
            args=(recon_jobs, dl_intervals, dl_output_dir, rec_log_q),
            daemon=True,
        )
        rt.start()

        rec_completed = 0
        while rt.is_alive() or not rec_log_q.empty():
            try:
                msg = rec_log_q.get(timeout=0.2)
                if msg.startswith("__DONE__"):
                    rec_completed = int(msg.replace("__DONE__", "").split("/")[0])
                    rec_prog_bar.progress(1.0)
                else:
                    rec_log_lines.append(msg)
                    if any(msg.startswith(p) for p in ["✅", "❌"]):
                        rec_prog_bar.progress(min(len(
                            [m for m in rec_log_lines if m.startswith(("✅", "❌"))]
                        ) / rec_total, 1.0))
                    rec_log_box.code("\n".join(rec_log_lines[-30:]))
            except queue.Empty:
                pass

        rt.join()
        rec_status.success(
            f"Reconstruction complete — {rec_total} job(s) processed. "
            "Add the new `{ETF}_EXT` ticker(s) to **My Portfolio** to analyze them."
        )
        st.divider()

# ─────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────

run = st.button("▶  Run Analysis", type="primary", width="stretch")

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
    split_events = detect_stock_splits(tickers, folder_path, filename_suffix, filter_date_string)
    anomaly_warnings = check_price_anomalies(tickers, folder_path, filename_suffix, filter_date_string)
    data_availability = compute_data_availability(tickers, folder_path, filename_suffix, filter_date_string)
    synthetic_info = read_synthetic_info(tickers, folder_path, filename_suffix, filter_date_string)
    currency_info = read_currency_info(tickers, folder_path, filename_suffix)
    merged_df = build_merged_dataframe(tickers, folder_path, filename_suffix, filter_date_string)
    portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix = compute_portfolio_returns_simple(merged_df)
    rolling_returns = compute_rolling_returns(merged_df, window_periods)

if len(merged_df) < window_periods:
    st.warning(
        f"Rolling window ({window_periods} periods) exceeds data length ({len(merged_df)}). "
        f"Rolling returns will have insufficient data."
    )

# ─────────────────────────────────────────────────────────
# RENDER SECTIONS
# ─────────────────────────────────────────────────────────

render_load_etf_data(tickers, split_events, anomaly_warnings, data_availability, synthetic_info, currency_info)
render_per_etf_analytics(merged_df, tickers, folder_path, filename_suffix, filter_date_string, annualisation_factor)
render_etf_prices(merged_df, tickers)
render_rolling_returns(rolling_returns, tickers, rolling_window_years)
render_returns_statistics(portfolio_returns_simple, portfolio_mean_returns,
                           portfolio_cov_matrix, tickers, annualisation_factor,
                           risk_free_rate)
render_input_portfolio_analysis(merged_df, portfolio_returns_simple, tickers,
                                my_portfolio_allocation, annualisation_factor, risk_free_rate,
                                alpha, window_periods, rolling_window_years)
render_monte_carlo(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                    tickers, annualisation_factor, risk_free_rate, num_portfolios, eps,
                    custom_target_ret, custom_target_vol, my_portfolio_allocation, alpha)
render_scipy_ef(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                  tickers, annualisation_factor, risk_free_rate, num_portfolios,
                  num_eff_portfolios, eps, custom_target_ret, custom_target_vol,
                  my_portfolio_allocation, alpha)
render_var_analysis(portfolio_returns_simple, alpha, my_portfolio_allocation)
