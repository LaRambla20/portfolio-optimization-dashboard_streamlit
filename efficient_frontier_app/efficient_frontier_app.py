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

from descriptions import DESCRIPTIONS

from ui_components import (
    render_load_etf_data,
    render_per_etf_analytics,
    render_etf_prices,
    render_returns_statistics,
    render_input_portfolio_analysis,
    render_monte_carlo,
    render_scipy_ef,
    render_rolling_returns,
    render_extend_wizard,
    wiz_reset,
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

# Radio labels for the Get Data panel. Both jobs download; the wording has to say so.
DL_MODE_PLAIN  = "Download ETF's price history as is"
DL_MODE_EXTEND = "Download & extend ETF's price history"

# ─────────────────────────────────────────────────────────
# SIDEBAR — ALL USER INPUTS
# ─────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")

# --- Data source ---
st.sidebar.subheader("Data Source")
folder_path = st.sidebar.text_input(
    "CSV folder path",
    value="individual_indices_data",
    help="Path to the folder containing the downloaded asset CSVs.",
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

# --- Get data (one panel: plain download, or download + history extension) ---
st.sidebar.subheader("📥 Get Data")
with st.sidebar.expander("Get data settings", expanded=False):
    st.caption(
        "Both options **download from Yahoo Finance** — nothing needs to be on disk first."
    )
    dl_mode = st.radio(
        "What do you want to do?",
        options=[DL_MODE_PLAIN, DL_MODE_EXTEND],
        index=0,
    )

    if dl_mode == DL_MODE_EXTEND:
        with st.expander("📖 How this works"):
            st.markdown(DESCRIPTIONS["extending_history"])

    # Ticker inputs are deliberately *never* pre-filled — examples live in the placeholder
    # and the column tooltips, so there is nothing to erase before each use.
    if dl_mode == DL_MODE_PLAIN:
        dl_tickers_raw = st.text_area(
            "Tickers (one per line)",
            value="",
            height=120,
            placeholder="VWCE.MI\nBTC-EUR\nDBMF",
            help="One ticker per line. They don't have to match your portfolio.",
        )
    else:
        dl_tickers_raw = ""
        st.caption(
            "A guided setup checks the fund, the index and the exchange rate against Yahoo "
            "one step at a time, so a ticker that won't work is caught before anything runs."
        )

    dl_intervals = st.multiselect(
        "Intervals to fetch",
        options=["daily", "weekly", "monthly"],
        default=[data_period],
        help="Which bar sizes to save. Defaults to the Data period selected above.",
    )
    # The default only binds on first render, so it can drift if Data period changes later.
    # Check live instead: fetching an interval you don't analyse is the classic missing-file trap.
    if dl_intervals and data_period not in dl_intervals:
        st.warning(
            f"You're analysing **{data_period}** data but not fetching it — "
            f"add \"{data_period}\", or Run Analysis will report a missing file."
        )

    if dl_mode == DL_MODE_PLAIN:
        dl_button = st.button("⬇️  Download", width="stretch")
        wizard_button = False
    else:
        dl_button = False
        wizard_button = st.button("🧬  Start guided setup", width="stretch", type="primary")

    with st.expander("⚙️ Advanced"):
        dl_columns_to_drop = st.multiselect(
            "Columns to drop",
            options=["Volume", "Capital Gains", "High", "Low", "Open", "Dividends", "Stock Splits"],
            default=["Volume", "Capital Gains", "High", "Low", "Open"],
            help="Applies to plain downloads only.",
        )
        dl_output_dir = st.text_input(
            "Output folder",
            value="",
            placeholder=folder_path,
            help="Leave blank to save into the CSV folder path above — they must match for "
                 "the app to find the files.",
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

    with st.expander("📖 How to find tickers"):
        st.markdown(DESCRIPTIONS["finding_tickers"])

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
    options=[1, 2, 3, 5, 7, 10],
    index=0,
    help="Window length for rolling return calculations",
)
rebalancing_frequency = st.sidebar.selectbox(
    "Rebalancing frequency",
    options=["Never", "Every 6 months", "Yearly", "Every period"],
    index=0,
    help="How often your portfolio is reset to its target weights. Governs §6 (Input Portfolio, "
         "incl. its tail-risk subsection) only — the §7/§8 efficient frontier always assumes "
         "per-period rebalancing, the basis MPT optimization requires. 'Never' = buy-and-hold "
         "(weights drift).",
)

real_terms = st.sidebar.checkbox(
    "Show real (inflation-adjusted) returns",
    value=False,
    help="Deflate by an assumed constant inflation rate so figures are in today's purchasing power. "
         "Affects §2 (Per-Asset Analytics) and §6 (Input Portfolio, incl. its tail risk). A constant "
         "rate lowers CAGR/average returns and deepens drawdowns, but leaves volatility, correlations "
         "and the §7/§8 efficient-frontier weights unchanged (the real risk premium is "
         "inflation-invariant), so those stay nominal.",
)
# The inflation input only appears when the toggle is on, so it's visually clear the two go
# together (an always-shown field would do nothing while the toggle is off — confusing).
if real_terms:
    annual_inflation_pct = st.sidebar.number_input(
        "Assumed annual inflation (%)",
        min_value=0.0, max_value=20.0, value=2.0, step=0.1, format="%.1f",
        help="Constant yearly inflation rate used to deflate §2 and §6 (e.g. Eurozone HICP ≈ 2%).",
    )
    annual_inflation = annual_inflation_pct / 100.0
else:
    annual_inflation = 0.0

# §6 capital-gains tax overlay: 0% (default) is an exact no-op, so the field is always shown.
cgt_rate = st.sidebar.number_input(
    "Capital-gains tax on realized gains (%)",
    min_value=0.0, max_value=60.0, value=0.0, step=1.0, format="%.0f",
    help="Adds an after-tax net-liquidation line to §6: this % is charged on net realized gains at "
         "each rebalance (losses net against gains, no carry-forward), plus tax owed on unrealized "
         "gains if liquidated today. 0% = off (e.g. Italy ≈ 26%).",
) / 100.0

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

# Rebalancing cadence (count-based: reset weights every K rows). K is derived from the data
# frequency so the calendar meaning is constant across daily/weekly/monthly data.
if rebalancing_frequency == "Every period":
    rebalance_every_periods = 1
    rebalance_label = "rebalanced every period"
elif rebalancing_frequency == "Every 6 months":
    rebalance_every_periods = max(1, round(annualisation_factor / 2))
    rebalance_label = "rebalanced every 6 months"
elif rebalancing_frequency == "Yearly":
    rebalance_every_periods = annualisation_factor
    rebalance_label = "rebalanced yearly"
else:  # "Never"
    rebalance_every_periods = None
    rebalance_label = "never rebalanced (buy-and-hold)"

tickers = list(raw_portfolio.keys())
total_invested = sum(raw_portfolio.values())
if total_invested > 0:
    my_portfolio_allocation = {k: v / total_invested for k, v in raw_portfolio.items()}
else:
    my_portfolio_allocation = {k: 1.0 / len(raw_portfolio) for k in raw_portfolio}

# ─────────────────────────────────────────────────────────
# GET DATA — job runner
# ─────────────────────────────────────────────────────────

def run_job_with_progress(title, target, args, kwargs, total_tasks, done_message):
    """Run a backend job on a thread, streaming its log queue into a live progress UI.

    Both backends share one contract: the log queue is their last positional argument,
    lines prefixed ✅/❌ each mark one finished task, and a final ``__DONE__done/total``
    carries the authoritative count.
    """
    st.subheader(title)
    log_box  = st.empty()
    prog_bar = st.progress(0)
    status   = st.empty()

    log_lines = []
    log_q = queue.Queue()
    t = threading.Thread(target=target, args=tuple(args) + (log_q,), kwargs=kwargs, daemon=True)
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
                if msg.startswith(("✅", "❌")):
                    completed += 1
                    prog_bar.progress(min(completed / total_tasks, 1.0))
                log_box.code("\n".join(log_lines[-30:]))
        except queue.Empty:
            pass

    t.join()
    status.success(f"{done_message} ({completed}/{total_tasks} processed)")
    st.divider()


if dl_button:
    # Blank output folder means "the folder the app reads from" — they have to match.
    output_dir = dl_output_dir.strip() or folder_path

    if not dl_intervals:
        st.warning("Select at least one interval to fetch.")

    elif dl_mode == DL_MODE_PLAIN:
        dl_tickers = [t.strip() for t in dl_tickers_raw.strip().splitlines() if t.strip()]
        if not dl_tickers:
            st.warning("No tickers entered — type at least one ticker, one per line.")
        else:
            run_job_with_progress(
                "📥 Downloading Data",
                run_download,
                (dl_tickers, dl_intervals, dl_columns_to_drop, output_dir),
                {"convert_to_eur": not dl_keep_native},
                len(dl_tickers) * len(dl_intervals),
                "Download complete.",
            )

# ─────────────────────────────────────────────────────────
# GUIDED EXTEND WIZARD
# ─────────────────────────────────────────────────────────

if wizard_button:
    if not dl_intervals:
        st.warning("Select at least one interval to fetch before starting the guided setup.")
    else:
        # The dialog can't read st.sidebar, so snapshot what it needs. The "primary" interval is
        # the one it validates and previews q_hat at; the run still covers every chosen interval.
        st.session_state["wiz_cfg"] = {
            "intervals": dl_intervals,
            "primary": data_period if data_period in dl_intervals else dl_intervals[0],
            "convert_to_eur": not dl_keep_native,
        }
        wiz_reset()
        st.session_state["wiz_open"] = True

if st.session_state.get("wiz_open"):
    render_extend_wizard()

# Queued by the wizard's final button; run out here because elements created outside a dialog
# accumulate across the dialog's own reruns.
queued_job = st.session_state.pop("recon_job", None)
if queued_job:
    cfg = st.session_state.get("wiz_cfg", {})
    job_intervals = cfg.get("intervals", [data_period])
    run_job_with_progress(
        "🧬 Downloading & Extending History",
        run_total_return_reconstruction,
        ([queued_job], job_intervals, dl_output_dir.strip() or folder_path),
        {"convert_to_eur": cfg.get("convert_to_eur", True)},
        len(job_intervals),
        f"Done — add `{queued_job['etf']}_EXT` to **My Portfolio** to analyse it.",
    )

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
render_per_etf_analytics(tickers, folder_path, filename_suffix, filter_date_string,
                         annualisation_factor, real_terms, annual_inflation)
render_etf_prices(merged_df, tickers)
render_rolling_returns(rolling_returns, tickers, rolling_window_years)
# Mixed-calendar flag (some 7-day crypto + some ~5-day equity) → covariance/correlation, and the
# frontier/VaR built on them, are calendar-approximate. Surface the §1 caveat at the point of use.
mixed_calendar = data_availability["mixed_calendar"]
seven_day_tickers = data_availability["seven_day_tickers"]

render_returns_statistics(portfolio_returns_simple, portfolio_mean_returns,
                           portfolio_cov_matrix, tickers, annualisation_factor,
                           risk_free_rate, mixed_calendar, seven_day_tickers)
render_input_portfolio_analysis(merged_df, portfolio_returns_simple, tickers,
                                my_portfolio_allocation, annualisation_factor, risk_free_rate,
                                alpha, window_periods, rolling_window_years,
                                rebalance_every_periods, rebalance_label,
                                real_terms, annual_inflation,
                                mixed_calendar, seven_day_tickers, cgt_rate)
render_monte_carlo(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                    tickers, annualisation_factor, risk_free_rate, num_portfolios, eps,
                    custom_target_ret, custom_target_vol, my_portfolio_allocation, alpha,
                    real_terms, mixed_calendar, seven_day_tickers)
render_scipy_ef(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                  tickers, annualisation_factor, risk_free_rate, num_portfolios,
                  num_eff_portfolios, eps, custom_target_ret, custom_target_vol,
                  my_portfolio_allocation, alpha, real_terms,
                  mixed_calendar, seven_day_tickers)

# End-of-pipeline signal (every section above has rendered). Also the e2e test's "done" marker.
st.success(" Analysis complete!")
