"""
Data handling functions for Efficient Frontier dashboard.
Handles CSV loading, validation, price spike checks, data merging, and return computations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import threading
import queue


def evaluate_simple_return(prices, start_date, end_date):
    sub = prices.loc[start_date:end_date]
    start_price = sub.iloc[0]
    end_price = sub.iloc[-1]
    return end_price / start_price - 1


def evaluate_CAGR(prices, start_date, end_date):
    sub = prices.loc[start_date:end_date]
    start_price = sub.iloc[0]
    end_price = sub.iloc[-1]
    n_years = (sub.index[-1] - sub.index[0]).days / 365.25
    return (end_price / start_price) ** (1 / n_years) - 1


def evaluate_return_metrics(returns, start_date, end_date):
    sub = returns.loc[start_date:end_date]
    std_dev = sub.std()
    mean = sub.mean()
    return std_dev, mean


@st.cache_data
def check_price_spikes(tickers, folder_path, filename_suffix, filter_date_string, spike_threshold=0.60):
    spike_warnings = {}
    for ticker in tickers:
        fp = os.path.join(folder_path, ticker + filename_suffix)
        df = pd.read_csv(fp, usecols=["date", "adj close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= filter_date_string].sort_values("date").reset_index(drop=True)
        df["_pct"] = df["adj close"].pct_change()
        hits = df[df["_pct"].abs() > spike_threshold].dropna(subset=["_pct"])
        if not hits.empty:
            spike_warnings[ticker] = [
                (row["date"].strftime("%Y-%m-%d"), df.loc[idx - 1, "adj close"], row["adj close"], row["_pct"])
                for idx, row in hits.iterrows()
                if idx > 0
            ]
    return spike_warnings


@st.cache_data
def compute_data_availability(tickers, folder_path, filename_suffix, filter_date_string):
    ticker_starts = []
    ticker_ends = []
    seven_day_tickers = []
    for ticker in tickers:
        fp = os.path.join(folder_path, ticker + filename_suffix)
        df = pd.read_csv(fp, usecols=["date"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= filter_date_string]
        ticker_starts.append(df["date"].min())
        ticker_ends.append(df["date"].max())
        # Assets trading 7 days/week (e.g. crypto) have substantial weekend dates; equity ETFs ~none.
        if len(df) and (df["date"].dt.dayofweek >= 5).mean() > 0.10:
            seven_day_tickers.append(ticker)

    common_start = max(ticker_starts)
    common_end = min(ticker_ends)
    total_days = max((common_end - common_start).days, 0)
    total_years = total_days / 365.25
    # Mixed-calendar baskets (some 7-day, some 5-day) lose weekend moves to the inner join.
    mixed_calendar = 0 < len(seven_day_tickers) < len(tickers)

    return {
        "common_start": common_start,
        "common_end": common_end,
        "total_years": total_years,
        "ticker_starts": ticker_starts,
        "ticker_ends": ticker_ends,
        "seven_day_tickers": seven_day_tickers,
        "mixed_calendar": mixed_calendar,
    }


@st.cache_data
def build_merged_dataframe(tickers, folder_path, filename_suffix, filter_date_string):
    merged_df = None
    for ticker in tickers:
        file_path = os.path.join(folder_path, ticker + filename_suffix)
        df = pd.read_csv(file_path)
        subdf = df.iloc[:, :2].copy()
        subdf["date"] = pd.to_datetime(subdf["date"])
        subdf[ticker] = df["adj close"]
        subdf = subdf[["date", ticker]]
        subdf = subdf[subdf["date"] <= filter_date_string]
        if merged_df is None:
            merged_df = subdf
        else:
            merged_df = pd.merge(merged_df, subdf, on="date", how="inner")
    return merged_df


@st.cache_data
def compute_returns(merged_df, return_type):
    table = merged_df.set_index("date")
    if return_type == "simple":
        returns = table.pct_change()
    else:
        returns = np.log(table / table.shift(1))
    returns = returns.dropna()
    return returns


@st.cache_data
def compute_portfolio_returns_simple(merged_df):
    table = merged_df.set_index("date")
    portfolio_returns_simple = table.pct_change().dropna()
    portfolio_mean_returns = portfolio_returns_simple.mean()
    portfolio_cov_matrix = portfolio_returns_simple.cov()
    return portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix


@st.cache_data
def compute_rolling_returns(merged_df, window_periods, return_type):
    """Rolling return for each ticker: (price[t]/price[t-window_periods])-1 (simple) or log ratio."""
    price_df = merged_df.set_index("date")
    if return_type == "logarithmic":
        rolling = np.log(price_df) - np.log(price_df).shift(window_periods)
    else:
        rolling = price_df / price_df.shift(window_periods) - 1
    return rolling.reset_index()


def run_download(tickers, intervals, columns_to_drop, output_dir, log_queue):
    import yfinance as yf

    interval_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    suffix_map   = {"daily": "_data_daily.csv", "weekly": "_data_weekly.csv", "monthly": "_data_monthly.csv"}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_queue.put(f"Created folder: {output_dir}")

    today = datetime.date.today().strftime("%Y-%m-%d")
    total = len(tickers) * len(intervals)
    done  = 0

    for interval_period in intervals:
        yf_interval = interval_map[interval_period]
        for symbol in tickers:
            log_queue.put(f"Downloading {symbol} ({interval_period})...")
            try:
                ticker_obj = yf.Ticker(symbol)
                data = ticker_obj.history(
                    start="1900-01-01", end=today,
                    interval=yf_interval, auto_adjust=False,
                )
                data.reset_index(inplace=True)
                if pd.api.types.is_datetime64_any_dtype(data["Date"]):
                    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
                # Lowercase all columns first to match lowercased columns_to_drop
                data.columns = [c.lower() for c in data.columns]
                columns_to_drop_lower = [c.lower() for c in columns_to_drop]
                for col in columns_to_drop_lower:
                    if col in data.columns:
                        data.drop(columns=[col], inplace=True)
                data["ticker"] = symbol
                data.drop(columns=["ticker"], inplace=True, errors="ignore")
                out_path = os.path.join(output_dir, symbol + suffix_map[interval_period])
                data.to_csv(out_path, index=False)
                done += 1
                log_queue.put(f"Saved {out_path}  ({len(data)} rows)")
            except Exception as e:
                log_queue.put(f"Error downloading {symbol}: {e}")
                done += 1

    log_queue.put(f"__DONE__{done}/{total}")
