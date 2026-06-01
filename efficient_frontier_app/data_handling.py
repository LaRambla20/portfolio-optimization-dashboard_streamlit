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


def synthesize_total_return(price_index, etf, eurusd=None, periods_per_year=252):
    """Reconstruct a total-return EUR level series from a *price-return* index.

    Price-return indices (most Yahoo ``^`` tickers, all Stooq indices) exclude
    reinvested dividends, so splicing one in front of an *accumulating* ETF would
    bias every long-run return/risk figure low. This calibrates the missing yield
    against the ETF itself over their overlap window, then grosses up the older
    index history and chains it onto the real ETF series.

    All inputs are price *level* Series indexed by ``DatetimeIndex``:
      - ``price_index`` : price-return index in its native currency (e.g. ^GSPC, USD).
      - ``etf``         : accumulating ETF (a total-return instrument) in EUR (e.g. VWCE.MI).
      - ``eurusd``      : USD-per-EUR FX level Series (e.g. EURUSD=X). If ``None`` the
                          index is assumed already in EUR. Converting *before*
                          differencing is essential — otherwise ``q_hat`` silently
                          absorbs FX drift, which is not a dividend and not constant.
      - ``periods_per_year`` : annualisation basis (252 daily / 52 weekly / 12 monthly).

    Calibration is geometric and exact for chaining. With CAGRs ``g`` over the
    overlap, the annual gross-up factor is ``f = (1+g_etf)/(1+g_index)`` and the
    reported yield ``q_hat = f - 1`` (this also absorbs TER / tracking / FX residual
    — intentional, so the synthetic tail meets the real ETF seamlessly at the join).
    Each older period is multiplied by ``f**(1/periods_per_year)``.

    Returns a dict:
      - ``q_hat``       : calibrated annual gross-up (synthetic net yield).
      - ``series``      : spliced EUR total-return level Series; equals ``etf`` from
                          the ETF's first date onward, grossed-up index before it.
      - ``join_date``   : first ETF date (boundary between synthetic and real).
      - ``overlap_start`` / ``overlap_end`` / ``n_overlap`` : calibration window.
      - ``synthetic_start`` / ``synthetic_end`` : span reconstructed from the index
                          (``None`` if the index adds no history before the ETF).
    """
    price_index = price_index.dropna().sort_index()
    etf = etf.dropna().sort_index()

    if eurusd is not None:
        fx = eurusd.dropna().sort_index().reindex(price_index.index).ffill().bfill()
        index_eur = (price_index / fx).dropna()
    else:
        index_eur = price_index

    overlap = index_eur.index.intersection(etf.index)
    if len(overlap) < 2:
        raise ValueError("price_index and etf must overlap on at least 2 dates to calibrate.")
    overlap = overlap.sort_values()
    o_start, o_end = overlap[0], overlap[-1]
    years = (o_end - o_start).days / 365.25
    if years <= 0:
        raise ValueError("Overlap window spans no time.")

    g_index = (index_eur.loc[o_end] / index_eur.loc[o_start]) ** (1.0 / years) - 1.0
    g_etf   = (etf.loc[o_end]       / etf.loc[o_start])       ** (1.0 / years) - 1.0
    f = (1.0 + g_etf) / (1.0 + g_index)          # annual gross-up factor
    q_hat = f - 1.0
    per_period_gross = f ** (1.0 / periods_per_year)

    join_date = etf.index.min()
    idx_pre = index_eur[index_eur.index <= join_date]
    if len(idx_pre) >= 2:
        # Chain grossed-up index *returns*, then scale the level to meet the ETF
        # at join_date (continuous splice — never concatenate raw price levels).
        gross = (1.0 + idx_pre.pct_change()) * per_period_gross
        cum = gross.fillna(1.0).cumprod()
        level_pre = cum / cum.iloc[-1] * etf.loc[join_date]
        level_pre = level_pre[level_pre.index < join_date]
        synthetic_start = level_pre.index.min()
        synthetic_end = level_pre.index.max()
        spliced = pd.concat([level_pre, etf])
    else:
        synthetic_start = synthetic_end = None
        spliced = etf.copy()

    spliced = spliced[~spliced.index.duplicated(keep="last")].sort_index()

    return {
        "q_hat": q_hat,
        "series": spliced,
        "join_date": join_date,
        "overlap_start": o_start,
        "overlap_end": o_end,
        "n_overlap": len(overlap),
        "synthetic_start": synthetic_start,
        "synthetic_end": synthetic_end,
    }


def build_reconstructed_frame(index_prices, etf_prices, fx_prices, periods_per_year):
    """Splice a price-return index in front of an accumulating ETF and tag the join.

    Wraps :func:`synthesize_total_return` and shapes the result into the on-disk CSV
    schema: ``date, adj close, synthetic, recon_yield``. ``synthetic`` is ``True`` for
    rows reconstructed from the index (before the ETF's first date), ``False`` for the
    real ETF rows. ``recon_yield`` carries the calibrated annual gross-up ``q_hat`` on
    synthetic rows and is blank on real rows (a single scalar, repeated only so the
    value travels inside the CSV — the app's other readers slice the first two columns
    or use ``usecols`` and ignore both extra columns).

    Returns ``(frame, meta)`` where ``meta`` is the dict from ``synthesize_total_return``.
    """
    meta = synthesize_total_return(index_prices, etf_prices, fx_prices, periods_per_year)
    series = meta["series"]
    join_date = meta["join_date"]
    q_hat = meta["q_hat"]

    frame = pd.DataFrame({"date": series.index, "adj close": series.values})
    is_synth = frame["date"] < join_date
    frame["synthetic"] = is_synth
    frame["recon_yield"] = np.where(is_synth, q_hat, np.nan)
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    return frame, meta


@st.cache_data
def read_synthetic_info(tickers, folder_path, filename_suffix, filter_date_string):
    """Per-ticker reconstruction metadata for the §1 data panel.

    Returns ``{ticker: {join_date, n_synthetic, q_hat}}`` only for tickers whose CSV
    carries a ``synthetic`` column with at least one reconstructed row inside the
    filter window. Tickers without the column (ordinary downloads) are omitted.
    """
    info = {}
    for ticker in tickers:
        fp = os.path.join(folder_path, ticker + filename_suffix)
        if not os.path.exists(fp):
            continue
        header = pd.read_csv(fp, nrows=0)
        if "synthetic" not in header.columns:
            continue
        cols = ["date", "synthetic"] + (["recon_yield"] if "recon_yield" in header.columns else [])
        df = pd.read_csv(fp, usecols=cols)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= filter_date_string]
        df["synthetic"] = df["synthetic"].astype(str).str.lower().isin(["true", "1", "1.0"])
        synth = df[df["synthetic"]]
        if synth.empty:
            continue
        q_hat = None
        if "recon_yield" in synth.columns:
            vals = pd.to_numeric(synth["recon_yield"], errors="coerce").dropna()
            if not vals.empty:
                q_hat = float(vals.iloc[0])
        info[ticker] = {
            "join_date": synth["date"].max(),   # last synthetic row = boundary with real data
            "n_synthetic": int(len(synth)),
            "q_hat": q_hat,
        }
    return info


@st.cache_data
def read_currency_info(tickers, folder_path, filename_suffix):
    """Resolve each ticker's stored currency for the §1 currency check.

    Prefers the ``currency`` column written at download time (offline, instant). Falls
    back to a network sniff (``detect_currency``) for legacy CSVs that predate the
    column. Returns ``{ticker: currency_or_None}``; ``None`` means undeterminable and
    the caller should stay silent rather than warn.
    """
    yf = None
    info = {}
    for ticker in tickers:
        fp = os.path.join(folder_path, ticker + filename_suffix)
        if not os.path.exists(fp):
            continue
        currency = None
        header = pd.read_csv(fp, nrows=0)
        if "currency" in header.columns:
            vals = pd.read_csv(fp, usecols=["currency"])["currency"].dropna()
            vals = vals[vals.astype(str).str.strip() != ""]
            if not vals.empty:
                currency = str(vals.iloc[0]).strip()
        if not currency:  # legacy file (no column) or blank → sniff once
            if yf is None:
                import yfinance as yf
            currency = detect_currency(ticker, yf)
        info[ticker] = currency
    return info


# Columns whose values are per-share prices and must be scaled by FX when converting
# currency. Volume / stock splits are counts/ratios and must be left untouched.
_PRICE_COLUMNS = ("open", "high", "low", "close", "adj close", "dividends", "capital gains")


def detect_currency(symbol, yf):
    """Best-effort native trading currency for a Yahoo symbol (e.g. 'USD', 'EUR', 'GBp').

    Tries the lightweight ``fast_info`` first, then ``info``. Returns ``None`` if it
    cannot be determined (network/symbol issues) — callers must treat that as unknown.
    """
    try:
        tk = yf.Ticker(symbol)
        cur = None
        try:
            cur = tk.fast_info.get("currency")
        except Exception:
            cur = None
        if not cur:
            cur = (tk.info or {}).get("currency")
        return cur or None
    except Exception:
        return None


def fetch_eur_multiplier(currency, yf_interval, end_date, yf):
    """Series mapping date → (EUR per 1 unit of ``currency``), for FX conversion.

    Uses Yahoo's ``{CCY}EUR=X`` pair (units-of-EUR per 1 CCY). Handles the GBp/GBX
    pence quote (London) by using GBP and dividing by 100. Returns ``None`` when the
    currency is already EUR, unknown, or the FX series can't be fetched.
    """
    if not currency:
        return None
    cur = currency.strip()
    if cur.upper() == "EUR":
        return None

    divisor = 1.0
    base = cur
    if cur in ("GBp", "GBX", "ZAc", "ILA"):   # minor (pence-like) units → major / 100
        base = {"GBp": "GBP", "GBX": "GBP", "ZAc": "ZAR", "ILA": "ILS"}[cur]
        divisor = 100.0

    try:
        fx = yf.Ticker(f"{base}EUR=X").history(
            start="1900-01-01", end=end_date, interval=yf_interval, auto_adjust=False,
        )
        if fx.empty:
            return None
        col = "Adj Close" if "Adj Close" in fx.columns else "Close"
        s = fx[col].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return (s / divisor).dropna()
    except Exception:
        return None


def apply_eur_conversion(data, eur_multiplier):
    """Scale a downloaded price frame into EUR in place-safe fashion.

    ``data`` has a string ``date`` column plus lowercased Yahoo columns; only
    :data:`_PRICE_COLUMNS` present are multiplied. ``eur_multiplier`` is a datetime-
    indexed Series (EUR per 1 unit); it is aligned to each row's date and forward/back
    filled (FX trades a slightly different calendar than equities). Returns a new frame.
    """
    out = data.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out["date"]).dt.normalize().values)
    mult = eur_multiplier.sort_index()
    mult = mult[~mult.index.duplicated(keep="last")]
    # Union FX dates with equity dates, forward-fill across the combined timeline so
    # each equity row takes the most recent prior FX rate (nearest-prior), back-fill any
    # leading gap, then select the equity dates.
    aligned = mult.reindex(mult.index.union(idx)).ffill().bfill().reindex(idx)
    aligned = pd.Series(aligned.values, index=out.index)
    for col in _PRICE_COLUMNS:
        if col in out.columns:
            out[col] = out[col] * aligned.values
    return out


def run_download(tickers, intervals, columns_to_drop, output_dir, log_queue,
                 convert_to_eur=True):
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

                # Record the stored currency so §1 can flag non-EUR series offline.
                # When converting, scale every price column into EUR and store 'EUR'.
                native = detect_currency(symbol, yf)
                stored_currency = native
                if convert_to_eur:
                    mult = fetch_eur_multiplier(native, yf_interval, today, yf)
                    if mult is not None:
                        data = apply_eur_conversion(data, mult)
                        stored_currency = "EUR"
                        log_queue.put(f"Converted {symbol} {native}→EUR via {native}EUR=X")
                    elif native and native.upper() != "EUR":
                        log_queue.put(
                            f"⚠️  Could not fetch {native}EUR=X for {symbol}; saved in {native} (unconverted)."
                        )
                data["currency"] = stored_currency if stored_currency else ""

                out_path = os.path.join(output_dir, symbol + suffix_map[interval_period])
                data.to_csv(out_path, index=False)
                done += 1
                cur_note = f", currency={stored_currency}" if stored_currency else ""
                log_queue.put(f"Saved {out_path}  ({len(data)} rows{cur_note})")
            except Exception as e:
                log_queue.put(f"Error downloading {symbol}: {e}")
                done += 1

    log_queue.put(f"__DONE__{done}/{total}")


def run_total_return_reconstruction(jobs, intervals, output_dir, log_queue):
    """Download price-return indices + accumulating ETFs and splice them into
    total-return histories, saving each as ``{etf}_EXT_data_{period}.csv``.

    ``jobs`` is a list of dicts ``{"index": str, "etf": str, "fx": str|""}``. For each
    job × interval it downloads the index (price-return), the ETF (total-return) and,
    if given, the FX pair (USD-per-EUR), then calls :func:`build_reconstructed_frame`.
    Mirrors :func:`run_download`'s log/progress contract (``__DONE__done/total``).
    """
    import yfinance as yf

    interval_map = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}
    suffix_map   = {"daily": "_data_daily.csv", "weekly": "_data_weekly.csv", "monthly": "_data_monthly.csv"}
    ppy_map      = {"daily": 252, "weekly": 52, "monthly": 12}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_queue.put(f"Created folder: {output_dir}")

    today = datetime.date.today().strftime("%Y-%m-%d")
    total = len(jobs) * len(intervals)
    done = 0

    def fetch(symbol, yf_interval):
        h = yf.Ticker(symbol).history(
            start="1900-01-01", end=today, interval=yf_interval, auto_adjust=False,
        )
        if h.empty:
            raise ValueError(f"no data returned for {symbol}")
        col = "Adj Close" if "Adj Close" in h.columns else "Close"
        s = h[col].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.dropna()

    for interval_period in intervals:
        yf_interval = interval_map[interval_period]
        ppy = ppy_map[interval_period]
        for job in jobs:
            index_sym = job["index"]
            etf_sym = job["etf"]
            fx_sym = (job.get("fx") or "").strip()
            tag = f"{index_sym}->{etf_sym}_EXT ({interval_period})"
            log_queue.put(f"Reconstructing {tag}...")
            try:
                index_prices = fetch(index_sym, yf_interval)
                etf_prices = fetch(etf_sym, yf_interval)
                fx_prices = fetch(fx_sym, yf_interval) if fx_sym else None

                frame, meta = build_reconstructed_frame(index_prices, etf_prices, fx_prices, ppy)
                out_path = os.path.join(output_dir, f"{etf_sym}_EXT" + suffix_map[interval_period])
                frame.to_csv(out_path, index=False)

                n_synth = int(frame["synthetic"].sum())
                start_str = frame["date"].iloc[0]
                join_str = meta["join_date"].strftime("%Y-%m-%d")
                q = meta["q_hat"]
                log_queue.put(
                    f"✅ Saved {out_path}  ({len(frame)} rows, {n_synth} synthetic before {join_str}, "
                    f"q={q*100:.2f}%/yr, history from {start_str})"
                )
                # A recovered yield outside a plausible dividend band almost always means
                # the index doesn't track the same underlying as the ETF (or a data quirk:
                # splits, wrong currency, distributing-vs-accumulating share class).
                if not (0.0 <= q <= 0.06):
                    log_queue.put(
                        f"⚠️  {etf_sym}_EXT: q={q*100:.2f}%/yr is outside the usual 0–4% dividend "
                        f"band — check that '{index_sym}' tracks the same underlying as '{etf_sym}' "
                        f"and that the FX ticker is correct. The series was still saved."
                    )
            except Exception as e:
                log_queue.put(f"❌ Error reconstructing {tag}: {e}")
            done += 1

    log_queue.put(f"__DONE__{done}/{total}")
