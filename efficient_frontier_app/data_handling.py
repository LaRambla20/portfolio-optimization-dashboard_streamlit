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
    if len(sub) == 0:
        return np.nan
    start_price = sub.iloc[0]
    end_price = sub.iloc[-1]
    return end_price / start_price - 1


def evaluate_CAGR(prices, start_date, end_date):
    sub = prices.loc[start_date:end_date]
    if len(sub) == 0:
        return np.nan
    start_price = sub.iloc[0]
    end_price = sub.iloc[-1]
    n_years = (sub.index[-1] - sub.index[0]).days / 365.25
    if n_years <= 0:  # single-row / zero-span slice → CAGR undefined
        return np.nan
    return (end_price / start_price) ** (1 / n_years) - 1


def evaluate_return_metrics(returns, start_date, end_date):
    sub = returns.loc[start_date:end_date]
    std_dev = sub.std()
    mean = sub.mean()
    return std_dev, mean


@st.cache_data
def detect_stock_splits(tickers, folder_path, filename_suffix, filter_date_string):
    """Ground-truth stock splits, read from yfinance's ``stock splits`` column.

    A non-zero entry there is a split yfinance *recorded* on that ex-date, with the
    exact ratio (e.g. ``2.0`` = 2-for-1, ``0.1`` = 1-for-10 reverse) — interval- and
    asset-agnostic because it is recorded data, not inferred from a price jump. Note
    that ``adj close`` is already split-adjusted, so these splits do **not** create a
    jump in the series the app analyses; they are reported as *informational*.

    Returns ``{ticker: [(date_str, ratio), ...]}`` only for tickers whose CSV carries
    a ``stock splits`` column with at least one split inside the filter window. Files
    without the column (legacy downloads, ``_EXT`` reconstructions) are omitted.
    """
    splits = {}
    for ticker in tickers:
        fp = os.path.join(folder_path, ticker + filename_suffix)
        if not os.path.exists(fp):
            continue
        header = pd.read_csv(fp, nrows=0)
        if "stock splits" not in header.columns:
            continue
        df = pd.read_csv(fp, usecols=["date", "stock splits"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= filter_date_string]
        ratios = pd.to_numeric(df["stock splits"], errors="coerce").fillna(0.0)
        hits = df[ratios != 0.0]
        if not hits.empty:
            splits[ticker] = [
                (row["date"].strftime("%Y-%m-%d"), float(ratios.loc[idx]))
                for idx, row in hits.iterrows()
            ]
    return splits


@st.cache_data
def check_price_anomalies(tickers, folder_path, filename_suffix, filter_date_string,
                          z_threshold=8.0, min_abs_move=0.45):
    """Flag step-to-step moves that are extreme *relative to the series' own history*.

    Replaces a fixed 60% threshold, which can't serve every asset × interval at once
    (a monthly bar compounds ~21 daily moves, and BTC swings far more than an equity
    ETF — so 60% is routine for crypto monthly yet near-impossible for an equity daily).

    We use a robust z-score on the simple returns ``r_t``: centre on the median and
    scale by ``1.4826·MAD`` (≈ a fat-tail-resistant σ), then flag bars where
    ``|z| > z_threshold`` **and** ``|r_t| > min_abs_move``. The MAD scale auto-adapts
    per asset and per interval, so BTC's genuine ±65% months pass while a data glitch
    (a fat-finger tick, a currency mix-up, or a split that wasn't adjusted in this
    series) stands out. The absolute floor stops a tiny but statistically-extreme wobble
    in an ultra-calm series from tripping the flag.

    Each hit is cross-referenced against :func:`detect_stock_splits`: ``matches_split``
    marks a flagged move that lands on a recorded split ex-date (→ very likely just an
    unadjusted split, harmless). Returns
    ``{ticker: [(date_str, prev_price, new_price, pct, z, matches_split), ...]}``.
    """
    split_dates = {
        t: {d for d, _ in events}
        for t, events in detect_stock_splits(
            tickers, folder_path, filename_suffix, filter_date_string).items()
    }
    anomalies = {}
    for ticker in tickers:
        fp = os.path.join(folder_path, ticker + filename_suffix)
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, usecols=["date", "adj close"])
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= filter_date_string].sort_values("date").reset_index(drop=True)
        r = df["adj close"].pct_change()
        rv = r.dropna()
        if len(rv) < 10:  # too few points for a stable median/MAD
            continue
        med = rv.median()
        mad = (rv - med).abs().median()
        robust_sigma = 1.4826 * mad
        if robust_sigma <= 0:  # degenerate (mostly identical prices) → fall back to std
            robust_sigma = rv.std()
        if not robust_sigma or robust_sigma <= 0:
            continue
        z = (r - med) / robust_sigma
        ticker_splits = split_dates.get(ticker, set())
        hits = []
        for idx in range(1, len(df)):
            rr = r.iloc[idx]
            if pd.isna(rr):
                continue
            if abs(z.iloc[idx]) > z_threshold and abs(rr) > min_abs_move:
                dstr = df.loc[idx, "date"].strftime("%Y-%m-%d")
                hits.append((dstr, df.loc[idx - 1, "adj close"], df.loc[idx, "adj close"],
                             float(rr), float(z.iloc[idx]), dstr in ticker_splits))
        if hits:
            anomalies[ticker] = hits
    return anomalies


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
def load_asset_series(folder_path, ticker, filename_suffix, filter_date_string):
    """Date-indexed ``adj close`` Series for one asset over its *own* full history.

    Unlike :func:`build_merged_dataframe` (which inner-joins all tickers to a common
    window), this loads a single ticker's complete history up to ``filter_date_string``,
    so per-asset analytics (§2) reflect that asset's actual lifespan rather than the
    shortest-asset overlap. Reads only the first two columns via ``usecols`` so the
    extra ``synthetic`` / ``recon_yield`` / ``currency`` columns on ``_EXT`` /
    downloaded files are ignored.
    """
    fp = os.path.join(folder_path, ticker + filename_suffix)
    df = pd.read_csv(fp, usecols=["date", "adj close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= filter_date_string].sort_values("date")
    return df.set_index("date")["adj close"]


@st.cache_data
def compute_portfolio_returns_simple(merged_df):
    table = merged_df.set_index("date")
    portfolio_returns_simple = table.pct_change().dropna()
    portfolio_mean_returns = portfolio_returns_simple.mean()
    portfolio_cov_matrix = portfolio_returns_simple.cov()
    return portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix


@st.cache_data
def compute_rolling_returns(merged_df, window_periods):
    """Rolling **cumulative** simple return for each ticker over the window:
    ``price[t] / price[t-window_periods] - 1`` — the total return across the whole
    `window_periods`-row window, **not** annualised (so longer windows yield larger figures and
    are not comparable across window lengths; see §2 CAGR for a per-year rate)."""
    price_df = merged_df.set_index("date")
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


def build_reconstructed_frame(index_prices, etf_prices, fx_prices, periods_per_year, currency=None):
    """Splice a price-return index in front of an accumulating ETF and tag the join.

    Wraps :func:`synthesize_total_return` and shapes the result into the on-disk CSV
    schema: ``date, adj close, synthetic, recon_yield`` (plus ``currency`` when given). ``synthetic`` is ``True`` for
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
    # Appended *after* the tag columns so `date`/`adj close` stay at positions 0-1 for
    # build_merged_dataframe's positional `iloc[:, :2]` read.
    if currency is not None:
        frame["currency"] = currency
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


def convert_series_to_eur(prices, eur_multiplier):
    """Scale a date-indexed price Series into EUR.

    Thin wrapper over :func:`apply_eur_conversion` so the reconstruction path reuses the
    same nearest-prior FX alignment as the download path instead of re-deriving it.
    """
    tmp = pd.DataFrame({"date": prices.index.strftime("%Y-%m-%d"), "adj close": prices.values})
    out = apply_eur_conversion(tmp, eur_multiplier)
    return pd.Series(out["adj close"].values, index=prices.index)


# ─────────────────────────────────────────────────────────
# GUIDED EXTEND WIZARD — validation helpers
# ─────────────────────────────────────────────────────────

# Period name -> yfinance interval code. Shared by the download, reconstruction and wizard paths.
YF_INTERVALS = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}

# Index families whose Yahoo search results are useless, mapped to tickers verified to have
# real downloadable history. These are *hints only*: every entry is run through probe_ticker
# like a search result, so a stale one gets marked unusable rather than silently misleading.
# (yf.Search for "FTSE All-World" returns only quote-only *.FGI tickers with zero history.)
# Yahoo Search is poor at commodities: "Physical Gold" returns nothing, and "Gold index"
# returns S&P/TSX *Global Gold* — gold **miners**, a different asset from bullion.
CURATED_INDEX_HINTS = {
    "gold":                [("GC=F", "Gold futures (COMEX) — spot-linked, long history"),
                            ("IAU", "iShares Gold Trust (proxy)"),
                            ("GLD", "SPDR Gold Shares (proxy)")],
    "silver":              [("SI=F", "Silver futures (COMEX)")],
    "ftse all-world":      [("VT", "Vanguard Total World — FTSE Global All Cap (proxy)")],
    "ftse global all cap": [("VT", "Vanguard Total World — FTSE Global All Cap")],
    "s&p 500":             [("^GSPC", "S&P 500 price index")],
    "msci world":          [("^990100-USD-STRD", "MSCI World price index (developed only)")],
    "msci acwi":           [("ACWI", "iShares MSCI ACWI (proxy)")],
    "all country world":   [("ACWI", "iShares MSCI ACWI (proxy)")],
}

# Wrapper words that appear in an ETF's longName but never in an index's name.
_NAME_NOISE = (
    "ucits", "etf", "etc", "acc", "accumulating", "accumulation", "dist", "distributing",
    "inc", "income", "fund", "index", "shares", "core", "plc", "usd", "eur", "gbp", "chf",
    "hedged", "class", "a", "b", "c", "1c", "1d",
)


def suggest_fx_ticker(etf_currency, index_currency):
    """Yahoo FX pair that converts the index into the ETF's currency, or None.

    ``synthesize_total_return`` computes ``index / fx``, and Yahoo quotes ``ABCDEF=X`` as
    DEF per 1 ABC. To turn an index priced in ``index_currency`` into ``etf_currency`` the
    divisor must be *index per etf*, i.e. ``{etf}{index}=X`` — a USD index with an EUR ETF
    needs ``EURUSD=X`` (USD per EUR). Reversing this silently inverts the FX leg.

    Returns None when either currency is unknown or the two already match (no conversion).
    """
    if not etf_currency or not index_currency:
        return None
    a, b = etf_currency.strip().upper(), index_currency.strip().upper()
    if not a or not b or a == b:
        return None
    return f"{a}{b}=X"


def index_query_from_name(long_name):
    """Reduce an ETF's longName to a searchable index name.

    'Vanguard FTSE All-World UCITS ETF USD Accumulation' -> 'FTSE All-World'. Strips the
    issuer prefix and the wrapper words that only ever appear on funds, never on indices.
    """
    if not long_name:
        return ""
    cleaned = str(long_name).replace("(", " ").replace(")", " ")
    words = [w for w in cleaned.split() if w.strip()]
    # Drop a leading issuer token (Vanguard, iShares, Amundi, ...) when more words follow.
    known_issuers = {"vanguard", "ishares", "ishs", "amundi", "xtrackers", "spdr",
                     "invesco", "lyxor", "hsbc", "ubs", "wisdomtree", "img", "imgp"}
    if words and words[0].lower() in known_issuers:
        words = words[1:]
    kept = [w for w in words if w.lower().strip(".,-") not in _NAME_NOISE]
    return " ".join(kept).strip()


# Which gap to expect depends on whether the index leg *omits* income the fund collects.
#   price_index : a price-return index vs an income-collecting fund -> gap is the dividend yield.
#   same_income : the index already carries the income, or the asset pays none (gold, commodities)
#                 -> the only gap left is fees/tracking, so it sits near zero.
# Calibrated on real pairings: correct ones land within ±0.25% in `same_income`, while the
# tightest *wrong* one (world equity spliced onto gold) is +0.92%.
Q_BANDS = {
    "price_index": (0.00, 0.06),
    "same_income": (-0.005, 0.005),
}


def q_hat_verdict(q_hat, regime="price_index"):
    """(ok, message) for a recovered yield, judged against the band for ``regime``.

    ``q_hat`` is the annual return the index is *missing* versus the fund. Which value is
    plausible depends entirely on the pairing (see :data:`Q_BANDS`): a price-return equity
    index should be short by a dividend yield (~1.5-4%/yr), whereas an index that already
    includes income — or an asset with none, like gold — should differ by fees alone.
    """
    if q_hat is None or (isinstance(q_hat, float) and np.isnan(q_hat)):
        return False, "Could not recover a yield from the overlap."
    low, high = Q_BANDS.get(regime, Q_BANDS["price_index"])
    pct = q_hat * 100.0

    if regime == "same_income":
        if q_hat < low:
            return False, (
                f"Recovered yield is **{pct:.2f}%/yr**. These two should differ by fees alone, so "
                f"anything below {low * 100:.1f}%/yr is more than a fee normally explains — either "
                f"they don't track the same thing, or this fund charges more than {abs(low) * 100:.1f}%/yr "
                f"(check its ongoing charge; a high fee alone can produce this)."
            )
        if q_hat > high:
            return False, (
                f"Recovered yield is **{pct:.2f}%/yr** — the fund grew *faster* than the index with "
                f"no income to explain it. Nothing legitimate does that; these two are not tracking "
                f"the same thing."
            )
        return True, (
            f"Recovered yield is **{pct:.2f}%/yr** — near zero, which is what fees alone look like. "
            f"Consistent with both sides carrying the same income."
        )

    if q_hat < low:
        return False, (
            f"Recovered yield is **{pct:.2f}%/yr** — negative, so the index *outperformed* the "
            f"fund. That is not a dividend; these two are not tracking the same market. (If this "
            f"asset pays no income at all, switch the setting above.)"
        )
    if q_hat > high:
        return False, (
            f"Recovered yield is **{pct:.2f}%/yr** — far above any real dividend yield. The gap "
            f"between two different markets is being mistaken for dividends."
        )
    return True, f"Recovered yield is **{pct:.2f}%/yr** — a plausible dividend yield."


def default_q_regime(index_probe):
    """Pick the expected-gap regime from what the *index leg* actually is.

    The fund side can't help: an accumulating equity fund and a gold ETC both report zero
    dividends, so nothing in the data separates them — which is why the wizard offers an
    override. The index leg is decidable:

    - it pays dividends            -> its Adj Close already carries income  -> ``same_income``
    - FUTURE / CURRENCY            -> no income exists to omit              -> ``same_income``
    - INDEX with no dividends      -> a price index, missing the fund's yield -> ``price_index``
    - anything else (a fund used as the index) -> Adj Close is total-return -> ``same_income``
    """
    if not index_probe or not index_probe.get("ok"):
        return "price_index"
    if index_probe.get("pays_dividends"):
        return "same_income"
    if (index_probe.get("quote_type") or "").upper() == "INDEX":
        return "price_index"
    return "same_income"


@st.cache_data(show_spinner=False)
def fetch_history_frame(symbol, yf_interval):
    """Full tz-naive history frame for a Yahoo symbol; empty frame when unavailable.

    The single cached download behind both :func:`fetch_price_series` and the dividend check in
    :func:`probe_ticker` — the price column and the ``Dividends`` column come from the same
    fetch instead of two.
    """
    import yfinance as yf

    try:
        hist = yf.Ticker(symbol).history(
            start="1900-01-01", end=datetime.date.today().strftime("%Y-%m-%d"),
            interval=yf_interval, auto_adjust=False,
        )
        if hist.empty:
            return pd.DataFrame()
        hist = hist.copy()
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        return hist
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_price_series(symbol, yf_interval):
    """Adjusted-close history for a Yahoo symbol as a tz-naive Series; empty when unavailable.

    Adj Close (not Close) matters: it is dividend-adjusted, so a *distributing* fund reads as
    total return just like an accumulating one, and the calibration doesn't care which it is.
    """
    hist = fetch_history_frame(symbol, yf_interval)
    if hist.empty:
        return pd.Series(dtype="float64")
    col = "Adj Close" if "Adj Close" in hist.columns else "Close"
    return hist[col].dropna()


@st.cache_data(show_spinner=False)
def probe_ticker(symbol, yf_interval):
    """Does this Yahoo symbol actually have downloadable history? Used by every wizard step.

    Returns ``{ok, error, start, end, n_rows, currency, long_name}``. ``ok`` is False when the
    fetch comes back empty — the case that makes a ticker look valid while being unusable
    (a live quote with no history, e.g. AW01.FGI).
    """
    import yfinance as yf

    out = {"ok": False, "error": None, "start": None, "end": None, "n_rows": 0,
           "currency": None, "long_name": None, "quote_type": None, "pays_dividends": False}
    sym = (symbol or "").strip()
    if not sym:
        out["error"] = "No ticker given."
        return out

    hist = fetch_history_frame(sym, yf_interval)
    series = fetch_price_series(sym, yf_interval)
    if series.empty:
        out["error"] = ("Yahoo returned no history for this ticker — it may be a quote-only "
                        "symbol, delisted, or misspelled.")
        return out
    if len(series) < 2:
        out["error"] = "Only a single price point is available — not enough to splice."
        return out

    out.update(ok=True, start=series.index.min(), end=series.index.max(), n_rows=int(len(series)))
    # Distributing funds pay out (non-zero Dividends); accumulating funds and assets with no
    # income at all both report zero. That tells us whether this series *carries* income, which
    # is what default_q_regime needs — it does not tell equities apart from gold.
    if "Dividends" in hist.columns:
        out["pays_dividends"] = bool(pd.to_numeric(hist["Dividends"], errors="coerce").fillna(0).sum() > 0)
    try:
        info = yf.Ticker(sym).info or {}
        out["long_name"] = info.get("longName")
        out["quote_type"] = info.get("quoteType")
    except Exception:
        pass
    out["currency"] = detect_currency(sym, yf)
    return out


# Quote types worth offering as an index leg. EQUITY is deliberately absent — it floods the
# list with individual stocks. FUTURE is what makes GC=F (gold) reachable at all.
SEARCH_QUOTE_TYPES = ("INDEX", "ETF", "FUTURE", "CURRENCY", "MUTUALFUND")


def _search_queries(cleaned):
    """Query variants, most specific first.

    Order is load-bearing for commodities: "Gold index" finds gold *miners*, while plain "Gold"
    finds the bullion future — so the full name is tried before progressively shorter ones.
    """
    if not cleaned:
        return []
    queries, words = [f"{cleaned} index", cleaned], cleaned.split()
    for drop in range(1, len(words)):
        queries.append(" ".join(words[drop:]))
    return queries


@st.cache_data(show_spinner=False)
def preview_candidate_fit(index_symbol, etf_symbol, yf_interval, periods_per_year):
    """How well would this index extend this fund? Everything step 2 needs to rank a candidate.

    Returns ``{ok, reason, extends, n_overlap, extra_years, q_hat, fx_symbol, regime,
    verdict_ok, verdict}``.

    The FX pair is **auto-derived** from the two currencies (:func:`suggest_fx_ticker`), because
    the user doesn't choose one until the next step — and q_hat without it is badly misleading:
    a JPY-quoted gold ETF reads -1.75%/yr unconverted versus +0.14%/yr converted, i.e. the
    difference between "rejected" and "fine". Cheap despite the extra work: both price series are
    already cached by the probing pass, so this is pure pandas.
    """
    out = {"ok": False, "reason": None, "extends": False, "n_overlap": 0, "extra_years": 0.0,
           "q_hat": None, "fx_symbol": None, "regime": None, "verdict_ok": False, "verdict": ""}

    idx_probe = probe_ticker(index_symbol, yf_interval)
    etf_probe = probe_ticker(etf_symbol, yf_interval)
    if not idx_probe["ok"]:
        out["reason"] = idx_probe["error"] or "no usable history"
        return out
    if not etf_probe["ok"]:
        out["reason"] = "the fund itself has no usable history"
        return out

    out["extends"] = bool(idx_probe["start"] < etf_probe["start"])
    out["extra_years"] = max(0.0, (etf_probe["start"] - idx_probe["start"]).days / 365.25)

    idx_s = fetch_price_series(index_symbol, yf_interval)
    etf_s = fetch_price_series(etf_symbol, yf_interval)
    out["n_overlap"] = int(len(idx_s.index.intersection(etf_s.index)))
    if out["n_overlap"] < 2:
        out["reason"] = "shares fewer than 2 dates with the fund"
        return out
    if not out["extends"]:
        out["reason"] = f"starts {idx_probe['start'].date()} — no earlier history to add"
        return out

    out["fx_symbol"] = suggest_fx_ticker(etf_probe["currency"], idx_probe["currency"])
    fx_s = fetch_price_series(out["fx_symbol"], yf_interval) if out["fx_symbol"] else None
    if out["fx_symbol"] and (fx_s is None or fx_s.empty):
        out["fx_symbol"] = None   # rate unavailable; fall back to no conversion
        fx_s = None
    try:
        out["q_hat"] = float(synthesize_total_return(idx_s, etf_s, fx_s, periods_per_year)["q_hat"])
    except Exception as e:
        out["reason"] = f"couldn't calibrate ({type(e).__name__})"
        return out

    out["regime"] = default_q_regime(idx_probe)
    out["verdict_ok"], out["verdict"] = q_hat_verdict(out["q_hat"], out["regime"])
    out["ok"] = True
    return out


@st.cache_data(show_spinner=False)
def suggest_index_candidates(etf_long_name, yf_interval, max_results=6, etf_symbol=""):
    """Candidate index tickers for an ETF, each verified against real history.

    Merges :data:`CURATED_INDEX_HINTS` with a ``yf.Search`` on the cleaned fund name, then runs
    **every** candidate through :func:`probe_ticker`. Search alone is not enough: its top hits
    for the common European ETFs are quote-only tickers with no history.

    Returns a list of ``{symbol, name, source, probe}`` with usable candidates first.
    """
    import yfinance as yf

    query = index_query_from_name(etf_long_name)
    seen, candidates = set(), []

    lowered = (etf_long_name or "").lower()
    for family, entries in CURATED_INDEX_HINTS.items():
        if family in lowered:
            for sym, label in entries:
                if sym.upper() not in seen:
                    seen.add(sym.upper())
                    candidates.append({"symbol": sym, "name": label, "source": "curated"})

    if etf_symbol:
        seen.add(etf_symbol.strip().upper())  # searching a fund's name returns the fund itself

    for attempt in _search_queries(query):
        found = False
        try:
            for quote in (yf.Search(attempt, max_results=max_results).quotes or []):
                sym = (quote.get("symbol") or "").strip()
                if not sym or sym.upper() in seen:
                    continue
                if quote.get("quoteType") not in SEARCH_QUOTE_TYPES:
                    continue
                seen.add(sym.upper())
                candidates.append({
                    "symbol": sym,
                    "name": quote.get("shortname") or quote.get("longname") or "",
                    "source": "search",
                })
                found = True
        except Exception:
            pass  # search is a convenience; manual entry always remains available
        if found:
            break  # a more specific query matched; don't dilute it with vaguer ones

    del candidates[8:]  # probing costs a request each — keep the step responsive
    for cand in candidates:
        cand["probe"] = probe_ticker(cand["symbol"], yf_interval)
    candidates.sort(key=lambda c: (not c["probe"]["ok"], c["source"] != "curated"))
    return candidates


def run_download(tickers, intervals, columns_to_drop, output_dir, log_queue,
                 convert_to_eur=True):
    import yfinance as yf

    interval_map = YF_INTERVALS
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
                log_queue.put(f"✅ Saved {out_path}  ({len(data)} rows{cur_note})")
            except Exception as e:
                log_queue.put(f"❌ Error downloading {symbol}: {e}")
                done += 1

    log_queue.put(f"__DONE__{done}/{total}")


def run_total_return_reconstruction(jobs, intervals, output_dir, log_queue, convert_to_eur=True):
    """Download price-return indices + accumulating ETFs and splice them into
    total-return histories, saving each as ``{etf}_EXT_data_{period}.csv``.

    ``jobs`` is a list of dicts ``{"index": str, "etf": str, "fx": str|"", "regime": str|None}``.
    ``regime`` selects which q_hat band the log warning judges against (see :data:`Q_BANDS`);
    omit it and one is detected, so the warning can never contradict the wizard's verdict on
    the same pairing. For each
    job × interval it downloads the index (price-return), the ETF (total-return) and,
    if given, the FX pair (USD-per-EUR), then calls :func:`build_reconstructed_frame`.
    Mirrors :func:`run_download`'s log/progress contract (``__DONE__done/total``).

    ``convert_to_eur`` scales the **ETF leg** into EUR before splicing and records the
    stored currency, mirroring :func:`run_download`. Only the ETF leg is converted: the
    older synthetic rows are already put into the ETF's currency by the job's ``fx``
    ticker inside :func:`synthesize_total_return`, so converting the spliced output
    instead would double-convert them.
    """
    import yfinance as yf

    interval_map = YF_INTERVALS
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

                # Same currency contract as run_download: convert the ETF leg to EUR and
                # record what actually got stored, so §1 can check it offline.
                native = detect_currency(etf_sym, yf)
                stored_currency = native
                if convert_to_eur:
                    mult = fetch_eur_multiplier(native, yf_interval, today, yf)
                    if mult is not None:
                        etf_prices = convert_series_to_eur(etf_prices, mult)
                        stored_currency = "EUR"
                        log_queue.put(f"Converted {etf_sym} {native}→EUR via {native}EUR=X")
                    elif native and native.upper() != "EUR":
                        log_queue.put(
                            f"⚠️  Could not fetch {native}EUR=X for {etf_sym}; "
                            f"saved in {native} (unconverted)."
                        )

                frame, meta = build_reconstructed_frame(
                    index_prices, etf_prices, fx_prices, ppy,
                    currency=stored_currency if stored_currency else "",
                )
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
                # Judge the recovered yield with the *same* helper and band the wizard used, so
                # the log can't call a pairing implausible right after the wizard passed it. A
                # gold future vs a gold ETC legitimately lands near -0.2%/yr, which the old fixed
                # 0-6% band flagged as an error.
                regime = job.get("regime") or default_q_regime(probe_ticker(index_sym, yf_interval))
                verdict_ok, verdict_msg = q_hat_verdict(q, regime)
                if not verdict_ok:
                    log_queue.put(
                        f"⚠️  {etf_sym}_EXT: {verdict_msg} Check that '{index_sym}' tracks the same "
                        f"underlying as '{etf_sym}' and that the FX ticker is correct. "
                        f"The series was still saved."
                    )
            except Exception as e:
                log_queue.put(f"❌ Error reconstructing {tag}: {e}")
            done += 1

    log_queue.put(f"__DONE__{done}/{total}")
