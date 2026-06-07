"""
UI rendering functions for Efficient Frontier dashboard.
All section renderers receive data as explicit parameters (C-like style).
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy import stats
from portfolio_calculations import (
    portfolio_annualised_performance,
    portfolio_downside_deviation,
    random_portfolios,
    random_portfolios_sortino,
    max_sharpe_ratio,
    max_sortino_ratio,
    minimize_volatility,
    maximize_return,
    efficient_return,
    efficient_volatility,
    efficient_frontier_fn,
    make_allocation_df,
    cvar,
    max_drawdown,
    buy_and_hold_value_series,
    rebalanced_value_series,
    underwater_episodes,
    deepest_drawdown_episode,
    longest_underwater_episode,
    downside_deviation_series,
    real_deflator,
)
from data_handling import (
    evaluate_simple_return,
    evaluate_CAGR,
    evaluate_return_metrics,
    load_asset_series,
)
from descriptions import render_section_help


# ─────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────

def collect_portfolio_info(name, weights, mean_returns, cov_matrix, risk_free_rate,
                           portfolio_returns_simple, tickers, annualisation_factor,
                           ax, marker, color, size):
    std_dev, ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor)
    sharpe  = (ret - risk_free_rate) / std_dev
    dd      = portfolio_downside_deviation(weights, portfolio_returns_simple, annualisation_factor, risk_free_rate)
    sortino = (ret - risk_free_rate) / dd if dd > 0 else np.nan
    alloc   = make_allocation_df(weights, tickers)
    port_returns = portfolio_returns_simple.dot(weights)
    mdd     = max_drawdown(port_returns)
    ax.scatter(std_dev, ret, marker=marker, color=color, s=size, label=name, zorder=5)
    return {"name": name, "std_dev": std_dev, "ret": ret, "sharpe": sharpe,
            "sortino": sortino, "alloc": alloc,
            "max_dd": mdd, "port_returns": port_returns}


def collect_portfolio_info_mtc(name, index, results, weights_list, mean_returns,
                               cov_matrix, risk_free_rate, portfolio_returns_simple,
                               tickers, annualisation_factor, ax, marker, color, size):
    std_dev = results[0, index]
    ret     = results[1, index]
    sharpe  = results[2, index]
    weights = weights_list[index]
    dd      = portfolio_downside_deviation(weights, portfolio_returns_simple, annualisation_factor, risk_free_rate)
    sortino = (ret - risk_free_rate) / dd if dd > 0 else np.nan
    alloc   = make_allocation_df(weights, tickers)
    port_returns = portfolio_returns_simple.dot(weights)
    mdd     = max_drawdown(port_returns)
    ax.scatter(std_dev, ret, marker=marker, color=color, s=size, label=name, zorder=5)
    return {"name": name, "std_dev": std_dev, "ret": ret, "sharpe": sharpe,
            "sortino": sortino, "alloc": alloc,
            "max_dd": mdd, "port_returns": port_returns}


def display_portfolio_cards(portfolios, alpha):
    for p in portfolios:
        with st.expander(f"{p['name']}", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric(
                "Average annual return",
                f"{p['ret']:.2%}",
                help="Average return per year, estimated from per-period returns — the same figure "
                     "used on the efficient-frontier chart and to compare portfolios.",
            )
            c2.metric("Ann. Volatility", f"{p['std_dev']:.2%}")
            c3.metric("Sharpe Ratio",  f"{p['sharpe']:.3f}")
            sortino_val = p.get("sortino")
            c4.metric("Sortino Ratio", f"{sortino_val:.3f}" if sortino_val is not None and not np.isnan(sortino_val) else "n/a")
            c5.metric(
                "Max Drawdown",
                f"{p.get('max_dd', 0):.2%}",
                help="Worst peak-to-trough fall over the full history shown — a cumulative figure, "
                     "not annualised. Assumes constant weights (per-period rebalancing), not buy-and-hold. "
                     "For a less-frequent cadence see §6.",
            )
            # CVaR computed once here, at the user's chosen confidence level, from the portfolio's
            # own return series (single CVaR definition / sign across the whole app: positive = loss).
            port_returns = p.get("port_returns")
            cvar_val = cvar(port_returns, 1 - alpha) if port_returns is not None else 0.0
            c6.metric(
                f"CVaR ({alpha:.0%})",
                f"{cvar_val:.2%}",
                help="Expected shortfall: the average loss on the worst (1−confidence) slice of "
                     "individual periods (days, weeks, or months). A per-period figure — not annualised — so it "
                     "sits on a shorter horizon than the annual return and volatility above. "
                     "Computed on a constant-weight (per-period-rebalanced) portfolio.",
            )
            alloc_series = p["alloc"].iloc[0]
            left, right  = st.columns([1, 1])
            with left:
                st.dataframe(p["alloc"], width="stretch")
            with right:
                nonzero = alloc_series[alloc_series > 0]
                if nonzero.empty:
                    st.caption("All weights are zero — no chart to display.")
                else:
                    fig_pie, ax_pie = plt.subplots(figsize=(3.5, 3.5))
                    ax_pie.pie(nonzero.values, labels=nonzero.index, autopct="%1.1f%%",
                               startangle=90, pctdistance=0.82)
                    ax_pie.axis("equal")
                    st.pyplot(fig_pie)
                    plt.close(fig_pie)


# ─────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA (split detection + anomaly warnings + data availability)
# ─────────────────────────────────────────────────────────────────

def render_load_etf_data(tickers, split_events, anomaly_warnings, data_availability,
                         synthetic_info=None, currency_info=None):
    st.header("1. Load Data")
    render_section_help(
        "This section loads your price data and checks it is usable — identifying recorded "
        "stock splits, flagging statistically anomalous jumps, and showing how much history "
        "all your assets share.",
        ["price_spikes", "data_window"],
    )

    # Recorded stock splits (ground truth from yfinance's 'stock splits' column).
    # Adj Close is already split-adjusted, so these are informational, not problems.
    if split_events:
        tickers_split = ", ".join(f"**{t}**" for t in split_events)
        st.info(f"📐 Recorded stock splits found in: {tickers_split} "
                "(already reflected in Adj Close — informational)")
        with st.expander("Show splits", expanded=False):
            for ticker, events in split_events.items():
                st.markdown(f"**{ticker}**")
                rows = [
                    {"Date": d, "Split ratio": f"{ratio:g}-for-1" if ratio >= 1
                     else f"1-for-{(1 / ratio):g} (reverse)"}
                    for d, ratio in events
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Statistically anomalous moves (robust MAD z-score), i.e. likely data glitches —
    # or a split not reflected in this series.
    if anomaly_warnings:
        tickers_flagged = ", ".join(f"**{t}**" for t in anomaly_warnings)
        st.warning(f"⚠️ Statistically anomalous price moves detected in: {tickers_flagged}")
        with st.expander("Show details", expanded=False):
            for ticker, events in anomaly_warnings.items():
                st.markdown(f"**{ticker}**")
                rows = [
                    {"Date": d, "Prev. Price": f"{prev:.4f}", "New Price": f"{new:.4f}",
                     "Change": f"{pct:+.1%}", "Robust z": f"{z:+.1f}",
                     "On split date?": "yes" if on_split else "—"}
                    for d, prev, new, pct, z, on_split in events
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(
                "These moves are extreme relative to each asset's own return history "
                "(|robust z| > 8), so normal high-volatility swings (e.g. crypto) are not "
                "flagged. A move marked **On split date? = yes** is almost certainly just a "
                "split not yet reflected in this series; otherwise check for a data glitch "
                "(bad tick, currency mix-up) before trusting the numbers built on top of it."
            )

    common_start = data_availability["common_start"]
    common_end   = data_availability["common_end"]
    total_years  = data_availability["total_years"]
    ticker_starts = data_availability["ticker_starts"]
    ticker_ends   = data_availability["ticker_ends"]

    years_int  = int(total_years)
    months_int = int(round((total_years - years_int) * 12))
    if months_int == 12:
        years_int += 1
        months_int = 0
    label = f"{years_int}y {months_int}m of common data"
    if len(tickers) == 1:
        label += " (single asset)"

    MAX_YEARS = 30
    pct       = min(total_years / MAX_YEARS, 1.0)

    BAR_X    = 40
    BAR_W    = 520
    BAR_H    = 26
    TITLE_Y  = 18
    NEEDLE_TIP_Y = TITLE_Y + 10
    BAR_Y    = NEEDLE_TIP_Y + 14
    TICK_Y1  = BAR_Y + BAR_H + 4
    TICK_Y2  = TICK_Y1 + 6
    TICK_LBL = TICK_Y2 + 11
    DETAIL_Y = TICK_LBL + 20
    LINE_H   = 15

    needle_x   = BAR_X + pct * BAR_W
    label_x    = min(max(needle_x, BAR_X + 60), BAR_X + BAR_W - 60)
    needle_top = BAR_Y - 14

    arrow   = "→"
    binding = "◄"

    binding_ticker = tickers[ticker_starts.index(common_start)]
    detail_lines   = []
    for t, s, e in zip(tickers, ticker_starts, ticker_ends):
        weight   = "bold" if t == binding_ticker else "normal"
        suffix   = f"  {binding} binding" if t == binding_ticker else ""
        line_y   = DETAIL_Y + len(detail_lines) * LINE_H
        s_str    = s.strftime("%b %Y")
        e_str    = e.strftime("%b %Y")
        detail_lines.append(
            f'<text x="{BAR_X}" y="{line_y}" '
            f'font-family="sans-serif" font-size="11" fill="#555" font-weight="{weight}">'
            f'{t}: {s_str} {arrow} {e_str}{suffix}</text>'
        )
    common_line_y  = DETAIL_Y + len(detail_lines) * LINE_H
    cs_str         = common_start.strftime("%b %Y")
    ce_str         = common_end.strftime("%b %Y")
    detail_lines.append(
        f'<text x="{BAR_X}" y="{common_line_y}" '
        f'font-family="sans-serif" font-size="11" fill="#888">'
        f'Common window: {cs_str} {arrow} {ce_str}</text>'
    )

    SVG_H = common_line_y + 16
    SVG_W = 860
    BAR_W_PX = SVG_W - 2 * BAR_X
    needle_x_px = BAR_X + pct * BAR_W_PX
    label_x_px  = min(max(needle_x_px, BAR_X + 80), BAR_X + BAR_W_PX - 80)

    ticks_svg = "".join(
        f'<line x1="{BAR_X + i/30*BAR_W_PX:.1f}" y1="{TICK_Y1}" '
        f'x2="{BAR_X + i/30*BAR_W_PX:.1f}" y2="{TICK_Y2}" stroke="#888" stroke-width="1"/>'
        f'<text x="{BAR_X + i/30*BAR_W_PX:.1f}" y="{TICK_LBL}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="10" fill="#666">{i}y</text>'
        for i in range(0, 31, 5)
    )

    svg = (
        f'<svg width="{SVG_W}" height="{SVG_H}" '
        f'viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg">'
        f'<defs>'
        f'<linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        f'<stop offset="0%"   stop-color="#d73027"/>'
        f'<stop offset="33%"  stop-color="#f46d43"/>'
        f'<stop offset="55%"  stop-color="#fee08b"/>'
        f'<stop offset="75%"  stop-color="#a6d96a"/>'
        f'<stop offset="100%" stop-color="#1a9850"/>'
        f'</linearGradient>'
        f'</defs>'
        f'<text x="{BAR_X}" y="{TITLE_Y}" font-family="sans-serif" font-size="12" '
        f'fill="#555" font-weight="600">Common data available for portfolio analysis</text>'
        f'<rect x="{BAR_X}" y="{BAR_Y}" width="{BAR_W_PX}" height="{BAR_H}" '
        f'rx="5" ry="5" fill="url(#gaugeGrad)" opacity="0.92"/>'
        f'{ticks_svg}'
        f'<polygon points="{needle_x_px:.1f},{BAR_Y} {needle_x_px-7:.1f},{needle_top} {needle_x_px+7:.1f},{needle_top}" fill="#222"/>'
        f'<text x="{label_x_px:.1f}" y="{needle_top - 3}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="11" font-weight="700" fill="#222">{label}</text>'
        + "".join(detail_lines)
        + "</svg>"
    )

    # Render the gauge SVG inline. We can't use st.html here: its DOMPurify pass (with
    # Streamlit's default config) strips <svg>, leaving nothing. st.markdown with
    # unsafe_allow_html keeps the SVG, renders inline (no iframe), and avoids the
    # deprecated components.html. The SVG carries its own width/height so it self-sizes.
    html_gauge = f'<div style="padding:4px 0 0 0;">{svg}</div>'
    st.markdown(html_gauge, unsafe_allow_html=True)

    if data_availability.get("mixed_calendar"):
        seven = ", ".join(f"**{t}**" for t in data_availability["seven_day_tickers"])
        st.caption(
            f"📅 Calendar note: {seven} trade 7 days a week while the other assets trade ~5. To compare "
            "them, the analysis keeps only the dates all assets share, so weekend moves of the 7-day "
            "asset(s) are absorbed into the next shared day. Their daily volatility and correlations are "
            "therefore slightly approximate — switch the data period to weekly or monthly for cleaner "
            "mixed-calendar figures."
        )

    if currency_info:
        non_eur = [
            (t, str(c)) for t in tickers
            if (c := currency_info.get(t)) and str(c).strip().upper() != "EUR"
        ]
        if non_eur:
            listed = ", ".join(f"**{t}** ({c})" for t, c in non_eur)
            st.warning(
                f"💱 Currency mismatch: {listed} {'is' if len(non_eur) == 1 else 'are'} not in EUR. "
                "The app does **no** FX conversion outside the downloader, so these prices are treated "
                "as if they were EUR — making their returns, volatility and (especially) correlations "
                "wrong for a EUR investor, which biases the frontier and VaR. Re-download with "
                "**Convert prices to EUR** ticked, or use the EUR-listed share class (e.g. `.MI` / `.DE` / `.AS`)."
            )

    if synthetic_info:
        parts = []
        for t in tickers:
            meta = synthetic_info.get(t)
            if not meta:
                continue
            join_str = meta["join_date"].strftime("%b %Y")
            q = meta.get("q_hat")
            q_str = f", q≈{q*100:.1f}%/yr" if q is not None else ""
            parts.append(
                f"**{t}**: {meta['n_synthetic']} rows before {join_str} are reconstructed "
                f"total-return{q_str}"
            )
        if parts:
            st.caption(
                "🧬 Synthetic history: " + "; ".join(parts) + ". "
                "These pre-ETF rows are a price-return index grossed up by a dividend yield "
                "calibrated against the ETF — an *estimate*, not measured data."
            )

    st.divider()


# ─────────────────────────────────────────────────────────────────
# SECTION 2 — PER-ASSET ANALYTICS
# ─────────────────────────────────────────────────────────────────

def _full_history_label(start, end):
    """Honest label for a full-history window, e.g. 'Full (8y 4m)'."""
    days = max((end - start).days, 0)
    yrs = days / 365.25
    y = int(yrs)
    m = int(round((yrs - y) * 12))
    if m == 12:
        y += 1
        m = 0
    return f"Full ({y}y {m}m)"


def _lookback_year_windows(ref_end, data_start):
    """(label, start) pairs for the 1/3/5-year look-backs that *fully fit* within the asset's
    history, plus a true full-history row. Windows longer than the asset's life are dropped rather
    than silently truncated to the data start (which would mislabel, e.g., a 3y span as '5y')."""
    wins = [(f"{y}y", ref_end - pd.DateOffset(years=y)) for y in (1, 3, 5)]
    wins = [(lbl, sd) for lbl, sd in wins if pd.notna(sd) and sd >= data_start]
    wins.append((_full_history_label(data_start, ref_end), data_start))
    return wins


def render_per_etf_analytics(tickers, folder_path, filename_suffix, filter_date_string,
                              annualisation_factor, real_terms=False, annual_inflation=0.0):
    st.header("2. Per-Asset Analytics")
    st.caption("Each asset is shown over its **own full history** (up to the date filter), which may "
               "start earlier than the portfolio's shared window in §1 — so these figures can differ "
               "from the portfolio sections, by design.")
    if real_terms:
        st.caption(f"📉 **Real terms** — prices deflated by an assumed {annual_inflation:.1%}/yr "
                   "inflation, so returns/CAGR are in today's purchasing power.")
    render_section_help(
        "This section looks at each asset on its own: how much it returned, how much it swung, "
        "and its worst fall — so you understand each holding before combining them.",
        ["simple_return", "calendar_year_return", "cagr", "avg_annual_return",
         "annual_volatility", "max_drawdown", "cumulative_return", "lookback_annual_metrics"]
        + (["real_returns"] if real_terms else []),
    )

    real_sfx = " (real)" if real_terms else ""

    for ticker in tickers:
        # Load this asset's *own* full history (not the inner-joined common window), so "full history"
        # CAGR/returns/drawdown reflect the asset's actual lifespan. usecols ignores _EXT/currency extras.
        series = load_asset_series(folder_path, ticker, filename_suffix, filter_date_string)
        subdf = series.to_frame("adj close")
        # Deflate the price level once; every figure below (simple/calendar returns, CAGR, annualised
        # metrics, cumulative chart, look-back tables) then reads real prices with no further changes.
        if real_terms:
            subdf["adj close"] = subdf["adj close"] / real_deflator(subdf.index, annual_inflation)

        with st.expander(f"{ticker}", expanded=False):
            data_start = subdf.index[0]
            end_date = subdf.index[-1]

            # Look-back windows that fully fit this asset's history (+ a true full-history row);
            # windows longer than its life are omitted, not silently truncated and mislabelled.
            simple_windows = []
            ytd_idx = subdf[subdf.index.year == (end_date.year - 1)].index
            if len(ytd_idx):
                simple_windows.append(("YTD", ytd_idx.max()))
            for lbl, off in (("1mo", pd.DateOffset(months=1)),
                             ("3mo", pd.DateOffset(months=3)),
                             ("6mo", pd.DateOffset(months=6))):
                sd = end_date - off
                if sd >= data_start:
                    simple_windows.append((lbl, sd))
            simple_windows += _lookback_year_windows(end_date, data_start)

            simple_rows = []
            for label, sd in simple_windows:
                sr = evaluate_simple_return(subdf["adj close"], sd, end_date)
                simple_rows.append({"Period": label, "From": sd.date(), "To": end_date.date(),
                                     "Simple Return": f"{sr:.2%}"})
            st.subheader("Simple Return (single-period)")
            st.dataframe(pd.DataFrame(simple_rows), width="stretch", hide_index=True)
            st.caption("Windows longer than this asset's available history are omitted; "
                       "**Full** covers its entire history.")

            yearly_rows = []
            for yr_offset in range(3):
                considered_year = end_date.year - (1 + yr_offset)
                year_idx = subdf[subdf.index.year == considered_year].index
                if len(year_idx) == 0:  # asset wasn't trading that calendar year
                    continue
                ed = year_idx.max()
                sd = ed - pd.DateOffset(years=1)
                sr = evaluate_simple_return(subdf["adj close"], sd, ed)
                yearly_rows.append({"Year": considered_year, "From": sd.date(), "To": ed.date(),
                                     "Simple Return": f"{sr:.2%}"})
            if yearly_rows:
                st.dataframe(pd.DataFrame(yearly_rows), width="stretch", hide_index=True)

            cagr_windows = _lookback_year_windows(end_date, data_start)
            cagr_rows = []
            for label, sd in cagr_windows:
                cagr = evaluate_CAGR(subdf["adj close"], sd, end_date)
                cagr_rows.append({"Period": label, "From": sd.date(), "To": end_date.date(),
                                   "CAGR": f"{cagr:.2%}"})
            st.subheader("CAGR (Compound Annual Growth Rate)")
            st.dataframe(pd.DataFrame(cagr_rows), width="stretch", hide_index=True)

            st.subheader("Annualised Metrics (simple returns, full history)")
            simple_ret = subdf["adj close"].pct_change().dropna()
            ann_mean = simple_ret.mean() * annualisation_factor
            ann_std = simple_ret.std() * np.sqrt(annualisation_factor)
            mdd = max_drawdown(simple_ret)
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Average annual return",
                f"{ann_mean:.2%}",
                help="Average return per year, estimated from per-period returns — the same figure "
                     "used on the efficient-frontier chart and to compare portfolios.",
            )
            col2.metric("Annualised Volatility", f"{ann_std:.2%}")
            col3.metric("Max Drawdown", f"{mdd:.2%}")

            # Plot cumulative returns (always simple returns for interpretability)
            cumulative_ret = (1 + simple_ret).cumprod() - 1
            fig_cum, ax_cum = plt.subplots(figsize=(10, 4))
            ax_cum.plot(cumulative_ret.index, cumulative_ret.values, lw=1)
            ax_cum.set_title(f"{ticker} — Cumulative Returns{real_sfx}")
            ax_cum.set_xlabel("Date")
            ax_cum.set_ylabel(f"Cumulative Return{real_sfx}")
            ax_cum.grid(True)
            st.pyplot(fig_cum)
            plt.close(fig_cum)

            st.subheader("Annualised Metrics by Look-back Period (as of today)")
            rolling_rows = []
            for label, sd in cagr_windows:
                simple_ret = subdf["adj close"].loc[sd:end_date].pct_change().dropna()
                ann_ret = simple_ret.mean() * annualisation_factor
                vol = simple_ret.std() * np.sqrt(annualisation_factor)
                rolling_rows.append({
                    "Period": label, "From": sd.date(), "To": end_date.date(),
                    "Ann. Avg Return": f"{ann_ret:.2%}", "Ann. Volatility": f"{vol:.2%}",
                })
            st.dataframe(pd.DataFrame(rolling_rows), width="stretch", hide_index=True)

            st.subheader("Annualised Metrics by Look-back Period (as of last full year-end)")
            ly_idx = subdf[subdf.index.year == (end_date.year - 1)].index
            if len(ly_idx) == 0:
                st.caption("No prior full calendar year of data for this asset — this view is omitted.")
            else:
                end_date_ly = ly_idx.max()
                rolling_ly_rows = []
                for label, sd in _lookback_year_windows(end_date_ly, data_start):
                    simple_ret = subdf["adj close"].loc[sd:end_date_ly].pct_change().dropna()
                    ann_ret = simple_ret.mean() * annualisation_factor
                    vol = simple_ret.std() * np.sqrt(annualisation_factor)
                    rolling_ly_rows.append({
                        "Period": label, "From": sd.date(), "To": end_date_ly.date(),
                        "Ann. Avg Return": f"{ann_ret:.2%}", "Ann. Volatility": f"{vol:.2%}",
                    })
                st.dataframe(pd.DataFrame(rolling_ly_rows), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────
# SECTION 3 — PER-ASSET PRICES
# ─────────────────────────────────────────────────────────────────

def render_etf_prices(merged_df, tickers):
    st.header("3. Per-Asset Prices")
    render_section_help(
        "This section plots prices so you can compare how your assets moved — both in real "
        "terms and rebased to a common starting point.",
        ["normalized_prices"],
    )

    st.subheader("Raw Closing Prices")
    with st.expander("Show raw price table"):
        st.dataframe(merged_df, width="stretch", hide_index=True)

    fig_raw, ax_raw = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        ax_raw.plot(merged_df["date"], merged_df[ticker], lw=1, label=ticker)
    ax_raw.set_title("Assets Closing Prices Over Time")
    ax_raw.set_xlabel("Date")
    ax_raw.set_ylabel("Closing Prices [EUR]")
    ax_raw.legend(title="Asset")
    ax_raw.grid(True)
    st.pyplot(fig_raw)
    plt.close(fig_raw)

    norm_df = merged_df.copy()
    for ticker in tickers:
        norm_df[ticker] = (norm_df[ticker] / norm_df[ticker].iloc[0]) * 1000

    st.subheader("Normalized Closing Prices (base = 1000)")
    with st.expander("Show normalized price table"):
        st.dataframe(norm_df, width="stretch", hide_index=True)

    fig_norm, ax_norm = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        ax_norm.plot(norm_df["date"], norm_df[ticker], lw=1, label=ticker)
    ax_norm.set_title("Assets Normalized Closing Prices Over Time")
    ax_norm.set_xlabel("Date")
    ax_norm.set_ylabel("Normalized Closing Prices [EUR]")
    ax_norm.legend(title="Asset")
    ax_norm.grid(True)
    st.pyplot(fig_norm)
    plt.close(fig_norm)

    return norm_df


def render_rolling_returns(rolling_returns, tickers, rolling_window_years):
    st.header("4. Per-Asset Rolling Returns")
    render_section_help(
        "This section shows returns over long rolling windows, so you can see what an investor "
        "would have earned holding for 1, 5 or 10 years starting at any point in time.",
        ["rolling_returns_asset"],
    )
    if rolling_returns.shape[0] <= rolling_window_years:
        st.warning(f"Insufficient data for {rolling_window_years}-year rolling window.")
        return

    # Individual assets (the portfolio rolling-returns chart lives in §6, Input Portfolio Analysis).
    st.subheader(f"Individual Assets — {rolling_window_years}Y Rolling Returns")
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        ax.plot(rolling_returns["date"], rolling_returns[ticker], lw=1, label=ticker)
    ax.set_title(f"Rolling {rolling_window_years}-Year Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(title="Asset")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    st.divider()



# ─────────────────────────────────────────────────────────────────
# SECTION 4 — RETURNS & STATISTICS
# ─────────────────────────────────────────────────────────────────

def render_returns_statistics(portfolio_returns_simple, portfolio_mean_returns,
                               portfolio_cov_matrix, tickers, annualisation_factor,
                               risk_free_rate):
    st.header("5. Per-Asset Returns & Statistics")
    render_section_help(
        "This section measures how rewarding and how risky your assets have been, and crucially "
        "how they move together — the raw material for diversification.",
        ["return_stats", "sortino", "covariance", "correlation"],
    )

    min_return = portfolio_returns_simple.min()
    max_return = portfolio_returns_simple.max()
    mean_returns = portfolio_returns_simple.mean()
    median_returns = portfolio_returns_simple.median()
    returns_std_dev = portfolio_returns_simple.std()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.subheader("Min Return (%)")
        st.dataframe((min_return * 100).round(2).to_frame("Min"), width="stretch")
    with col2:
        st.subheader("Max Return (%)")
        st.dataframe((max_return * 100).round(2).to_frame("Max"), width="stretch")
    with col3:
        st.subheader("Mean Return (%)")
        st.dataframe((mean_returns * 100).round(2).to_frame("Mean"), width="stretch")
    with col4:
        st.subheader("Median Return (%)")
        st.dataframe((median_returns * 100).round(2).to_frame("Median"), width="stretch")
    with col5:
        st.subheader("Std Dev (%)")
        st.dataframe((returns_std_dev * 100).round(2).to_frame("Std Dev"), width="stretch")
    st.caption("All figures are per-period **simple** returns (the data period selected in the sidebar). "
               "The gap between **mean** and **median** is a skew read: mean above median = a right tail "
               "(occasional big gains); below = a left tail (occasional big losses). Not annualised — a "
               "median has no linear `×N` annualisation (quantiles aren't additive); for an annual-scale "
               "central tendency use the geometric **CAGR** in §2.")

    single_asset_sortino = {}
    for t in tickers:
        w_single = np.zeros(len(tickers))
        w_single[tickers.index(t)] = 1.0
        dd = portfolio_downside_deviation(w_single, portfolio_returns_simple, annualisation_factor, risk_free_rate)
        ret_single = portfolio_mean_returns[t] * annualisation_factor
        single_asset_sortino[t] = (ret_single - risk_free_rate) / dd if dd > 0 else np.nan
    st.subheader("Per-Asset Sortino Ratio (annualised)")
    st.dataframe(
        pd.DataFrame.from_dict(single_asset_sortino, orient="index", columns=["Sortino"]).round(3),
        width="stretch"
    )

    st.subheader("Covariance Matrix (used for optimization — simple returns)")
    st.dataframe(portfolio_cov_matrix, width="stretch")

    st.subheader("Correlation Matrix (used for optimization — simple returns)")
    st.dataframe(portfolio_returns_simple.corr(), width="stretch")
    st.caption("The correlation heatmap is shown in §6, Input Portfolio Analysis.")

    fig_ret, ax_ret = plt.subplots(figsize=(12, 5))
    for c in portfolio_returns_simple.columns.values:
        ax_ret.plot(portfolio_returns_simple.index, portfolio_returns_simple[c] * 100,
                    lw=1, alpha=0.8, label=c)
    ax_ret.legend(loc="upper right", fontsize=10)
    ax_ret.set_ylabel("Return [%]")
    ax_ret.set_title("Asset Returns")
    st.pyplot(fig_ret)
    plt.close(fig_ret)


# ─────────────────────────────────────────────────────────────────
# SECTION 5 — INPUT PORTFOLIO ANALYSIS (BUY-AND-HOLD)
# ─────────────────────────────────────────────────────────────────

def _fmt_period(days, ongoing):
    """Human-readable underwater duration: '482 days (~1.3y)', or '≥ N days (ongoing)'."""
    years = days / 365.25
    span = f"{days} days (~{years:.1f}y)" if years >= 1 else f"{days} days"
    return f"≥ {span}, ongoing" if ongoing else span


def render_input_portfolio_analysis(merged_df, portfolio_returns_simple, tickers,
                                    my_portfolio_allocation, annualisation_factor,
                                    risk_free_rate, alpha, window_periods, rolling_window_years,
                                    rebalance_every_periods=None, rebalance_label="never rebalanced (buy-and-hold)",
                                    real_terms=False, annual_inflation=0.0):
    st.header("6. Input Portfolio Analysis")
    st.caption(f"⚖️ Rebalancing: **{rebalance_label}** (set in the sidebar).")
    if real_terms:
        st.caption(f"📉 **Real terms** — value deflated by an assumed {annual_inflation:.1%}/yr "
                   "inflation; the risk-free rate is deflated to match.")
    render_section_help(
        "This section analyses your actual allocation, held at the rebalancing cadence you chose in "
        "the sidebar, covering its growth, its worst falls and recovery times, and its tail risk.",
        ["rebalancing", "buy_and_hold", "cumulative_return", "underwater_curve", "max_underwater_period",
         "max_drawdown", "cagr", "avg_annual_return", "annual_volatility", "sharpe", "sortino",
         "rolling_returns_portfolio", "correlation",
         "zscore", "var_parametric", "var_historical", "cvar", "return_distribution"]
        + (["real_returns"] if real_terms else []),
    )

    real_sfx = " (real)" if real_terms else ""

    weights = np.array([my_portfolio_allocation[t] for t in tickers], dtype=np.float64)

    # ── User-defined allocation (the input being analysed) ───────────────────────────────
    st.subheader("Your Allocation")
    alloc_series = pd.Series(weights, index=tickers)
    nonzero = alloc_series[alloc_series > 0]
    pie_col, tbl_col = st.columns([1, 1])
    with pie_col:
        if nonzero.empty:
            st.caption("All weights are zero — nothing to chart.")
        else:
            fig_alloc, ax_alloc = plt.subplots(figsize=(4, 4))
            ax_alloc.pie(nonzero.values, labels=nonzero.index, autopct="%1.1f%%",
                         startangle=90, pctdistance=0.82,
                         wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            ax_alloc.axis("equal")
            st.pyplot(fig_alloc)
            plt.close(fig_alloc)
    with tbl_col:
        alloc_df = (alloc_series * 100).round(2).to_frame("Weight (%)")
        st.dataframe(alloc_df, width="stretch")
        st.caption("Weights are normalised from the market values you entered in the sidebar.")

    # ── Rebalanced value series (the single basis for every figure below) ────────────────
    value = rebalanced_value_series(merged_df, tickers, weights, rebalance_every_periods)
    # Deflate the value level once: bh_ret and every downstream figure (CAGR, vol, Sharpe, Sortino,
    # drawdown, underwater episodes, rolling returns, and the tail-risk subsection) then read real
    # terms with no further changes. The risk-free rate is deflated to match so the risk premium —
    # hence Sharpe/Sortino — stays meaningful (and ~invariant) in real terms.
    if real_terms:
        value = value / real_deflator(value.index, annual_inflation)
        rf_used = (1.0 + risk_free_rate) / (1.0 + annual_inflation) - 1.0
    else:
        rf_used = risk_free_rate
    bh_ret = value.pct_change().dropna()
    if len(bh_ret) < 2:
        st.warning("Not enough overlapping history to analyse the portfolio.")
        return

    N = annualisation_factor
    ann_ret = bh_ret.mean() * N
    ann_vol = bh_ret.std() * np.sqrt(N)
    sharpe = (ann_ret - rf_used) / ann_vol if ann_vol > 0 else np.nan
    dd_dev = downside_deviation_series(bh_ret, N, rf_used)
    sortino = (ann_ret - rf_used) / dd_dev if dd_dev > 0 else np.nan

    years = (value.index[-1] - value.index[0]).days / 365.25
    cagr = (value.iloc[-1] / value.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan

    drawdown = value / value.cummax() - 1.0
    mdd = abs(drawdown.min())

    st.info(
        f"📌 This section holds your weights **{rebalance_label}**; between rebalances the mix drifts "
        "with prices. The tail-risk subsection below uses this *same* cadence. The §7/§8 "
        "efficient-frontier cards instead assume per-period rebalancing (the MPT basis), so the "
        "*same* portfolio's return, risk and drawdown can legitimately differ there — unless you "
        "set the cadence to 'Every period'."
        + (f" Figures are in **real terms** (deflated by {annual_inflation:.1%}/yr): CAGR and average "
           "return drop and drawdowns deepen, but volatility is ~unchanged." if real_terms else "")
    )

    # ── Headline metrics (growth & drawdown; tail risk lives in the subsection below) ─────
    a1, a2, a3 = st.columns(3)
    a1.metric(
        "CAGR (compound)", f"{cagr:.2%}" if not np.isnan(cagr) else "n/a",
        help="Geometric compound annual growth rate of the buy-and-hold value: "
             "(V_end / V_start)^(1/years) − 1. The single steady yearly rate that reproduces the "
             "actual end value — what you really earned per year with compounding baked in. "
             "Differs from the arithmetic 'Average annual return' alongside, which is a linear "
             "(non-compounded) mean × N: volatility drag pulls CAGR down, intra-year compounding "
             "pulls it up, so either can be the larger.",
    )
    a2.metric(
        "Average annual return", f"{ann_ret:.2%}",
        help="Mean per-period return of the buy-and-hold portfolio, annualised linearly (mean × N). "
             "An arithmetic/expected figure, distinct from the compounded CAGR shown alongside.",
    )
    a3.metric("Annual volatility", f"{ann_vol:.2%}",
              help="Standard deviation of per-period returns, annualised as σ·√N.")

    b1, b2, b3 = st.columns(3)
    b1.metric("Sharpe ratio", f"{sharpe:.3f}" if not np.isnan(sharpe) else "n/a",
              help="(Average annual return − risk-free rate) ÷ annual volatility.")
    b2.metric("Sortino ratio", f"{sortino:.3f}" if not np.isnan(sortino) else "n/a",
              help="Like Sharpe but divides by downside deviation only — penalises harmful "
                   "volatility, not upside.")
    b3.metric("Max drawdown", f"{mdd:.2%}",
              help="Worst peak-to-trough fall of the buy-and-hold value over the full history.")

    # ── Cumulative returns: portfolio (bold) + per-asset buy-and-hold overlays ───────────
    st.subheader("Cumulative Returns (buy-and-hold)")
    prices = merged_df.set_index("date")[tickers]
    norm = prices / prices.iloc[0]
    if real_terms:  # deflate the per-asset overlays to match the (real) portfolio line
        norm = norm.div(real_deflator(norm.index, annual_inflation), axis=0)
    pct = plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    fig_cum, ax_cum = plt.subplots(figsize=(12, 5))
    for t in tickers:
        ax_cum.plot(norm.index, norm[t] - 1.0, lw=1, alpha=0.45, label=t)
    ax_cum.plot(value.index, value - 1.0, lw=2.2, color="black", label="Portfolio")
    ax_cum.axhline(0, color="#999", lw=0.8)
    ax_cum.set_title(f"Cumulative Return Since Start{real_sfx}")
    ax_cum.set_xlabel("Date")
    ax_cum.set_ylabel(f"Cumulative Return{real_sfx}")
    ax_cum.yaxis.set_major_formatter(pct)
    ax_cum.legend(title="Holding", ncol=2)
    ax_cum.grid(True, alpha=0.3)
    st.pyplot(fig_cum)
    plt.close(fig_cum)

    # ── Underwater curve (annotated) + drawdown/recovery periods ────────────────────────
    st.subheader("Underwater Curve & Drawdown Periods")
    deepest = deepest_drawdown_episode(value)
    longest = longest_underwater_episode(value)
    last_date = value.index[-1]

    fig_uw, ax_uw = plt.subplots(figsize=(12, 5))
    ax_uw.fill_between(drawdown.index, drawdown.values, 0.0, color="#d73027", alpha=0.25)
    ax_uw.plot(drawdown.index, drawdown.values, color="#a50026", lw=1)

    if longest is not None:
        l_ep, l_days, l_ongoing = longest
        l_end = l_ep["recovery_date"] if l_ep["recovery_date"] is not None else last_date
        ax_uw.axvspan(l_ep["peak_date"], l_end, color="#fdae61", alpha=0.20,
                      label="Longest underwater stretch")

    if deepest is not None:
        d_peak = deepest["peak_date"]
        d_trough = deepest["trough_date"]
        d_rec = deepest["recovery_date"]
        d_depth = deepest["trough_val"] / deepest["peak_val"] - 1.0
        ax_uw.axvline(d_peak, color="#1a9850", ls="--", lw=1.2, label="Deepest-DD peak")
        ax_uw.scatter([d_trough], [d_depth], color="#a50026", s=60, zorder=5,
                      label=f"Deepest trough ({d_depth:.1%})")
        if d_rec is not None:
            ax_uw.axvline(d_rec, color="#4575b4", ls="--", lw=1.2, label="Recovered")

    ax_uw.axhline(0, color="#999", lw=0.8)
    ax_uw.set_title(f"Drawdown From Running Peak{real_sfx}")
    ax_uw.set_xlabel("Date")
    ax_uw.set_ylabel("Drawdown")
    ax_uw.yaxis.set_major_formatter(pct)
    ax_uw.legend(loc="lower left", fontsize=8)
    ax_uw.grid(True, alpha=0.3)
    st.pyplot(fig_uw)
    plt.close(fig_uw)

    u1, u2 = st.columns(2)
    if deepest is not None:
        d_peak = deepest["peak_date"]
        d_rec = deepest["recovery_date"]
        d_depth = abs(deepest["trough_val"] / deepest["peak_val"] - 1.0)
        if d_rec is not None:
            deep_days, deep_ongoing = (d_rec - d_peak).days, False
        else:
            deep_days, deep_ongoing = (last_date - d_peak).days, True
        u1.metric(
            "Deepest-drawdown recovery", _fmt_period(deep_days, deep_ongoing),
            help=f"The deepest fall was {d_depth:.1%}, from a peak on {d_peak.date()} to a trough on "
                 f"{deepest['trough_date'].date()}. This is the time from that peak until value "
                 f"{'regained it' if not deep_ongoing else 'first regains it (not yet recovered)'}.",
        )
    else:
        u1.metric("Deepest-drawdown recovery", "n/a", help="Value never fell below a prior peak.")

    if longest is not None:
        l_ep, l_days, l_ongoing = longest
        u2.metric(
            "Longest underwater stretch", _fmt_period(l_days, l_ongoing),
            help=f"The most time spent below a peak before recovering — peak on "
                 f"{l_ep['peak_date'].date()}"
                 + ("" if not l_ongoing else " (still underwater at the end of the data)") + ".",
        )
    else:
        u2.metric("Longest underwater stretch", "n/a", help="Value never fell below a prior peak.")

    # ── Portfolio rolling returns (moved from §4), on the buy-and-hold value ────────────
    st.subheader(f"Portfolio — {rolling_window_years}Y Rolling Returns (Buy-and-Hold)")
    if len(value) > window_periods:
        roll = (value / value.shift(window_periods) - 1.0).dropna()
        fig_rr, ax_rr = plt.subplots(figsize=(12, 5))
        ax_rr.plot(roll.index, roll.values, lw=1.5, color="black", label="Portfolio")
        ax_rr.set_title(f"Portfolio Rolling {rolling_window_years}-Year Returns{real_sfx}")
        ax_rr.set_xlabel("Date")
        ax_rr.set_ylabel(f"Rolling Return{real_sfx}")
        ax_rr.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax_rr.legend()
        ax_rr.grid(True, alpha=0.3)
        st.pyplot(fig_rr)
        plt.close(fig_rr)
    else:
        st.info(f"Insufficient data for a {rolling_window_years}-year rolling window "
                f"({len(value)} periods ≤ {window_periods}).")

    # ── Asset correlation heatmap (moved from §5) ───────────────────────────────────────
    st.subheader("Asset Correlation Heatmap (simple returns)")
    if real_terms:
        st.caption("Unchanged by the real-terms setting — subtracting a constant inflation rate "
                   "leaves correlations (and volatility) unaffected.")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    sns.heatmap(portfolio_returns_simple.corr(), annot=True, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Asset Correlation Matrix (simple returns)")
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    # ── Tail risk & return distribution (merged from the former standalone VaR section) ──
    render_tail_risk(bh_ret, alpha)

    st.divider()


# ─────────────────────────────────────────────────────────────────
# SECTION 6 — MONTE CARLO EFFICIENT FRONTIER (VOL-BASED)
# ─────────────────────────────────────────────────────────────────

def render_monte_carlo(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                        tickers, annualisation_factor, risk_free_rate, num_portfolios, eps,
                        custom_target_ret, custom_target_vol, my_portfolio_allocation, alpha,
                        real_terms=False):
    st.header("7. Monte Carlo Efficient Frontier Portfolio Optimization")
    st.caption("⚖️ Rebalancing: **per period** — the frontier and all return/volatility figures here "
               "assume per-period rebalancing (the basis MPT optimization requires), independent of the "
               "sidebar Rebalancing-frequency setting (which governs §6).")
    if real_terms:
        st.caption("💶 Shown in **nominal** terms despite the real-returns setting: a constant inflation "
                   "rate leaves the efficient-frontier *weights* unchanged (the real risk premium is "
                   "inflation-invariant) — it would only shift the return axis down by the inflation rate.")
    render_section_help(
        "This section randomly simulates thousands of portfolios to map the trade-off between "
        "risk and return, and highlights a few notable ones.",
        ["monte_carlo", "avg_annual_return", "annual_volatility", "sharpe", "sortino",
         "efficient_frontier", "marked_portfolios", "max_drawdown", "cvar"],
    )

    results_mc, weights_mc = random_portfolios(num_portfolios, portfolio_mean_returns,
                                                portfolio_cov_matrix, risk_free_rate, annualisation_factor)
    weights_array_mc = np.array(weights_mc)

    my_portfolio_weights = np.array(list(my_portfolio_allocation.values()), dtype=np.float64)
    if len(my_portfolio_weights) != len(tickers):
        st.error("Portfolio weights count does not match tickers. Check your portfolio definition.")
        return

    fig_mc, ax_mc = plt.subplots(figsize=(10, 7))
    sc = ax_mc.scatter(results_mc[0, :], results_mc[1, :], c=results_mc[2, :],
                       cmap="YlGnBu", marker="o", s=10, alpha=0.3)
    plt.colorbar(sc, ax=ax_mc, label="Sharpe Ratio")

    portfolios_mc = []

    p = collect_portfolio_info("My Portfolio", my_portfolio_weights, portfolio_mean_returns,
                                portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                tickers, annualisation_factor, ax_mc, "P", "y", 500)
    portfolios_mc.append(p)
    my_portfolio_ret_mc = p["ret"]
    my_portfolio_std_mc = p["std_dev"]

    mask = (results_mc[1, :] >= my_portfolio_ret_mc - eps) & (results_mc[1, :] <= my_portfolio_ret_mc + eps)
    filt_r = results_mc[:, mask]; filt_w = weights_array_mc[mask, :]
    if filt_r.shape[1] > 0:
        idx = np.argmin(filt_r[0, :])
        p = collect_portfolio_info_mtc("My Portfolio Min Vol", idx, filt_r, filt_w,
                                        portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                        portfolio_returns_simple, tickers, annualisation_factor,
                                        ax_mc, "P", "c", 200)
        portfolios_mc.append(p)

    mask = (results_mc[0, :] >= my_portfolio_std_mc - eps) & (results_mc[0, :] <= my_portfolio_std_mc + eps)
    filt_r = results_mc[:, mask]; filt_w = weights_array_mc[mask, :]
    if filt_r.shape[1] > 0:
        idx = np.argmax(filt_r[1, :])
        p = collect_portfolio_info_mtc("My Portfolio Max Ret", idx, filt_r, filt_w,
                                        portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                        portfolio_returns_simple, tickers, annualisation_factor,
                                        ax_mc, "P", "m", 200)
        portfolios_mc.append(p)

    min_vol_idx = np.argmin(results_mc[0])
    p_minv = collect_portfolio_info_mtc("Min Volatility Portfolio", min_vol_idx, results_mc, weights_mc,
                                         portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                         portfolio_returns_simple, tickers, annualisation_factor,
                                         ax_mc, "*", "c", 500)
    portfolios_mc.append(p_minv)
    min_vol_ret_mc = p_minv["ret"]

    max_ret_idx = np.argmax(results_mc[1])
    p_maxr = collect_portfolio_info_mtc("Max Return Portfolio", max_ret_idx, results_mc, weights_mc,
                                         portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                         portfolio_returns_simple, tickers, annualisation_factor,
                                         ax_mc, "*", "m", 500)
    portfolios_mc.append(p_maxr)
    max_ret_ret_mc = p_maxr["ret"]
    max_ret_std_mc = p_maxr["std_dev"]

    max_sharpe_idx = np.argmax(results_mc[2])
    p_ms = collect_portfolio_info_mtc("Max Sharpe Portfolio", max_sharpe_idx, results_mc, weights_mc,
                                       portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                       portfolio_returns_simple, tickers, annualisation_factor,
                                       ax_mc, "*", "r", 500)
    portfolios_mc.append(p_ms)

    with st.spinner("Computing Sortino for Monte Carlo portfolios..."):
        results_mc_so, weights_mc_so = random_portfolios_sortino(
            num_portfolios, portfolio_mean_returns, portfolio_cov_matrix, portfolio_returns_simple,
            risk_free_rate, annualisation_factor)
    max_sortino_mc_idx = np.argmax(results_mc_so[3])
    p_mso = collect_portfolio_info_mtc("Max Sortino Portfolio", max_sortino_mc_idx,
        results_mc_so, weights_mc_so, portfolio_mean_returns, portfolio_cov_matrix,
        risk_free_rate, portfolio_returns_simple, tickers, annualisation_factor,
        ax_mc, "*", "orange", 500)
    portfolios_mc.append(p_mso)

    if custom_target_ret is not None:
        if min_vol_ret_mc <= custom_target_ret <= max_ret_ret_mc:
            mask = (results_mc[1, :] >= custom_target_ret - eps) & (results_mc[1, :] <= custom_target_ret + eps)
            filt_r = results_mc[:, mask]; filt_w = weights_array_mc[mask, :]
            if filt_r.shape[1] > 0:
                idx = np.argmin(filt_r[0, :])
                p = collect_portfolio_info_mtc("Custom Portfolio Min Vol", idx, filt_r, filt_w,
                                                portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                                portfolio_returns_simple, tickers, annualisation_factor,
                                                ax_mc, "X", "c", 200)
                portfolios_mc.append(p)

    if custom_target_vol is not None:
        if p_minv["std_dev"] <= custom_target_vol <= max_ret_std_mc:
            mask = (results_mc[0, :] >= custom_target_vol - eps) & (results_mc[0, :] <= custom_target_vol + eps)
            filt_r = results_mc[:, mask]; filt_w = weights_array_mc[mask, :]
            if filt_r.shape[1] > 0:
                idx = np.argmax(filt_r[1, :])
                p = collect_portfolio_info_mtc("Custom Portfolio Max Ret", idx, filt_r, filt_w,
                                                portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                                portfolio_returns_simple, tickers, annualisation_factor,
                                                ax_mc, "X", "m", 200)
                portfolios_mc.append(p)

    single_etfs_std_dev = portfolio_returns_simple.std() * np.sqrt(annualisation_factor)
    single_etfs_ret = portfolio_mean_returns * annualisation_factor
    ax_mc.scatter(single_etfs_std_dev, single_etfs_ret, marker="o", s=200, zorder=6)
    for i, txt in enumerate(tickers):
        ax_mc.annotate(txt, (single_etfs_std_dev.iloc[i], single_etfs_ret.iloc[i]),
                       xytext=(10, 0), textcoords="offset points", fontsize=9)

    ax_mc.set_title("Simulated Portfolio Optimization based on Efficient Frontier")
    ax_mc.set_xlabel("Annualised Volatility")
    ax_mc.set_ylabel("Annualised Returns")
    ax_mc.legend(labelspacing=0.8)
    st.pyplot(fig_mc)
    plt.close(fig_mc)

    fig_mc_so, ax_mc_so = plt.subplots(figsize=(10, 7))
    sc_so = ax_mc_so.scatter(results_mc_so[0, :], results_mc_so[1, :],
        c=results_mc_so[3, :], cmap="YlGnBu", marker="o", s=10, alpha=0.3)
    plt.colorbar(sc_so, ax=ax_mc_so, label="Sortino Ratio")
    for p in portfolios_mc:
        marker = {"My Portfolio": "P", "Min Volatility Portfolio": "*",
                   "Max Return Portfolio": "*", "Max Sharpe Portfolio": "*",
                   "Max Sortino Portfolio": "*"}.get(p["name"], "X")
        color  = {"My Portfolio": "y", "Min Volatility Portfolio": "c",
                   "Max Return Portfolio": "m", "Max Sharpe Portfolio": "r",
                   "Max Sortino Portfolio": "orange"}.get(p["name"], "gray")
        size   = 500 if marker == "*" else 200
        ax_mc_so.scatter(p["std_dev"], p["ret"], marker=marker, color=color,
                         s=size, label=p["name"], zorder=5)
    single_etfs_std_dev_so = portfolio_returns_simple.std() * np.sqrt(annualisation_factor)
    single_etfs_ret_so = portfolio_mean_returns * annualisation_factor
    ax_mc_so.scatter(single_etfs_std_dev_so, single_etfs_ret_so, marker="o", s=200, zorder=6)
    for i, txt in enumerate(tickers):
        ax_mc_so.annotate(txt, (single_etfs_std_dev_so.iloc[i], single_etfs_ret_so.iloc[i]),
                          xytext=(10, 0), textcoords="offset points", fontsize=9)
    ax_mc_so.set_title("Simulated Portfolio Optimization — Sortino Ratio")
    ax_mc_so.set_xlabel("Annualised Volatility")
    ax_mc_so.set_ylabel("Annualised Returns")
    ax_mc_so.legend(labelspacing=0.8)
    st.pyplot(fig_mc_so)
    plt.close(fig_mc_so)

    display_portfolio_cards(portfolios_mc, alpha)


# ─────────────────────────────────────────────────────────────────
# SECTION 7 — SCIPY EFFICIENT FRONTIER (VOL-BASED)
# ─────────────────────────────────────────────────────────────────

def render_scipy_ef(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                     tickers, annualisation_factor, risk_free_rate, num_portfolios,
                     num_eff_portfolios, eps, custom_target_ret, custom_target_vol,
                     my_portfolio_allocation, alpha, real_terms=False):
    st.header("8. Scipy Efficient Frontier Portfolio Optimization")
    st.caption("⚖️ Rebalancing: **per period** — the frontier and all return/volatility figures here "
               "assume per-period rebalancing (the basis MPT optimization requires), independent of the "
               "sidebar Rebalancing-frequency setting (which governs §6).")
    if real_terms:
        st.caption("💶 Shown in **nominal** terms despite the real-returns setting: a constant inflation "
                   "rate leaves the efficient-frontier *weights* unchanged (the real risk premium is "
                   "inflation-invariant) — it would only shift the return axis down by the inflation rate.")
    render_section_help(
        "This section mathematically solves for the best portfolios — rather than guessing "
        "randomly — and draws the efficient frontier: the best return achievable at each level of risk.",
        ["scipy_optimization", "efficient_frontier_line", "avg_annual_return", "annual_volatility",
         "sharpe", "sortino", "marked_portfolios", "max_drawdown", "cvar"],
    )

    results_sc, _ = random_portfolios(num_portfolios, portfolio_mean_returns,
                                       portfolio_cov_matrix, risk_free_rate, annualisation_factor)

    fig_sc, ax_sc = plt.subplots(figsize=(10, 7))
    sc2 = ax_sc.scatter(results_sc[0, :], results_sc[1, :], c=results_sc[2, :],
                        cmap="YlGnBu", marker="o", s=10, alpha=0.3)
    plt.colorbar(sc2, ax=ax_sc, label="Sharpe Ratio")

    portfolios_sc = []

    my_portfolio_weights = np.array(list(my_portfolio_allocation.values()), dtype=np.float64)

    p = collect_portfolio_info("My Portfolio", my_portfolio_weights, portfolio_mean_returns,
                                portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                tickers, annualisation_factor, ax_sc, "P", "y", 500)
    portfolios_sc.append(p)
    my_ret_sc = p["ret"]
    my_std_sc = p["std_dev"]

    my_min_vol_sc = efficient_return(portfolio_mean_returns, portfolio_cov_matrix, my_ret_sc, annualisation_factor)
    if my_min_vol_sc.success:
        p = collect_portfolio_info("My Portfolio Min Vol", my_min_vol_sc["x"], portfolio_mean_returns,
                                    portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                    tickers, annualisation_factor, ax_sc, "P", "c", 200)
        portfolios_sc.append(p)
    else:
        st.warning(f"Optimization failed for My Portfolio Min Vol: {my_min_vol_sc.message}")

    my_max_ret_sc = efficient_volatility(portfolio_mean_returns, portfolio_cov_matrix, my_std_sc, annualisation_factor)
    if my_max_ret_sc.success:
        p = collect_portfolio_info("My Portfolio Max Ret", my_max_ret_sc["x"], portfolio_mean_returns,
                                    portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                    tickers, annualisation_factor, ax_sc, "P", "m", 200)
        portfolios_sc.append(p)
    else:
        st.warning(f"Optimization failed for My Portfolio Max Ret: {my_max_ret_sc.message}")

    min_vol_sc = minimize_volatility(portfolio_mean_returns, portfolio_cov_matrix, annualisation_factor)
    if min_vol_sc.success:
        p_mv = collect_portfolio_info("Min Volatility Portfolio", min_vol_sc["x"], portfolio_mean_returns,
                                       portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                       tickers, annualisation_factor, ax_sc, "*", "c", 500)
        portfolios_sc.append(p_mv)
    else:
        st.warning(f"Optimization failed for Min Volatility Portfolio: {min_vol_sc.message}")
        p_mv = None

    max_ret_sc = maximize_return(portfolio_mean_returns, portfolio_cov_matrix, annualisation_factor)
    if max_ret_sc.success:
        p_mr = collect_portfolio_info("Max Return Portfolio", max_ret_sc["x"], portfolio_mean_returns,
                                       portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                       tickers, annualisation_factor, ax_sc, "*", "m", 500)
        portfolios_sc.append(p_mr)
    else:
        st.warning(f"Optimization failed for Max Return Portfolio: {max_ret_sc.message}")
        p_mr = None

    max_sharpe_sc = max_sharpe_ratio(portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate, annualisation_factor)
    if max_sharpe_sc.success:
        p_ms_sc = collect_portfolio_info("Max Sharpe Portfolio", max_sharpe_sc["x"], portfolio_mean_returns,
                                          portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                          tickers, annualisation_factor, ax_sc, "*", "r", 500)
        portfolios_sc.append(p_ms_sc)
    else:
        st.warning(f"Optimization failed for Max Sharpe Portfolio: {max_sharpe_sc.message}")

    max_sortino_sc_res = max_sortino_ratio(portfolio_mean_returns, portfolio_cov_matrix, portfolio_returns_simple,
                                           risk_free_rate, annualisation_factor)
    if max_sortino_sc_res.success:
        p_mso_sc = collect_portfolio_info("Max Sortino Portfolio", max_sortino_sc_res["x"],
            portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
            tickers, annualisation_factor, ax_sc, "*", "orange", 500)
        portfolios_sc.append(p_mso_sc)
    else:
        st.warning(f"Optimization failed for Max Sortino Portfolio: {max_sortino_sc_res.message}")

    if custom_target_ret is not None and p_mv is not None and p_mr is not None:
        if p_mv["ret"] <= custom_target_ret <= p_mr["ret"]:
            cust_mv = efficient_return(portfolio_mean_returns, portfolio_cov_matrix, custom_target_ret, annualisation_factor)
            if cust_mv.success:
                p = collect_portfolio_info("Custom Portfolio Min Vol", cust_mv["x"], portfolio_mean_returns,
                                            portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                            tickers, annualisation_factor, ax_sc, "X", "c", 200)
                portfolios_sc.append(p)
            else:
                st.warning(f"Optimization failed for Custom Portfolio Min Vol: {cust_mv.message}")

    if custom_target_vol is not None and p_mv is not None and p_mr is not None:
        if p_mv["std_dev"] <= custom_target_vol <= p_mr["std_dev"]:
            cust_mr = efficient_volatility(portfolio_mean_returns, portfolio_cov_matrix, custom_target_vol, annualisation_factor)
            if cust_mr.success:
                p = collect_portfolio_info("Custom Portfolio Max Ret", cust_mr["x"], portfolio_mean_returns,
                                            portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                            tickers, annualisation_factor, ax_sc, "X", "m", 200)
                portfolios_sc.append(p)
            else:
                st.warning(f"Optimization failed for Custom Portfolio Max Ret: {cust_mr.message}")

    if p_mv is not None and p_mr is not None:
        target_range = np.linspace(p_mv["ret"], p_mr["ret"], num_eff_portfolios)
        eff_portfolios = efficient_frontier_fn(portfolio_mean_returns, portfolio_cov_matrix, target_range, annualisation_factor)
        ax_sc.plot([p_["fun"] for p_ in eff_portfolios], target_range, linestyle="-.", linewidth=2.0, color="black", label="efficient frontier")
    else:
        st.warning("Cannot plot efficient frontier: Min Volatility or Max Return optimization failed.")

    single_etfs_std_dev = portfolio_returns_simple.std() * np.sqrt(annualisation_factor)
    single_etfs_ret = portfolio_mean_returns * annualisation_factor
    ax_sc.scatter(single_etfs_std_dev, single_etfs_ret, marker="o", s=200, zorder=6)
    for i, txt in enumerate(tickers):
        ax_sc.annotate(txt, (single_etfs_std_dev.iloc[i], single_etfs_ret.iloc[i]), xytext=(10, 0), textcoords="offset points", fontsize=9)

    ax_sc.set_title("Calculated Portfolio Optimization based on Efficient Frontier")
    ax_sc.set_xlabel("Annualised Volatility")
    ax_sc.set_ylabel("Annualised Returns")
    ax_sc.legend(labelspacing=0.8)
    st.pyplot(fig_sc)
    plt.close(fig_sc)

    with st.spinner("Computing Sortino for Scipy portfolios..."):
        results_sc_so, _ = random_portfolios_sortino(
            num_portfolios, portfolio_mean_returns, portfolio_cov_matrix, portfolio_returns_simple,
            risk_free_rate, annualisation_factor)

    fig_sc_so, ax_sc_so = plt.subplots(figsize=(10, 7))
    sc2_so = ax_sc_so.scatter(results_sc_so[0, :], results_sc_so[1, :],
        c=results_sc_so[3, :], cmap="YlGnBu", marker="o", s=10, alpha=0.3)
    plt.colorbar(sc2_so, ax=ax_sc_so, label="Sortino Ratio")
    for p in portfolios_sc:
        marker = {"My Portfolio": "P", "Min Volatility Portfolio": "*",
                   "Max Return Portfolio": "*", "Max Sharpe Portfolio": "*",
                   "Max Sortino Portfolio": "*"}.get(p["name"], "X")
        color  = {"My Portfolio": "y", "Min Volatility Portfolio": "c",
                   "Max Return Portfolio": "m", "Max Sharpe Portfolio": "r",
                   "Max Sortino Portfolio": "orange"}.get(p["name"], "gray")
        size   = 500 if marker == "*" else 200
        ax_sc_so.scatter(p["std_dev"], p["ret"], marker=marker, color=color,
                         s=size, label=p["name"], zorder=5)
    single_etfs_std_dev_sc = portfolio_returns_simple.std() * np.sqrt(annualisation_factor)
    single_etfs_ret_sc = portfolio_mean_returns * annualisation_factor
    ax_sc_so.scatter(single_etfs_std_dev_sc, single_etfs_ret_sc, marker="o", s=200, zorder=6)
    for i, txt in enumerate(tickers):
        ax_sc_so.annotate(txt, (single_etfs_std_dev_sc.iloc[i], single_etfs_ret_sc.iloc[i]),
                          xytext=(10, 0), textcoords="offset points", fontsize=9)
    if p_mv is not None and p_mr is not None:
        ax_sc_so.plot([p_["fun"] for p_ in eff_portfolios], target_range,
                      linestyle="-.", linewidth=2.0, color="black", label="efficient frontier")
    else:
        st.warning("Cannot plot efficient frontier (Sortino): Min Volatility or Max Return optimization failed.")
    ax_sc_so.set_title("Calculated Portfolio Optimization — Sortino Ratio")
    ax_sc_so.set_xlabel("Annualised Volatility")
    ax_sc_so.set_ylabel("Annualised Returns")
    ax_sc_so.legend(labelspacing=0.8)
    st.pyplot(fig_sc_so)
    plt.close(fig_sc_so)

    display_portfolio_cards(portfolios_sc, alpha)


# ─────────────────────────────────────────────────────────────────
# §6 SUBSECTION — TAIL RISK & RETURN DISTRIBUTION (merged from the former standalone VaR section)
# ─────────────────────────────────────────────────────────────────

def render_tail_risk(my_portfolio_returns, alpha):
    """Tail-risk / return-distribution subsection of §6.

    Takes the portfolio's per-period return series already computed by
    `render_input_portfolio_analysis` (held at the sidebar rebalancing cadence) — so the
    series is built once — and renders parametric + historical VaR/CVaR, the distribution
    histogram, and fat-tail diagnostics. No section header or help expander of its own; §6
    owns those (its expander pulls the zscore/var_parametric/var_historical/cvar/
    return_distribution descriptions).
    """
    st.subheader("Tail Risk & Return Distribution")
    st.caption("Per-period losses for the portfolio held at the rebalancing cadence set above — "
               "the distribution of its single-period returns.")

    z_score = stats.norm.ppf(1 - alpha)
    tail = 1 - alpha  # tail probability (e.g. 0.05 for 95% confidence)
    mu_var = my_portfolio_returns.mean()
    median_var = my_portfolio_returns.median()
    sigma_var = my_portfolio_returns.std()

    # Parametric (normal-model) tail risk, per period — reported as positive losses
    VaR_return = mu_var + z_score * sigma_var            # parametric VaR threshold return (negative)
    param_var_loss = -VaR_return
    phi_z = stats.norm.pdf(z_score)
    param_cvar_loss = sigma_var * phi_z / tail - mu_var  # normal expected shortfall

    # Historical (empirical) tail risk, per period — no distribution assumption. Only trustworthy
    # when enough observations fall in the tail: n_tail ~ (1 - alpha) * n.
    n_obs = len(my_portfolio_returns)
    n_tail = int(round(tail * n_obs))
    hist_ok = n_tail >= 5                # below this the empirical tail is essentially noise
    hist_indicative = 5 <= n_tail < 20   # usable but shaky
    hist_var_return = my_portfolio_returns.quantile(tail)  # empirical tail quantile (negative)
    hist_var_loss = -hist_var_return
    hist_cvar_loss = cvar(my_portfolio_returns, tail)      # mean of the worst `tail` fraction

    # Shape of the actual return distribution
    skew = my_portfolio_returns.skew()
    exkurt = my_portfolio_returns.kurtosis()  # excess (Fisher): 0 = normal

    x = np.linspace(
        min(my_portfolio_returns.min(), mu_var - 4 * sigma_var),
        max(my_portfolio_returns.max(), mu_var + 4 * sigma_var),
        1000,
    )
    pdf = stats.norm.pdf(x, mu_var, sigma_var)

    fig_var, ax_var = plt.subplots(figsize=(10, 5))
    ax_var.plot(x, pdf, "b--", linewidth=1, label="Normal model (PDF)")
    ax_var.fill_between(x, 0, pdf, where=(x < VaR_return), color="red", alpha=0.25, label=f"worst {tail:.0%} (normal)")
    ax_var.hist(my_portfolio_returns, bins=50, density=True, alpha=0.7, color="magenta", edgecolor="black")
    ax_var.axvline(VaR_return, color="red", linestyle="--", linewidth=1.2, label=f"Parametric VaR ({param_var_loss:.1%})")
    if hist_ok:
        ax_var.axvline(hist_var_return, color="darkorange", linestyle=":", linewidth=1.8, label=f"Historical VaR ({hist_var_loss:.1%})")
    ax_var.set_title("Distribution of Returns — My Portfolio")
    ax_var.set_xlabel("Per-period return")
    ax_var.set_ylabel("Frequency (density)")
    ax_var.legend()
    ax_var.grid(alpha=0.3)
    st.pyplot(fig_var)
    plt.close(fig_var)

    st.markdown("**Per-period return profile**")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Mean (μ)", f"{mu_var:.3%}")
    d2.metric("Median", f"{median_var:.3%}",
              help="The middle per-period return — half of periods did better, half worse. Compare it "
                   "to the mean alongside: mean above median = a right tail (occasional big gains), "
                   "below = a left tail (occasional big losses). A robust companion to skew.")
    d3.metric("Volatility (σ)", f"{sigma_var:.3%}")
    d4.metric("Skew", f"{skew:.2f}",
              help="Asymmetry of returns. Negative means crashes tend to be bigger than rallies — a warning sign.")
    d5.metric("Excess kurtosis", f"{exkurt:.2f}",
              help="Tail fatness vs a normal bell curve. 0 = normal; higher means extreme moves happen more often "
                   "than the normal model assumes.")

    st.markdown(f"**Tail-risk at {alpha:.0%} confidence — per-period loss**")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Parametric VaR", f"{param_var_loss:.2%}",
        help=f"Assuming returns follow a perfect bell curve, there is only about a {tail:.0%} chance of a "
             "single-period loss worse than this.",
    )
    r2.metric(
        "Parametric CVaR", f"{param_cvar_loss:.2%}",
        help=f"Assuming a perfect bell curve: if a period does land in that worst {tail:.0%}, this is the "
             "average loss the model expects across those periods.",
    )
    if hist_ok:
        note = f" (indicative - only {n_tail} observations in the tail)" if hist_indicative else ""
        r3.metric(
            "Historical VaR", f"{hist_var_loss:.2%}",
            help=f"From the actual return history: the portfolio lost more than this on only its worst "
                 f"{tail:.0%} of periods ({n_tail} of {n_obs}). No bell-curve assumption.{note}",
        )
        r4.metric(
            "Historical CVaR", f"{hist_cvar_loss:.2%}",
            help=f"From the actual return history: across its worst {tail:.0%} of periods ({n_tail} of "
                 f"{n_obs}), this was the average loss. No bell-curve assumption.{note}",
        )
    else:
        insufficient = (
            f"Not enough history for a reliable estimate: only {n_tail} observation(s) fall in the worst "
            f"{tail:.0%} tail (of {n_obs}). Use a longer date range, a higher-frequency data period, or a "
            "lower confidence level."
        )
        r3.metric("Historical VaR", "n/a", help=insufficient)
        r4.metric("Historical CVaR", "n/a", help=insufficient)

    if not hist_ok:
        st.info(
            f"Too little overlapping history to judge tail behaviour from actual data — only {n_tail} "
            f"observation(s) land in the worst {tail:.0%} (of {n_obs}). The parametric (normal) figures "
            "are shown, but treat tail risk with caution and add more data if you can."
        )
    elif hist_var_loss > param_var_loss or exkurt > 1.0:
        st.warning(
            f"This portfolio's losses are fatter-tailed than a normal bell curve (excess kurtosis "
            f"{exkurt:.1f}). The **historical** VaR/CVaR are the more honest picture of tail risk here — "
            "the parametric (normal) figures tend to understate it."
        )
    else:
        st.info(
            "This portfolio's returns are reasonably close to a normal bell curve, so the parametric and "
            "historical figures are broadly consistent."
        )
