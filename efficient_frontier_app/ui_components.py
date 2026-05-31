"""
UI rendering functions for Efficient Frontier dashboard.
All section renderers receive data as explicit parameters (C-like style).
"""

import streamlit as st
import streamlit.components.v1 as components
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
    portfolio_annualised_performance_VaR,
    portfolio_downside_deviation,
    random_portfolios,
    random_portfolios_sortino,
    random_portfolios_VaR,
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
)
from data_handling import evaluate_simple_return, evaluate_CAGR, evaluate_return_metrics
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
            "sortino": sortino, "alloc": alloc, "var": None,
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
            "sortino": sortino, "alloc": alloc, "var": None,
            "max_dd": mdd, "port_returns": port_returns}


def collect_portfolio_info_VaR(name, weights, mean_returns, cov_matrix, risk_free_rate,
                                portfolio_returns_simple, tickers, alpha, annualisation_factor,
                                ax, marker, color, size):
    std_dev, ret, var = portfolio_annualised_performance_VaR(weights, mean_returns, cov_matrix, alpha, annualisation_factor)
    sharpe  = (ret - risk_free_rate) / std_dev
    dd      = portfolio_downside_deviation(weights, portfolio_returns_simple, annualisation_factor, risk_free_rate)
    sortino = (ret - risk_free_rate) / dd if dd > 0 else np.nan
    alloc   = make_allocation_df(weights, tickers)
    port_returns = portfolio_returns_simple.dot(weights)
    mdd     = max_drawdown(port_returns)
    ax.scatter(var, ret, marker=marker, color=color, s=size, label=name, zorder=5)
    return {"name": name, "std_dev": std_dev, "ret": ret, "sharpe": sharpe,
            "sortino": sortino, "alloc": alloc, "var": var,
            "max_dd": mdd, "port_returns": port_returns}


def collect_portfolio_info_mtc_VaR(name, index, results, weights_list, mean_returns,
                                     cov_matrix, risk_free_rate, portfolio_returns_simple,
                                     tickers, annualisation_factor, ax, marker, color, size):
    std_dev = results[0, index]
    ret     = results[1, index]
    sharpe  = results[2, index]
    var     = results[3, index]
    weights = weights_list[index]
    dd      = portfolio_downside_deviation(weights, portfolio_returns_simple, annualisation_factor, risk_free_rate)
    sortino = (ret - risk_free_rate) / dd if dd > 0 else np.nan
    alloc   = make_allocation_df(weights, tickers)
    port_returns = portfolio_returns_simple.dot(weights)
    mdd     = max_drawdown(port_returns)
    ax.scatter(var, ret, marker=marker, color=color, s=size, label=name, zorder=5)
    return {"name": name, "std_dev": std_dev, "ret": ret, "sharpe": sharpe,
            "sortino": sortino, "alloc": alloc, "var": var,
            "max_dd": mdd, "port_returns": port_returns}


def display_portfolio_cards(portfolios, alpha):
    for p in portfolios:
        with st.expander(f"{p['name']}", expanded=False):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric(
                "Average annual return",
                f"{p['ret']:.2%}",
                help="Average return per year, estimated from daily returns — the same figure "
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
                     "not annualised.",
            )
            # CVaR computed once here, at the user's chosen confidence level, from the portfolio's
            # own return series (single CVaR definition / sign across the whole app: positive = loss).
            port_returns = p.get("port_returns")
            cvar_val = cvar(port_returns, 1 - alpha) if port_returns is not None else 0.0
            c6.metric(
                f"CVaR ({alpha:.0%})",
                f"{cvar_val:.2%}",
                help="Expected shortfall: the average loss on the worst (1−confidence) slice of "
                     "individual periods (e.g. days). A per-period figure — not annualised — so it "
                     "sits on a shorter horizon than the annual return and volatility above.",
            )
            if p.get("var") is not None:
                st.metric(
                    "Value at Risk (annual)",
                    f"{p['var']:.2%}",
                    help="Parametric 1-year Value at Risk at the chosen confidence level, assuming "
                         "normally-distributed returns.",
                )
            alloc_series = p["alloc"].iloc[0]
            left, right  = st.columns([1, 1])
            with left:
                st.dataframe(p["alloc"], use_container_width=True)
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
# SECTION 1 — LOAD ETF DATA (spike warnings + data availability)
# ─────────────────────────────────────────────────────────────────

def render_load_etf_data(tickers, spike_warnings, data_availability):
    st.header("1. Load ETF Data")
    render_section_help(
        "This section loads your price data and checks it is usable — flagging suspicious "
        "jumps and showing how much history all your assets share.",
        ["price_spikes", "data_window"],
    )

    if spike_warnings:
        tickers_flagged = ", ".join(f"**{t}**" for t in spike_warnings)
        st.warning(f" Large price moves (> 60%) detected in: {tickers_flagged}")
        with st.expander("Show details", expanded=False):
            for ticker, events in spike_warnings.items():
                st.markdown(f"**{ticker}**")
                rows = [
                    {"Date": d, "Prev. Price": f"{prev:.4f}",
                     "New Price": f"{new:.4f}", "Change": f"{pct:+.1%}"}
                    for d, prev, new, pct in events
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(
                "These events may be stock splits, data errors, or genuine extreme moves. "
                "If they are splits already reflected in the Adj Close column they are harmless; "
                "otherwise consider cleaning the data before running the analysis."
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

    html_gauge = (
        f'<html><body style="margin:0;padding:4px 0 0 0;background:transparent;">'
        f'{svg}'
        f'</body></html>'
    )

    components.html(html_gauge, height=SVG_H + 24, scrolling=False)
    st.divider()


# ─────────────────────────────────────────────────────────────────
# SECTION 2 — PER-ETF ANALYTICS
# ─────────────────────────────────────────────────────────────────

def render_per_etf_analytics(merged_df, tickers, folder_path, filename_suffix, filter_date_string,
                              return_type, annualisation_factor):
    st.header("2. Per-ETF Analytics")
    render_section_help(
        "This section looks at each ETF on its own: how much it returned, how much it swung, "
        "and its worst fall — so you understand each holding before combining them.",
        ["simple_return", "calendar_year_return", "cagr", "avg_annual_return",
         "annual_volatility", "max_drawdown", "cumulative_return", "lookback_annual_metrics"],
    )

    for ticker in tickers:
        subdf = merged_df[["date", ticker]].copy().rename(columns={ticker: "adj close"})
        subdf = subdf.sort_values("date")
        subdf = subdf[subdf["date"] <= filter_date_string]
        subdf.set_index("date", inplace=True)

        with st.expander(f"{ticker}", expanded=False):
            end_date = subdf.index[-1]
            full_years = int((subdf.index[-1] - subdf.index[0]).days / 365.25)
            start_dates_simple = {
                "YTD": subdf[subdf.index.year == (subdf.index.max().year - 1)].index.max(),
                "1mo": end_date - pd.DateOffset(months=1),
                "3mo": end_date - pd.DateOffset(months=3),
                "6mo": end_date - pd.DateOffset(months=6),
                "1y": end_date - pd.DateOffset(years=1),
                "3y": end_date - pd.DateOffset(years=3),
                "5y": end_date - pd.DateOffset(years=5),
                f"{full_years}y": end_date - pd.DateOffset(years=full_years),
            }

            simple_rows = []
            for label, sd in start_dates_simple.items():
                sr = evaluate_simple_return(subdf["adj close"], sd, end_date)
                simple_rows.append({"Period": label, "From": sd.date(), "To": end_date.date(),
                                     "Simple Return": f"{sr:.2%}"})
            st.subheader("Simple Return (single-period)")
            st.dataframe(pd.DataFrame(simple_rows), use_container_width=True, hide_index=True)

            yearly_rows = []
            for yr_offset in range(3):
                considered_year = subdf.index.max().year - (1 + yr_offset)
                ed = subdf[subdf.index.year == considered_year].index.max()
                sd = ed - pd.DateOffset(years=1)
                sr = evaluate_simple_return(subdf["adj close"], sd, ed)
                yearly_rows.append({"Year": considered_year, "From": sd.date(), "To": ed.date(),
                                     "Simple Return": f"{sr:.2%}"})
            st.dataframe(pd.DataFrame(yearly_rows), use_container_width=True, hide_index=True)

            start_dates_cagr = {
                "1y": end_date - pd.DateOffset(years=1),
                "3y": end_date - pd.DateOffset(years=3),
                "5y": end_date - pd.DateOffset(years=5),
                f"{full_years}y": end_date - pd.DateOffset(years=full_years),
            }
            cagr_rows = []
            for label, sd in start_dates_cagr.items():
                cagr = evaluate_CAGR(subdf["adj close"], sd, end_date)
                cagr_rows.append({"Period": label, "From": sd.date(), "To": end_date.date(),
                                   "CAGR": f"{cagr:.2%}"})
            st.subheader("CAGR (Compound Annual Growth Rate)")
            st.dataframe(pd.DataFrame(cagr_rows), use_container_width=True, hide_index=True)

            if return_type == "logarithmic":
                subdf["ret"] = np.log(subdf["adj close"] / subdf["adj close"].shift(1))
            else:
                subdf["ret"] = subdf["adj close"].pct_change()
            subdf = subdf.dropna(subset=["ret"])

            st.subheader("Annualised Metrics (simple returns, full history)")
            simple_ret = subdf["adj close"].pct_change().dropna()
            ann_mean = simple_ret.mean() * annualisation_factor
            ann_std = simple_ret.std() * np.sqrt(annualisation_factor)
            mdd = max_drawdown(simple_ret)
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Average annual return",
                f"{ann_mean:.2%}",
                help="Average return per year, estimated from daily returns — the same figure "
                     "used on the efficient-frontier chart and to compare portfolios.",
            )
            col2.metric("Annualised Volatility", f"{ann_std:.2%}")
            col3.metric("Max Drawdown", f"{mdd:.2%}")

            # Plot cumulative returns (always simple returns for interpretability)
            cumulative_ret = (1 + simple_ret).cumprod() - 1
            fig_cum, ax_cum = plt.subplots(figsize=(10, 4))
            ax_cum.plot(cumulative_ret.index, cumulative_ret.values, lw=1)
            ax_cum.set_title(f"{ticker} — Cumulative Returns")
            ax_cum.set_xlabel("Date")
            ax_cum.set_ylabel("Cumulative Return")
            ax_cum.grid(True)
            st.pyplot(fig_cum)
            plt.close(fig_cum)

            st.subheader("Annualised Metrics by Look-back Period (as of today)")
            rolling_rows = []
            for label, sd in start_dates_cagr.items():
                simple_ret = subdf["adj close"].loc[sd:end_date].pct_change().dropna()
                ann_ret = simple_ret.mean() * annualisation_factor
                vol = simple_ret.std() * np.sqrt(annualisation_factor)
                rolling_rows.append({
                    "Period": label, "From": sd.date(), "To": end_date.date(),
                    "Ann. Avg Return": f"{ann_ret:.2%}", "Ann. Volatility": f"{vol:.2%}",
                })
            st.dataframe(pd.DataFrame(rolling_rows), use_container_width=True, hide_index=True)

            st.subheader("Annualised Metrics by Look-back Period (as of last full year-end)")
            end_date_ly = subdf[subdf.index.year == (subdf.index.max().year - 1)].index.max()
            start_dates_ly = {
                "1y": end_date_ly - pd.DateOffset(years=1),
                "3y": end_date_ly - pd.DateOffset(years=3),
                "5y": end_date_ly - pd.DateOffset(years=5),
                f"{full_years}y": end_date_ly - pd.DateOffset(years=full_years),
            }
            rolling_ly_rows = []
            for label, sd in start_dates_ly.items():
                simple_ret = subdf["adj close"].loc[sd:end_date_ly].pct_change().dropna()
                ann_ret = simple_ret.mean() * annualisation_factor
                vol = simple_ret.std() * np.sqrt(annualisation_factor)
                rolling_ly_rows.append({
                    "Period": label, "From": sd.date(), "To": end_date_ly.date(),
                    "Ann. Avg Return": f"{ann_ret:.2%}", "Ann. Volatility": f"{vol:.2%}",
                })
            st.dataframe(pd.DataFrame(rolling_ly_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# SECTION 3 — ETF PRICES
# ─────────────────────────────────────────────────────────────────

def render_etf_prices(merged_df, tickers):
    st.header("3. ETF Prices")
    render_section_help(
        "This section plots prices so you can compare how your assets moved — both in real "
        "terms and rebased to a common starting point.",
        ["normalized_prices"],
    )

    st.subheader("Raw Closing Prices")
    with st.expander("Show raw price table"):
        st.dataframe(merged_df, use_container_width=True, hide_index=True)

    fig_raw, ax_raw = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        ax_raw.plot(merged_df["date"], merged_df[ticker], lw=1, label=ticker)
    ax_raw.set_title("ETFs Closing Prices Over Time")
    ax_raw.set_xlabel("Date")
    ax_raw.set_ylabel("Closing Prices [EUR]")
    ax_raw.legend(title="ETF")
    ax_raw.grid(True)
    st.pyplot(fig_raw)
    plt.close(fig_raw)

    norm_df = merged_df.copy()
    for ticker in tickers:
        norm_df[ticker] = (norm_df[ticker] / norm_df[ticker].iloc[0]) * 1000

    st.subheader("Normalized Closing Prices (base = 1000)")
    with st.expander("Show normalized price table"):
        st.dataframe(norm_df, use_container_width=True, hide_index=True)

    fig_norm, ax_norm = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        ax_norm.plot(norm_df["date"], norm_df[ticker], lw=1, label=ticker)
    ax_norm.set_title("ETFs Normalized Closing Prices Over Time")
    ax_norm.set_xlabel("Date")
    ax_norm.set_ylabel("Normalized Closing Prices [EUR]")
    ax_norm.legend(title="ETF")
    ax_norm.grid(True)
    st.pyplot(fig_norm)
    plt.close(fig_norm)

    return norm_df


def render_rolling_returns(rolling_returns, portfolio_rolling_returns, tickers,
                            rolling_window_years, return_type):
    st.header("3b. Rolling Returns")
    render_section_help(
        "This section shows returns over long rolling windows, so you can see what an investor "
        "would have earned holding for 1, 5 or 10 years starting at any point in time.",
        ["rolling_returns_asset", "rolling_returns_portfolio"],
    )
    if rolling_returns.shape[0] <= rolling_window_years:
        st.warning(f"Insufficient data for {rolling_window_years}-year rolling window.")
        return

    # Chart 1: Individual assets
    st.subheader(f"Individual Assets — {rolling_window_years}Y Rolling Returns ({return_type.capitalize()})")
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        ax.plot(rolling_returns["date"], rolling_returns[ticker], lw=1, label=ticker)
    ax.set_title(f"Rolling {rolling_window_years}-Year Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(title="ETF")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    # Chart 2: Portfolio
    st.subheader(f"Portfolio — {rolling_window_years}Y Rolling Returns (Simple)")
    port_df = portfolio_rolling_returns.reset_index()
    port_df.columns = ["date", "rolling_return"]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(port_df["date"], port_df["rolling_return"], lw=1.5, color="black", label="Portfolio")
    ax2.set_title(f"Portfolio Rolling {rolling_window_years}-Year Returns")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Rolling Return (Simple)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    plt.close(fig2)
    st.divider()



# ─────────────────────────────────────────────────────────────────
# SECTION 4 — RETURNS & STATISTICS
# ─────────────────────────────────────────────────────────────────

def render_returns_statistics(returns, portfolio_returns_simple, portfolio_mean_returns,
                               portfolio_cov_matrix, tickers, return_type, annualisation_factor,
                               risk_free_rate):
    st.header("4. Returns & Statistics")
    render_section_help(
        "This section measures how rewarding and how risky your assets have been, and crucially "
        "how they move together — the raw material for diversification.",
        ["return_stats", "sortino", "covariance", "correlation"],
    )

    min_return = returns.min()
    max_return = returns.max()
    mean_returns = returns.mean()
    returns_std_dev = returns.std()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader(f"Min Return ({return_type}, %)")
        st.dataframe((min_return * 100).round(2).to_frame("Min"), use_container_width=True)
    with col2:
        st.subheader(f"Max Return ({return_type}, %)")
        st.dataframe((max_return * 100).round(2).to_frame("Max"), use_container_width=True)
    with col3:
        st.subheader(f"Mean Return ({return_type}, %)")
        st.dataframe((mean_returns * 100).round(2).to_frame("Mean"), use_container_width=True)
    with col4:
        st.subheader("Std Dev (%)")
        st.dataframe((returns_std_dev * 100).round(2).to_frame(f"Std Dev ({return_type})"), use_container_width=True)

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
        use_container_width=True
    )

    st.subheader("Covariance Matrix (used for optimization — simple returns)")
    st.dataframe(portfolio_cov_matrix, use_container_width=True)

    st.subheader("Correlation Matrix (used for optimization — simple returns)")
    st.dataframe(portfolio_returns_simple.corr(), use_container_width=True)

    fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
    sns.heatmap(portfolio_returns_simple.corr(), annot=True, cmap="coolwarm", center=0, ax=ax_corr)
    ax_corr.set_title("Asset Correlation Matrix (simple returns)")
    st.pyplot(fig_corr)
    plt.close(fig_corr)

    fig_ret, ax_ret = plt.subplots(figsize=(12, 5))
    for c in returns.columns.values:
        ax_ret.plot(returns.index, returns[c] * 100, lw=1, alpha=0.8, label=c)
    ax_ret.legend(loc="upper right", fontsize=10)
    ax_ret.set_ylabel(f"{return_type} returns [%]")
    ax_ret.set_title(f"ETF {return_type.capitalize()} Returns")
    st.pyplot(fig_ret)
    plt.close(fig_ret)


# ─────────────────────────────────────────────────────────────────
# SECTION 5 — MONTE CARLO EFFICIENT FRONTIER (VOL-BASED)
# ─────────────────────────────────────────────────────────────────

def render_monte_carlo(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                        tickers, annualisation_factor, risk_free_rate, num_portfolios, eps,
                        custom_target_ret, custom_target_vol, my_portfolio_allocation, alpha):
    st.header("5. Monte Carlo Efficient Frontier (Volatility)")
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
# SECTION 6 — SCIPY EFFICIENT FRONTIER (VOL-BASED)
# ─────────────────────────────────────────────────────────────────

def render_scipy_ef(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                     tickers, annualisation_factor, risk_free_rate, num_portfolios,
                     num_eff_portfolios, eps, custom_target_ret, custom_target_vol,
                     my_portfolio_allocation, alpha):
    st.header("6. Scipy Efficient Frontier (Volatility)")
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
# SECTION 7 — VAR ANALYSIS
# ─────────────────────────────────────────────────────────────────

def render_var_analysis(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                         tickers, annualisation_factor, risk_free_rate, num_portfolios,
                         eps, alpha, custom_target_ret, custom_target_VaR, my_portfolio_allocation):
    st.header("7. Value at Risk (VaR) Analysis")
    render_section_help(
        "This section estimates how much you could lose on a bad day or in a bad tail of "
        "outcomes, using Value at Risk and its sibling CVaR.",
        ["zscore", "var_parametric", "var_historical", "cvar", "return_distribution",
         "var_frontier", "marked_portfolios"],
    )

    z_score = stats.norm.ppf(1 - alpha)
    tail = 1 - alpha  # tail probability (e.g. 0.05 for 95% confidence)
    my_portfolio_weights = np.array(list(my_portfolio_allocation.values()), dtype=np.float64)
    my_portfolio_returns = portfolio_returns_simple.dot(my_portfolio_weights)
    mu_var = my_portfolio_returns.mean()
    sigma_var = my_portfolio_returns.std()

    # Parametric (normal-model) tail risk, per period — reported as positive losses
    VaR_return = mu_var + z_score * sigma_var            # parametric VaR threshold return (negative)
    param_var_loss = -VaR_return
    phi_z = stats.norm.pdf(z_score)
    param_cvar_loss = sigma_var * phi_z / tail - mu_var  # normal expected shortfall

    # Historical (empirical) tail risk, per period — no distribution assumption
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
    ax_var.axvline(hist_var_return, color="darkorange", linestyle=":", linewidth=1.8, label=f"Historical VaR ({hist_var_loss:.1%})")
    ax_var.set_title("Distribution of Returns — My Portfolio")
    ax_var.set_xlabel("Per-period return")
    ax_var.set_ylabel("Frequency (density)")
    ax_var.legend()
    ax_var.grid(alpha=0.3)
    st.pyplot(fig_var)
    plt.close(fig_var)

    st.markdown("**Daily return profile**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Mean (μ)", f"{mu_var:.3%}")
    d2.metric("Volatility (σ)", f"{sigma_var:.3%}")
    d3.metric("Skew", f"{skew:.2f}",
              help="Asymmetry of returns. Negative means crashes tend to be bigger than rallies — a warning sign.")
    d4.metric("Excess kurtosis", f"{exkurt:.2f}",
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
    r3.metric(
        "Historical VaR", f"{hist_var_loss:.2%}",
        help=f"From the actual return history: the portfolio lost more than this on only its worst {tail:.0%} "
             "of periods. No bell-curve assumption.",
    )
    r4.metric(
        "Historical CVaR", f"{hist_cvar_loss:.2%}",
        help=f"From the actual return history: across its worst {tail:.0%} of periods, this was the average "
             "loss. No bell-curve assumption.",
    )

    if hist_var_loss > param_var_loss or exkurt > 1.0:
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

    st.subheader("Monte Carlo Efficient Frontier (VaR axis)")

    results_var, weights_var = random_portfolios_VaR(num_portfolios, portfolio_mean_returns,
                                                       portfolio_cov_matrix, risk_free_rate,
                                                       alpha, annualisation_factor)
    weights_array_var = np.array(weights_var)

    fig_var2, ax_var2 = plt.subplots(figsize=(10, 7))
    sc3 = ax_var2.scatter(results_var[3, :], results_var[1, :], c=results_var[2, :],
                           cmap="YlGnBu", marker="o", s=10, alpha=0.3)
    plt.colorbar(sc3, ax=ax_var2, label="Sharpe Ratio")

    portfolios_var = []

    p = collect_portfolio_info_VaR("My Portfolio", my_portfolio_weights, portfolio_mean_returns,
                                     portfolio_cov_matrix, risk_free_rate, portfolio_returns_simple,
                                     tickers, alpha, annualisation_factor, ax_var2, "P", "y", 500)
    portfolios_var.append(p)
    my_ret_var = p["ret"]
    my_VaR_var = p["var"]

    mask = (results_var[1, :] >= my_ret_var - eps) & (results_var[1, :] <= my_ret_var + eps)
    filt_r = results_var[:, mask]; filt_w = weights_array_var[mask, :]
    if filt_r.shape[1] > 0:
        idx = np.argmin(filt_r[3, :])
        p = collect_portfolio_info_mtc_VaR("My Portfolio Min VaR", idx, filt_r, filt_w,
                                            portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                            portfolio_returns_simple, tickers, annualisation_factor,
                                            ax_var2, "P", "c", 200)
        portfolios_var.append(p)

    mask = (results_var[3, :] >= my_VaR_var - eps) & (results_var[3, :] <= my_VaR_var + eps)
    filt_r = results_var[:, mask]; filt_w = weights_array_var[mask, :]
    if filt_r.shape[1] > 0:
        idx = np.argmax(filt_r[1, :])
        p = collect_portfolio_info_mtc_VaR("My Portfolio Max Ret", idx, filt_r, filt_w,
                                            portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                            portfolio_returns_simple, tickers, annualisation_factor,
                                            ax_var2, "P", "m", 200)
        portfolios_var.append(p)

    min_VaR_idx = np.argmin(results_var[3])
    p_mv_var = collect_portfolio_info_mtc_VaR("Min VaR Portfolio", min_VaR_idx, results_var, weights_var,
                                                portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                                portfolio_returns_simple, tickers, annualisation_factor,
                                                ax_var2, "*", "c", 500)
    portfolios_var.append(p_mv_var)
    min_VaR_ret = p_mv_var["ret"]
    min_VaR_VaR = p_mv_var["var"]

    max_ret_var_idx = np.argmax(results_var[1])
    p_mr_var = collect_portfolio_info_mtc_VaR("Max Return Portfolio", max_ret_var_idx, results_var, weights_var,
                                                portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                                portfolio_returns_simple, tickers, annualisation_factor,
                                                ax_var2, "*", "m", 500)
    portfolios_var.append(p_mr_var)
    max_ret_ret_var = p_mr_var["ret"]
    max_ret_VaR_var = p_mr_var["var"]

    max_sharpe_var_idx = np.argmax(results_var[2])
    p_ms_var = collect_portfolio_info_mtc_VaR("Max Sharpe Portfolio", max_sharpe_var_idx, results_var, weights_var,
                                               portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                               portfolio_returns_simple, tickers, annualisation_factor,
                                               ax_var2, "*", "r", 500)
    portfolios_var.append(p_ms_var)

    if custom_target_ret is not None:
        if min_VaR_ret <= custom_target_ret <= max_ret_ret_var:
            mask = (results_var[1, :] >= custom_target_ret - eps) & (results_var[1, :] <= custom_target_ret + eps)
            filt_r = results_var[:, mask]; filt_w = weights_array_var[mask, :]
            if filt_r.shape[1] > 0:
                idx = np.argmin(filt_r[0, :])
                p = collect_portfolio_info_mtc_VaR("Custom Portfolio Min Vol", idx, filt_r, filt_w,
                                                     portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                                     portfolio_returns_simple, tickers, annualisation_factor,
                                                     ax_var2, "X", "c", 200)
                portfolios_var.append(p)

    if custom_target_VaR is not None:
        if min_VaR_VaR <= custom_target_VaR <= max_ret_VaR_var:
            mask = (results_var[3, :] >= custom_target_VaR - eps) & (results_var[3, :] <= custom_target_VaR + eps)
            filt_r = results_var[:, mask]; filt_w = weights_array_var[mask, :]
            if filt_r.shape[1] > 0:
                idx = np.argmax(filt_r[1, :])
                p = collect_portfolio_info_mtc_VaR("Custom Portfolio Max Ret", idx, filt_r, filt_w,
                                                     portfolio_mean_returns, portfolio_cov_matrix, risk_free_rate,
                                                     portfolio_returns_simple, tickers, annualisation_factor,
                                                     ax_var2, "X", "m", 200)
                portfolios_var.append(p)

    single_etfs_std_dev = portfolio_returns_simple.std() * np.sqrt(annualisation_factor)
    single_etfs_ret = portfolio_mean_returns * annualisation_factor
    single_etfs_VaR = single_etfs_std_dev * abs(stats.norm.ppf(1 - alpha)) - single_etfs_ret
    ax_var2.scatter(single_etfs_VaR, single_etfs_ret, marker="o", s=200, zorder=6)
    for i, txt in enumerate(tickers):
        ax_var2.annotate(txt, (single_etfs_VaR.iloc[i], single_etfs_ret.iloc[i]), xytext=(10, 0), textcoords="offset points", fontsize=9)

    ax_var2.set_title(f"Simulated Portfolio Optimization based on Efficient Frontier (VaR, α={alpha})")
    ax_var2.set_xlabel(f"Value at Risk (α={alpha})")
    ax_var2.set_ylabel("Annualised Returns")
    ax_var2.legend(labelspacing=0.8)
    st.pyplot(fig_var2)
    plt.close(fig_var2)

    display_portfolio_cards(portfolios_var, alpha)

    st.success(" Analysis complete!")
