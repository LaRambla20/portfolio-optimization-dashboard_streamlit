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
    rebalanced_value_series,
    rebalanced_value_aftertax,
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
    YF_INTERVALS,
    fetch_price_series,
    probe_ticker,
    suggest_index_candidates,
    preview_candidate_fit,
    suggest_fx_ticker,
    q_hat_verdict,
    default_q_regime,
    Q_BANDS,
    synthesize_total_return,
)
from descriptions import DESCRIPTIONS, render_section_help


# ─────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────

def _geometric_annual_return(port_returns, annualisation_factor):
    """Geometric (compound) annual return of a per-period return series:
    ``(∏(1+r))^(N/n) − 1``. This is what compounding actually delivers — it sits ~σ²/2 below the
    arithmetic ``mean × N`` (volatility drag). NaN when the series is empty or wealth goes ≤ 0."""
    n = len(port_returns)
    if n == 0:
        return np.nan
    cum = float((1.0 + port_returns).prod())
    if cum <= 0:
        return np.nan
    return cum ** (annualisation_factor / n) - 1.0


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
    geo_ret = _geometric_annual_return(port_returns, annualisation_factor)
    ax.scatter(std_dev, ret, marker=marker, color=color, s=size, label=name, zorder=5)
    return {"name": name, "std_dev": std_dev, "ret": ret, "geo_ret": geo_ret, "sharpe": sharpe,
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
    geo_ret = _geometric_annual_return(port_returns, annualisation_factor)
    ax.scatter(std_dev, ret, marker=marker, color=color, s=size, label=name, zorder=5)
    return {"name": name, "std_dev": std_dev, "ret": ret, "geo_ret": geo_ret, "sharpe": sharpe,
            "sortino": sortino, "alloc": alloc,
            "max_dd": mdd, "port_returns": port_returns}


def display_portfolio_cards(portfolios, alpha):
    st.caption(
        "ℹ️ **Max Drawdown** and **CVaR** on these cards use a **per-period-rebalanced** "
        "(constant-weight) basis — the basis these optimization sections require. Frequent "
        "rebalancing damps drawdowns, so a **buy-and-hold** investor's drawdown is usually "
        "**larger** than the figure here; for your allocation at the sidebar cadence (default "
        "buy-and-hold) see §6."
    )
    for p in portfolios:
        with st.expander(f"{p['name']}", expanded=False):
            # Row 1: the two return figures together (arithmetic = optimizer input vs geometric =
            # realized), then volatility and Sharpe.
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            r1c1.metric(
                "Average annual return",
                f"{p['ret']:.2%}",
                help="Arithmetic mean of per-period returns × N — the *expected* return used as the "
                     "efficient-frontier axis and to rank portfolios. The realized compound growth (CAGR, "
                     "beside it) differs: volatility drag lowers it (≈σ²/2) while intra-period compounding "
                     "raises it, so for volatile portfolios CAGR is usually the lower of the two.",
            )
            geo_val = p.get("geo_ret")
            r1c2.metric(
                "Compound return (CAGR)",
                f"{geo_val:.2%}" if geo_val is not None and not np.isnan(geo_val) else "n/a",
                help="Geometric annual growth of this portfolio's return series — (∏(1+r))^(N/n) − 1 — "
                     "what compounding actually delivers. Differs from the arithmetic figure beside it "
                     "(volatility drag down ≈σ²/2 vs intra-period compounding up). Constant-weight "
                     "(per-period-rebalanced) basis, like the other card metrics.",
            )
            r1c3.metric("Ann. Volatility", f"{p['std_dev']:.2%}")
            r1c4.metric("Sharpe Ratio",  f"{p['sharpe']:.3f}")

            # Row 2: risk-adjusted + downside metrics.
            r2c1, r2c2, r2c3 = st.columns(3)
            sortino_val = p.get("sortino")
            r2c1.metric("Sortino Ratio", f"{sortino_val:.3f}" if sortino_val is not None and not np.isnan(sortino_val) else "n/a")
            r2c2.metric(
                "Max Drawdown",
                f"{p.get('max_dd', 0):.2%}",
                help="Worst peak-to-trough fall over the full history shown — a cumulative figure, "
                     "not annualised. Assumes constant weights (per-period rebalancing); a buy-and-hold "
                     "investor's drawdown is usually larger. For a less-frequent cadence see §6.",
            )
            # CVaR computed once here, at the user's chosen confidence level, from the portfolio's
            # own return series (single CVaR definition / sign across the whole app: positive = loss).
            port_returns = p.get("port_returns")
            cvar_val = cvar(port_returns, 1 - alpha) if port_returns is not None else 0.0
            r2c3.metric(
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


def _mixed_calendar_note(mixed_calendar, seven_day_tickers):
    """Caveat shown wherever the covariance/correlation (or the frontier/VaR built on it) is used,
    when the basket mixes 7-day (crypto) and ~5-day (equity) assets. The inner-join keeps only
    shared dates, folding the 7-day asset's weekend moves into the next shared day, so those figures
    are calendar-approximate. Detection comes from `compute_data_availability`; no-op otherwise."""
    if not mixed_calendar:
        return
    seven = ", ".join(f"**{t}**" for t in (seven_day_tickers or []))
    st.caption(
        f"📅 Mixed-calendar caveat: {seven} trade 7 days a week while the other assets trade ~5, so "
        "the covariance/correlation here — and the frontier/VaR built on it — are calendar-approximate "
        "(weekend moves fold into the next shared day). Switch the data period to weekly or monthly "
        "for cleaner figures."
    )


# ─────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA (split detection + anomaly warnings + data availability)
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# GUIDED EXTEND WIZARD (§ sidebar → modal dialog)
# ─────────────────────────────────────────────────────────────────
#
# Four validated steps: ETF → index → currency → confirm. Each step probes Yahoo and only
# enables its Continue button once the input actually works, so a dead ticker is caught here
# rather than after a reconstruction run produces nothing.
#
# Streamlit constraints this respects (see CLAUDE.md):
#   - st.dialog is a *fragment*: widget clicks rerun only this function, not the whole script.
#   - st.sidebar cannot be called in here — config arrives via st.session_state["wiz_cfg"].
#   - Elements created OUTSIDE the dialog are additive across dialog reruns, so the actual
#     reconstruction is queued into session_state and run by the main script after this closes.

WIZ_STEPS = ["1 · Fund", "2 · Index", "3 · Currency", "4 · Confirm"]
_WIZ_PPY = {"daily": 252, "weekly": 52, "monthly": 12}

# Step-3 conversion modes. Pre-selected from the two currencies, overridable by the user.
FX_MODE_PAIR = "Choose a conversion pair"
FX_MODE_NONE = "Proceed without a conversion pair"


WIZ_INPUT_KEYS = ("wiz_etf_in", "wiz_idx_in", "wiz_fx_in", "wiz_fx_mode", "wiz_regime_in")


def wiz_reset():
    """Clear the wizard's state *and* its widget keys (keyed widgets outlive the state dict)."""
    st.session_state.pop("wiz", None)
    for key in WIZ_INPUT_KEYS:
        st.session_state.pop(key, None)


def wiz_dismissed():
    """Clear the open flag when the user closes the modal with the ✕.

    Dismissing is only a client-side close: `wiz_open` stays True, so the *next* full script
    rerun — editing the portfolio table, nudging any sidebar widget — re-renders the dialog and
    it pops back up unwanted. `st.dialog(on_dismiss=...)` is the only hook that fires on the ✕,
    so the flag has to be cleared here. (Progress itself is reset by the launch button, which
    calls `wiz_reset()` every time.)
    """
    st.session_state["wiz_open"] = False


def _wiz():
    """The wizard's single state dict (one `wiz_reset()` resets the whole flow)."""
    if "wiz" not in st.session_state:
        st.session_state.wiz = {"step": 1}
    return st.session_state.wiz


def _wiz_seed(key, value):
    """Seed a keyed text_input once.

    A widget with `key=` ignores its `value=` argument on every rerun after the first, so the
    only way to fill one programmatically (the "Use this" buttons) is to write session_state
    directly. Seeding here, and writing the key in those buttons, keeps both paths consistent.
    """
    if key not in st.session_state:
        st.session_state[key] = value or ""


def _wiz_span(probe):
    return f"{probe['start'].date()} → {probe['end'].date()}"


def _wiz_show_probe(label, ticker, probe):
    """Render one ticker's identity card: name, currency, history span, row count."""
    st.markdown(f"**{label}: `{ticker}`**")
    if probe.get("long_name"):
        st.caption(probe["long_name"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Currency", probe["currency"] or "unknown")
    c2.metric("History", _wiz_span(probe))
    c3.metric("Rows", f"{probe['n_rows']:,}")
    st.caption(f"Income: {_wiz_income_label(probe)}")
    if not probe["currency"]:
        st.caption("⚠️ Yahoo doesn't report a currency for this symbol — common for indices. "
                   "You can still continue; you'll pick the conversion rate yourself.")


# Plain-language regime labels; the dict keys are what q_hat_verdict / Q_BANDS expect.
WIZ_REGIME_LABELS = {
    "price_index": "The index leaves out dividends the fund collects",
    "same_income": "Both carry the same income, or the asset pays none (gold, commodities)",
}


def _wiz_income_label(probe):
    """Describe what a leg does with income — the fact behind the auto-detected regime."""
    if probe.get("pays_dividends"):
        return "pays dividends out (distributing) — its adjusted prices already include them"
    if (probe.get("quote_type") or "").upper() == "INDEX":
        return "a price index — tracks prices only, dividends excluded"
    if (probe.get("quote_type") or "").upper() in ("FUTURE", "CURRENCY"):
        return "no income exists for this instrument"
    return "pays no dividends out — either accumulating (reinvested) or an asset with no income"


def _wiz_nav(back_to=None, forward=None, forward_label="Continue →", forward_ok=False,
             blocked_reason=None):
    """Back / Continue row. Continue is disabled until this step's gate passes."""
    left, right = st.columns(2)
    if back_to is not None and left.button("← Back", width="stretch", key=f"wiz_back_{back_to}"):
        _wiz()["step"] = back_to
        st.rerun(scope="fragment")
    if right.button(forward_label, width="stretch", type="primary",
                    disabled=not forward_ok, key=f"wiz_fwd_{forward}"):
        _wiz()["step"] = forward
        st.rerun(scope="fragment")
    if not forward_ok and blocked_reason:
        st.caption(blocked_reason)


@st.dialog("🧬 Extend an ETF's price history", width="large", on_dismiss=wiz_dismissed)
def render_extend_wizard():
    w = _wiz()
    cfg = st.session_state.get("wiz_cfg", {})
    yf_interval = YF_INTERVALS[cfg.get("primary", "monthly")]

    st.caption(
        f"Checking every ticker against Yahoo at the **{cfg.get('primary', 'monthly')}** interval. "
        f"Nothing is downloaded to disk until the final step."
    )
    st.progress((w["step"] - 1) / 3, text=" · ".join(
        f"**{s}**" if i + 1 == w["step"] else s for i, s in enumerate(WIZ_STEPS)))
    st.divider()

    # ── STEP 1 — the fund to extend ────────────────────────────────
    if w["step"] == 1:
        st.markdown("#### Which fund do you want to extend?")
        st.caption("The accumulating ETF whose history is too short. e.g. `VWCE.MI`, `SXR8.DE`")
        _wiz_seed("wiz_etf_in", w.get("etf", ""))
        ticker = st.text_input("Fund ticker", placeholder="SXR8.DE",
                               key="wiz_etf_in").strip()
        if st.button("Check fund", key="wiz_etf_check") and ticker:
            with st.spinner(f"Looking up {ticker}…"):
                w["etf"], w["etf_probe"] = ticker, probe_ticker(ticker, yf_interval)
                w.pop("cands", None)
            st.rerun(scope="fragment")

        probe = w.get("etf_probe")
        if probe and w.get("etf") == ticker:
            if probe["ok"]:
                _wiz_show_probe("Fund", ticker, probe)
            else:
                st.error(probe["error"])
        ok = bool(probe and probe["ok"] and w.get("etf") == ticker)
        _wiz_nav(forward=2, forward_ok=ok,
                 blocked_reason="Enter a fund ticker and press **Check fund** to continue.")

    # ── STEP 2 — the index that supplies the older history ─────────
    elif w["step"] == 2:
        etf_probe = w["etf_probe"]
        st.markdown("#### Which index should supply the older history?")
        name = etf_probe.get("long_name") or w["etf"]
        st.info(f"**{w['etf']}** is *{name}* — look for an index tracking that same market.")

        ppy = _WIZ_PPY[cfg.get("primary", "monthly")]
        if "cands" not in w:
            with st.spinner("Searching Yahoo for matching indices, checking history and fit…"):
                w["cands"] = suggest_index_candidates(etf_probe.get("long_name") or w["etf"],
                                                      yf_interval, etf_symbol=w["etf"])
        cands = w.get("cands") or []
        # Rank by what actually matters: a plausible fit first, then merely-usable history.
        rows = [(c, preview_candidate_fit(c["symbol"], w["etf"], yf_interval, ppy)) for c in cands]
        rows.sort(key=lambda cf: (not cf[1]["verdict_ok"], not cf[1]["extends"],
                                  not cf[1]["ok"], -cf[1]["extra_years"]))

        if rows:
            st.markdown("**Candidates found** — each one checked for real history *and* for how "
                        "well it fits your fund:")
            for c, fit in rows:
                pr = c["probe"]
                if not pr["ok"]:
                    st.markdown(f"❌ `{c['symbol']}` — {c['name']}  \n"
                                f"<span style='opacity:.6'>no usable history</span>",
                                unsafe_allow_html=True)
                    continue
                # Usable history, but does it extend, and does the fit hold up?
                if not fit["extends"]:
                    detail = f"⚠️ {fit['reason']}"
                elif fit["ok"]:
                    mark = "✓ plausible" if fit["verdict_ok"] else "✗ implausible"
                    conv = (f"via `{fit['fx_symbol']}`" if fit["fx_symbol"]
                            else "no conversion needed")
                    detail = f"**q̂ {fit['q_hat'] * 100:+.2f}%/yr** {mark} · {conv}"
                else:
                    detail = f"⚠️ {fit['reason']}"
                icon = "✅" if fit["verdict_ok"] else "⚠️"
                col_a, col_b = st.columns([3, 1])
                col_a.markdown(
                    f"{icon} `{c['symbol']}` — {c['name']}  \n"
                    f"<span style='opacity:.7'>{_wiz_span(pr)} · {pr['n_rows']:,} rows</span>  \n"
                    f"<span style='opacity:.85'>{detail}</span>",
                    unsafe_allow_html=True)
                if col_b.button("Use this", key=f"wiz_use_{c['symbol']}", width="stretch"):
                    st.session_state["wiz_idx_in"] = c["symbol"]
                    st.rerun(scope="fragment")

            st.caption(
                "**q̂ is a preview.** Each index was converted into the fund's currency using the "
                "rate pair shown on its row, derived automatically from the two currencies — you "
                "choose or skip that conversion in the next step, which can move the number."
            )
            if not any(f["ok"] for _, f in rows):
                if any(f["extends"] is False and c["probe"]["ok"] for c, f in rows):
                    st.info(f"None of these start before **{etf_probe['start'].date()}**, so none "
                            f"of them can extend {w['etf']}. Try a different search term or enter "
                            f"a ticker manually below.")
                else:
                    st.warning("None of these are usable. Try a different search term, or enter a "
                               "ticker manually below.")

        with st.expander("📖 What is the recovered yield (q̂)?"):
            st.markdown(DESCRIPTIONS["recovered_yield"])
        with st.expander("📖 How to find tickers"):
            st.markdown(DESCRIPTIONS["finding_tickers"])

        _wiz_seed("wiz_idx_in", w.get("index", ""))
        ticker = st.text_input("Index ticker", placeholder="^GSPC",
                               key="wiz_idx_in").strip()
        if st.button("Check index", key="wiz_idx_check") and ticker:
            with st.spinner(f"Looking up {ticker}…"):
                w["index"], w["index_probe"] = ticker, probe_ticker(ticker, yf_interval)
                # A new index leg invalidates both the detected regime and the suggested rate.
                w.pop("regime", None)
                w.pop("fx", None)
                for k in ("wiz_regime_in", "wiz_fx_in", "wiz_fx_mode"):
                    st.session_state.pop(k, None)
            st.rerun(scope="fragment")

        ok, reason = False, "Choose or enter an index ticker, then press **Check index**."
        probe = w.get("index_probe")
        if probe and w.get("index") == ticker:
            if not probe["ok"]:
                st.error(probe["error"])
                reason = "This ticker has no usable history — pick another."
            else:
                _wiz_show_probe("Index", ticker, probe)
                etf_s = fetch_price_series(w["etf"], yf_interval)
                idx_s = fetch_price_series(ticker, yf_interval)
                overlap = idx_s.index.intersection(etf_s.index)
                adds = probe["start"] < etf_probe["start"]
                if len(overlap) < 2:
                    st.error("This index and the fund share fewer than 2 dates, so the missing "
                             "dividends can't be calibrated. They likely trade on different "
                             "calendars or don't overlap at all.")
                    reason = "No usable overlap with the fund."
                elif not adds:
                    st.error(f"This index starts {probe['start'].date()}, which is **not before** "
                             f"the fund's {etf_probe['start'].date()} — it would add no history.")
                    reason = "This index is not longer than the fund."
                else:
                    gained = (etf_probe["start"] - probe["start"]).days / 365.25
                    st.success(f"Good pairing so far: **{len(overlap)} overlapping points** to "
                               f"calibrate on, and about **{gained:.1f} extra years** of history.")
                    fit = preview_candidate_fit(ticker, w["etf"], yf_interval, ppy)
                    if fit["ok"]:
                        conv = (f"assumes `{fit['fx_symbol']}`" if fit["fx_symbol"]
                                else "no conversion needed")
                        (st.info if fit["verdict_ok"] else st.warning)(
                            f"**q̂ {fit['q_hat'] * 100:+.2f}%/yr** — "
                            f"{'plausible' if fit['verdict_ok'] else 'implausible'} for this kind "
                            f"of pairing. *Preview · {conv}; confirmed at the last step.*"
                        )
                    ok = True
        _wiz_nav(back_to=1, forward=3, forward_ok=ok, blocked_reason=reason)

    # ── STEP 3 — currency conversion ───────────────────────────────
    elif w["step"] == 3:
        etf_ccy = w["etf_probe"]["currency"]
        idx_ccy = w["index_probe"]["currency"]
        st.markdown("#### Do the two need a currency conversion?")
        c1, c2 = st.columns(2)
        c1.metric(f"Fund · {w['etf']}", etf_ccy or "unknown")
        c2.metric(f"Index · {w['index']}", idx_ccy or "unknown")

        suggested = suggest_fx_ticker(etf_ccy, idx_ccy)
        same_ccy = bool(etf_ccy and idx_ccy and etf_ccy.upper() == idx_ccy.upper())
        if same_ccy:
            st.success(f"Both are priced in **{etf_ccy}** — no conversion needed.")
        elif suggested:
            st.info(f"The index is in **{idx_ccy}** but the fund is in **{etf_ccy}**, so the index "
                    f"must be converted first — otherwise the exchange-rate drift gets counted as "
                    f"dividends. The rate you need is **`{suggested}`**, already filled in below "
                    f"(it is also the pair the q̂ preview used).")
        else:
            st.warning("Yahoo doesn't report a currency for one of these, so no rate can be "
                       "suggested. If they are already in the same currency, choose *proceed "
                       "without* below.")

        # Default to whichever mode is right for these two currencies; the user can override.
        _wiz_seed("wiz_fx_mode", FX_MODE_NONE if same_ccy else FX_MODE_PAIR)
        modes = [FX_MODE_PAIR, FX_MODE_NONE]
        mode = st.radio("How should the currencies be handled?", options=modes,
                        index=modes.index(st.session_state["wiz_fx_mode"]), key="wiz_fx_mode")

        ok, reason = False, None
        if mode == FX_MODE_NONE:
            w["fx"] = ""
            ok = True
            if not same_ccy:
                st.warning(
                    f"The two are in **different currencies** ({idx_ccy or '?'} vs "
                    f"{etf_ccy or '?'}). Without a conversion the exchange-rate drift is folded "
                    f"into the recovered yield, so q̂ at the next step will not mean what it says."
                )
        else:
            # Pre-filled with the derived pair: this is a *computed* suggestion, not autofill.
            _wiz_seed("wiz_fx_in", w.get("fx") or suggested or "")
            ticker = st.text_input("Exchange-rate ticker", placeholder="EURUSD=X",
                                   key="wiz_fx_in").strip()
            reason = "Press **Check rate** to verify the pair before continuing."
            if st.button("Check rate", key="wiz_fx_check") and ticker:
                with st.spinner(f"Looking up {ticker}…"):
                    w["fx"], w["fx_probe"] = ticker, probe_ticker(ticker, yf_interval)
                st.rerun(scope="fragment")
            probe = w.get("fx_probe")
            if probe and w.get("fx") == ticker and ticker:
                if not probe["ok"]:
                    st.error(probe["error"])
                    reason = "That rate ticker has no usable history."
                else:
                    _wiz_show_probe("Rate", ticker, probe)
                    # The splice back-fills the earliest rate into anything older, which
                    # silently invents a flat exchange rate for those years. Say so.
                    idx_start = w["index_probe"]["start"]
                    if probe["start"] > idx_start:
                        st.warning(
                            f"This rate only goes back to **{probe['start'].date()}**, but the index "
                            f"starts **{idx_start.date()}**. Everything before the rate begins will "
                            f"reuse the earliest rate as a flat value — those early years are an "
                            f"approximation, not a real conversion."
                        )
                    ok = True
        _wiz_nav(back_to=2, forward=4, forward_ok=ok, blocked_reason=reason)

    # ── STEP 4 — recovered yield, then run ─────────────────────────
    else:
        st.markdown("#### Does the pairing hold up?")
        etf_s = fetch_price_series(w["etf"], yf_interval)
        idx_s = fetch_price_series(w["index"], yf_interval)
        fx_s = fetch_price_series(w["fx"], yf_interval) if w.get("fx") else None
        ppy = _WIZ_PPY[cfg.get("primary", "monthly")]

        try:
            meta = synthesize_total_return(idx_s, etf_s, fx_s, ppy)
            q_hat = meta["q_hat"]
            err = None
        except Exception as e:
            meta, q_hat, err = None, None, str(e)

        if err:
            st.error(f"Couldn't calibrate: {err}")
            _wiz_nav(back_to=3, forward=4, forward_ok=False)
            return

        # Which gap is plausible depends on the pairing. Auto-detect from the index leg, but let
        # the user correct it: nothing in the data separates an accumulating equity fund from a
        # gold ETC (both report zero dividends), so a wrong guess must never be a dead end.
        detected = default_q_regime(w["index_probe"])
        if "regime" not in w:
            w["regime"] = detected
        options = list(WIZ_REGIME_LABELS)
        picked = st.radio(
            "What gap should we expect between the index and the fund?",
            options=options,
            index=options.index(w["regime"]),
            format_func=lambda k: WIZ_REGIME_LABELS[k],
            key="wiz_regime_in",
        )
        if picked != w["regime"]:
            w["regime"] = picked
            st.rerun(scope="fragment")
        low, high = Q_BANDS[w["regime"]]
        st.caption(
            f"{'Auto-detected' if w['regime'] == detected else 'Changed from the auto-detected setting'}"
            f" · expected range **{low * 100:+.1f}% to {high * 100:+.1f}%/yr**  \n"
            f"Fund `{w['etf']}` — {_wiz_income_label(w['etf_probe'])}.  \n"
            f"Index `{w['index']}` — {_wiz_income_label(w['index_probe'])}."
        )

        ok, message = q_hat_verdict(q_hat, w["regime"])
        # No `delta=`: st.metric renders a direction arrow for it, and a non-numeric label
        # ("implausible") always draws an *up* arrow — actively misleading beside a negative
        # yield. The coloured verdict box below carries the meaning instead.
        st.metric("Recovered yield (q̂)", f"{q_hat * 100:.2f}%/yr")
        (st.success if ok else st.error)(message)

        st.markdown("**What you'll get**")
        r1, r2, r3 = st.columns(3)
        r1.metric("Extended history", f"{meta['series'].index.min().date()}")
        r2.metric("Reconstructed rows", f"{(meta['series'].index < meta['join_date']).sum():,}")
        r3.metric("Joins the fund", f"{meta['join_date'].date()}")
        st.caption(
            f"Saved as `{w['etf']}_EXT` for the interval(s) you picked "
            f"({', '.join(cfg.get('intervals', []))}). Add that ticker to **My Portfolio** to use it."
        )

        left, right = st.columns(2)
        if left.button("← Back", width="stretch", key="wiz_back_3"):
            w["step"] = 3
            st.rerun(scope="fragment")
        if right.button("🧬  Reconstruct", width="stretch", type="primary", disabled=not ok,
                        key="wiz_run"):
            # Queue the job and close: the progress UI must live in the main script, because
            # elements created outside a dialog accumulate across the dialog's own reruns.
            st.session_state["recon_job"] = {
                "index": w["index"], "etf": w["etf"], "fx": w.get("fx", ""),
                "regime": w["regime"],   # the runner's log warning must agree with this verdict
            }
            wiz_reset()
            st.session_state["wiz_open"] = False
            st.rerun(scope="app")
        if not ok:
            st.caption("Reconstruct stays disabled while the recovered yield is implausible — "
                       "go back and pair the fund with an index for the same market.")


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
    TITLE_Y  = 14            # gauge title baseline
    LABEL_Y  = 36            # needle value-label baseline (sits a clear row below the title)
    BAR_Y    = LABEL_Y + 17  # bar top: the value label (at BAR_Y-17) + needle triangle live above it
    TICK_Y1  = BAR_Y + BAR_H + 4
    TICK_Y2  = TICK_Y1 + 6
    TICK_LBL = TICK_Y2 + 11
    DETAIL_Y = TICK_LBL + 22
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
        if subdf.empty:  # empty CSV / fully excluded by the date filter → skip gracefully
            with st.expander(f"{ticker}", expanded=False):
                st.caption("No data available for this asset in the selected window.")
            continue
        # Deflate the price level once; every figure below (simple/calendar returns, CAGR, annualised
        # metrics, cumulative chart, look-back tables) then reads real prices with no further changes.
        if real_terms:
            subdf["adj close"] = subdf["adj close"] / real_deflator(subdf.index, annual_inflation)

        with st.expander(f"{ticker}", expanded=False):
            data_start = subdf.index[0]
            end_date = subdf.index[-1]

            # Look-back windows that fully fit this asset's history (+ a true full-history row);
            # windows longer than its life are omitted, not silently truncated and mislabelled.
            # Computed once and reused by the simple-return, CAGR and annualised-metrics tables.
            year_windows = _lookback_year_windows(end_date, data_start)
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
            simple_windows += year_windows

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

            cagr_rows = []
            for label, sd in year_windows:
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
                help="Arithmetic mean of per-period returns × N — the *expected* return used on the "
                     "efficient-frontier chart and to compare portfolios. The realized compound growth "
                     "differs (volatility drag lowers it ≈σ²/2, intra-period compounding raises it); for "
                     "that, see the CAGR table just above.",
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
            for label, sd in year_windows:
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
    st.subheader(f"Individual Assets — Cumulative {rolling_window_years}-Year Return")
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker in tickers:
        ax.plot(rolling_returns["date"], rolling_returns[ticker], lw=1, label=ticker)
    ax.set_title(f"Cumulative {rolling_window_years}-Year Return")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Cumulative {rolling_window_years}-Year Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.legend(title="Asset")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"Each point is the **total** return over the preceding {rolling_window_years} years "
        f"(`P_t / P_{{t−{rolling_window_years}y}} − 1`), not a per-year rate — so a longer window "
        "naturally shows a bigger figure purely from compounding more years. **Windows of "
        "different lengths are not directly comparable**; for a per-year figure see the CAGR in §2."
    )
    st.divider()



# ─────────────────────────────────────────────────────────────────
# SECTION 4 — RETURNS & STATISTICS
# ─────────────────────────────────────────────────────────────────

def render_returns_statistics(portfolio_returns_simple, portfolio_mean_returns,
                               portfolio_cov_matrix, tickers, annualisation_factor,
                               risk_free_rate, mixed_calendar=False, seven_day_tickers=None):
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
    _mixed_calendar_note(mixed_calendar, seven_day_tickers)

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
                                    real_terms=False, annual_inflation=0.0,
                                    mixed_calendar=False, seven_day_tickers=None, cgt_rate=0.0):
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
        + (["real_returns"] if real_terms else [])
        + (["capital_gains_tax"] if cgt_rate > 0 else []),
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

    # ── Optional capital-gains tax / net-liquidation overlay (feature T; off when cgt_rate == 0) ──
    netliq = None
    if cgt_rate > 0:
        _wtax, netliq = rebalanced_value_aftertax(merged_df, tickers, weights,
                                                  rebalance_every_periods, cgt_rate)
        if real_terms:  # deflate the after-tax level once, exactly like `value` above
            netliq = netliq / real_deflator(netliq.index, annual_inflation)

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
    ax_cum.plot(value.index, value - 1.0, lw=2.2, color="black", label="Portfolio (pre-tax)")
    if netliq is not None:
        ax_cum.plot(netliq.index, netliq - 1.0, lw=2.0, color="#c0392b", ls="--",
                    label="Portfolio (after-tax net-liq)")
    ax_cum.axhline(0, color="#999", lw=0.8)
    ax_cum.set_title(f"Cumulative Return Since Start{real_sfx}")
    ax_cum.set_xlabel("Date")
    ax_cum.set_ylabel(f"Cumulative Return{real_sfx}")
    ax_cum.yaxis.set_major_formatter(pct)
    ax_cum.legend(title="Holding", ncol=2)
    ax_cum.grid(True, alpha=0.3)
    st.pyplot(fig_cum)
    plt.close(fig_cum)
    if netliq is not None:
        nl_cagr = (netliq.iloc[-1] / netliq.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
        st.caption(
            f"💸 **After-tax net-liquidation** (red dashed): {cgt_rate:.0%} capital-gains tax realized "
            f"on net gains at each rebalance, plus tax owed on unrealized gains if sold today. "
            f"Net-liq CAGR **{nl_cagr:.2%}** vs pre-tax **{cagr:.2%}** — a tax drag of "
            f"**{(cagr - nl_cagr) * 100:.2f} pp/yr**. Realized losses net against gains within each "
            "rebalance (no carry-forward)."
        )

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
    st.subheader(f"Portfolio — Cumulative {rolling_window_years}-Year Return (Buy-and-Hold)")
    if len(value) > window_periods:
        roll = (value / value.shift(window_periods) - 1.0).dropna()
        fig_rr, ax_rr = plt.subplots(figsize=(12, 5))
        ax_rr.plot(roll.index, roll.values, lw=1.5, color="black", label="Portfolio")
        ax_rr.set_title(f"Cumulative {rolling_window_years}-Year Return{real_sfx}")
        ax_rr.set_xlabel("Date")
        ax_rr.set_ylabel(f"Cumulative {rolling_window_years}-Year Return{real_sfx}")
        ax_rr.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax_rr.legend()
        ax_rr.grid(True, alpha=0.3)
        st.pyplot(fig_rr)
        plt.close(fig_rr)
        st.caption(
            f"Each point is the portfolio's **total** return over the preceding "
            f"{rolling_window_years} years, not a per-year rate — longer windows show bigger "
            "figures from compounding more years and aren't comparable across window lengths "
            "(for a per-year figure see CAGR above)."
        )
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
    _mixed_calendar_note(mixed_calendar, seven_day_tickers)

    # ── Tail risk & return distribution (merged from the former standalone VaR section) ──
    render_tail_risk(bh_ret, alpha)

    st.divider()


# ─────────────────────────────────────────────────────────────────
# SECTION 6 — MONTE CARLO EFFICIENT FRONTIER (VOL-BASED)
# ─────────────────────────────────────────────────────────────────

def render_monte_carlo(portfolio_returns_simple, portfolio_mean_returns, portfolio_cov_matrix,
                        tickers, annualisation_factor, risk_free_rate, num_portfolios, eps,
                        custom_target_ret, custom_target_vol, my_portfolio_allocation, alpha,
                        real_terms=False, mixed_calendar=False, seven_day_tickers=None):
    st.header("7. Monte Carlo Efficient Frontier Portfolio Optimization")
    st.caption("⚖️ Rebalancing: **per period** — the frontier and all return/volatility figures here "
               "assume per-period rebalancing (the basis MPT optimization requires), independent of the "
               "sidebar Rebalancing-frequency setting (which governs §6).")
    _mixed_calendar_note(mixed_calendar, seven_day_tickers)
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
                     my_portfolio_allocation, alpha, real_terms=False,
                     mixed_calendar=False, seven_day_tickers=None):
    st.header("8. Scipy Efficient Frontier Portfolio Optimization")
    st.caption("⚖️ Rebalancing: **per period** — the frontier and all return/volatility figures here "
               "assume per-period rebalancing (the basis MPT optimization requires), independent of the "
               "sidebar Rebalancing-frequency setting (which governs §6).")
    _mixed_calendar_note(mixed_calendar, seven_day_tickers)
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
