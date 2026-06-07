"""
Portfolio calculation functions for Efficient Frontier dashboard.
All functions are pure math operations with no side effects.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco


def portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor):
    # Textbook MPT annualisation: the arithmetic mean scales linearly with the horizon
    # (mu_ann = mu * T), consistent with variance scaling linearly (sigma_ann = sigma * sqrt(T)).
    # This keeps the return and risk axes on the same (additive) convention. Note this is an
    # expected/arithmetic annual return, distinct from the geometric CAGR shown in Section 2.
    average = np.sum(mean_returns * weights) * annualisation_factor
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(annualisation_factor)
    return std, average


def real_deflator(date_index, annual_inflation):
    """Cumulative inflation deflator for converting a nominal level series to real terms.

    ``D_t = (1 + π)^(years elapsed since the first date)`` with ``D_0 = 1``, indexed by the same
    dates. Dividing a nominal price/value series by this expresses it in the purchasing power of the
    start date (constant-currency, "real" terms). ``annual_inflation`` is a decimal (0.02 = 2%/yr);
    a rate of 0 returns all ones (a no-op), so callers can deflate unconditionally if they wish.

    Calendar-time based (``.days / 365.25``), not row count, so it is robust to irregular spacing in
    the merged inner-join (mixed trading calendars, gaps). Subtracting a *constant* rate this way
    leaves variances/covariances ~unchanged — only means/levels shift — so volatility, correlation
    and the efficient-frontier weights are inflation-invariant while CAGR drops and drawdowns deepen.
    """
    years = (date_index - date_index[0]).days / 365.25
    return pd.Series((1.0 + annual_inflation) ** years, index=date_index)


def to_real(level_series, annual_inflation):
    """Deflate a nominal price/value *level* Series to real terms via ``real_deflator``.

    Thin convenience wrapper: ``level_series / real_deflator(level_series.index, annual_inflation)``.
    Pass ``annual_inflation=0`` for a no-op. Apply to *levels* (prices, portfolio value), not to a
    return series — deflating the level and then differencing yields the correct real returns.
    """
    return level_series / real_deflator(level_series.index, annual_inflation)


def cvar(returns, alpha=0.05):
    """Historical CVaR (Expected Shortfall) from a return series."""
    var_threshold = returns.quantile(alpha)
    return -returns[returns <= var_threshold].mean()


def max_drawdown(returns):
    """Maximum drawdown from a return series. Returns positive decimal."""
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return abs(drawdown.min())


def portfolio_cvar(weights, returns, alpha=0.05):
    """CVaR for a portfolio given weights and return dataframe."""
    portfolio_returns = returns.dot(weights)
    return cvar(portfolio_returns, alpha)


def portfolio_max_drawdown(weights, returns):
    """Max drawdown for a portfolio given weights and return dataframe."""
    portfolio_returns = returns.dot(weights)
    return max_drawdown(portfolio_returns)


def portfolio_downside_deviation(weights, returns, annualisation_factor, risk_free_rate):
    # Downside deviation measured below the (per-period) risk-free rate, so the Sortino MAR
    # matches its numerator (return - risk_free_rate). r_f is de-annualised linearly, consistent
    # with the mu * N return-annualisation convention.
    port_returns = returns.dot(weights)
    mar_period = risk_free_rate / annualisation_factor
    downside = np.minimum(port_returns - mar_period, 0.0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(annualisation_factor)


def portfolio_annualised_performance_sortino(weights, mean_returns, cov_matrix, returns,
                                             risk_free_rate, annualisation_factor):
    std, average = portfolio_annualised_performance(
        weights, mean_returns, cov_matrix, annualisation_factor)
    dd = portfolio_downside_deviation(weights, returns, annualisation_factor, risk_free_rate)
    sortino = (average - risk_free_rate) / dd if dd > 0 else np.nan
    return std, average, sortino


def neg_sortino_ratio(weights, mean_returns, cov_matrix, returns,
                      risk_free_rate, annualisation_factor):
    _, p_ret, sortino = portfolio_annualised_performance_sortino(
        weights, mean_returns, cov_matrix, returns,
        risk_free_rate, annualisation_factor)
    return -sortino if not np.isnan(sortino) else 0.0


def max_sortino_ratio(mean_returns, cov_matrix, returns,
                      risk_free_rate, annualisation_factor):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, returns, risk_free_rate, annualisation_factor)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    result = sco.minimize(neg_sortino_ratio, num_assets * [1.0 / num_assets], args=args,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate, annualisation_factor):
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        # Dirichlet(1,...,1) is uniform over the weight simplex; normalising raw uniforms is not,
        # and would cluster near equal weights and under-sample concentrated (corner) portfolios.
        weights = np.random.dirichlet(np.ones(mean_returns.shape[0]))
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(
            weights, mean_returns, cov_matrix, annualisation_factor
        )
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record


def random_portfolios_sortino(num_portfolios, mean_returns, cov_matrix, returns,
                              risk_free_rate, annualisation_factor):
    results = np.zeros((5, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        # Dirichlet(1,...,1) is uniform over the weight simplex; normalising raw uniforms is not,
        # and would cluster near equal weights and under-sample concentrated (corner) portfolios.
        weights = np.random.dirichlet(np.ones(mean_returns.shape[0]))
        weights_record.append(weights)
        std, ret, sortino = portfolio_annualised_performance_sortino(
            weights, mean_returns, cov_matrix, returns,
            risk_free_rate, annualisation_factor)
        dd = portfolio_downside_deviation(weights, returns, annualisation_factor, risk_free_rate)
        results[0, i] = std
        results[1, i] = ret
        results[2, i] = (ret - risk_free_rate) / std if std > 0 else 0.0
        results[3, i] = sortino if not np.isnan(sortino) else 0.0
        results[4, i] = dd
    return results, weights_record


def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate, annualisation_factor):
    p_std_dev, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor)
    return -(p_ret - risk_free_rate) / p_std_dev


def portfolio_volatility_fn(weights, mean_returns, cov_matrix, annualisation_factor):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor)[0]


def neg_portfolio_return(weights, mean_returns, cov_matrix, annualisation_factor):
    return -(portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor)[1])


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate, annualisation_factor):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate, annualisation_factor)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets * [1.0 / num_assets], args=args,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def minimize_volatility(mean_returns, cov_matrix, annualisation_factor):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, annualisation_factor)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    result = sco.minimize(portfolio_volatility_fn, num_assets * [1.0 / num_assets], args=args,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def maximize_return(mean_returns, cov_matrix, annualisation_factor):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, annualisation_factor)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    result = sco.minimize(neg_portfolio_return, num_assets * [1.0 / num_assets], args=args,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def efficient_return(mean_returns, cov_matrix, target, annualisation_factor):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, annualisation_factor)
    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor)[1]
    constraints = (
        {"type": "eq", "fun": lambda x: portfolio_return(x) - target},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    )
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    result = sco.minimize(portfolio_volatility_fn, num_assets * [1.0 / num_assets], args=args,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def efficient_volatility(mean_returns, cov_matrix, target, annualisation_factor):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, annualisation_factor)
    def pv(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor)[0]
    constraints = (
        {"type": "eq", "fun": lambda x: pv(x) - target},
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    )
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    result = sco.minimize(neg_portfolio_return, num_assets * [1.0 / num_assets], args=args,
                          method="SLSQP", bounds=bounds, constraints=constraints)
    return result


def efficient_frontier_fn(mean_returns, cov_matrix, returns_range, annualisation_factor):
    return [efficient_return(mean_returns, cov_matrix, ret, annualisation_factor) for ret in returns_range]


def make_allocation_df(weights, tickers):
    alloc = pd.DataFrame(weights, index=tickers, columns=["allocation"])
    alloc["allocation"] = [round(w * 100, 2) for w in alloc["allocation"]]
    return alloc.T


# ─────────────────────────────────────────────────────────────────
# REBALANCED PORTFOLIO VALUE SERIES (Input Portfolio Analysis §6, incl. its tail-risk subsection)
# ─────────────────────────────────────────────────────────────────
# A single value series for a portfolio held at target weights and periodically rebalanced. The
# two optimization sections (§7/§8) instead use the closed-form per-period-rebalanced MPT basis
# (mean/cov), which is *not* this series — so the same portfolio's realised return/risk/drawdown
# can legitimately differ between §6 and §7/§8. The cadence is chosen in the sidebar.

def rebalanced_value_series(merged_df, tickers, weights, rebalance_every_periods=None):
    """Normalized portfolio value (V_0 = 1) under periodic rebalancing to target `weights`.

    The portfolio is bought at `weights` and held; the mix drifts with prices, and every
    `rebalance_every_periods` rows it is reset back to the target weights. Between resets it is a
    buy-and-hold of the holdings set at the last reset:

        V_t = V_r * sum_i w_i * P_it / P_ir   (r = most recent reset row, V_r carried forward)

    so the series is continuous across resets (no spurious flat period). Scale-invariant — only
    the weight proportions matter. Returns a pd.Series indexed by date.

    Special cases (used to reproduce the app's two historical bases exactly):
      * rebalance_every_periods=None (or >= len) → never reset → buy-and-hold,
        V_t = sum_i w_i * P_it / P_i0.
      * rebalance_every_periods=1 → reset every period → V_t = prod_s (1 + sum_i w_i r_is),
        i.e. (1 + returns.dot(weights)).cumprod() — the constant-weight rebalanced series.
    """
    prices = merged_df.set_index("date")[list(tickers)]
    w = np.asarray(weights, dtype=np.float64)
    pv = prices.to_numpy(dtype=np.float64)
    n = len(prices)
    value = pd.Series(index=prices.index, dtype=np.float64)
    if n == 0:
        return value
    value.iloc[0] = 1.0
    if rebalance_every_periods is None or rebalance_every_periods < 1:
        resets = [0]
    else:
        resets = list(range(0, n, int(rebalance_every_periods)))
    # Each reset row r seeds a buy-and-hold segment (ref prices = P_ir, carried value V_r) running
    # up to and including the next reset row, which the following segment then re-seeds from.
    for si, r in enumerate(resets):
        r_next = resets[si + 1] if si + 1 < len(resets) else n - 1
        if r_next <= r:
            continue
        carry = value.iloc[r]                      # set by the previous segment (or 1.0 at r=0)
        seg = pv[r + 1:r_next + 1] / pv[r]         # (m, A) price ratios vs the reset reference
        value.iloc[r + 1:r_next + 1] = carry * seg.dot(w)
    return value


def buy_and_hold_value_series(merged_df, tickers, weights):
    """Buy-and-hold (never-rebalanced) value series — thin wrapper over `rebalanced_value_series`.

    Each asset is bought at its first price and held, so the mix drifts with prices:
    V_t = sum_i w_i * P_it / P_i0, V_0 = 1.
    """
    return rebalanced_value_series(merged_df, tickers, weights, rebalance_every_periods=None)


def underwater_episodes(value):
    """Decompose a value series into drawdown episodes.

    An episode opens the first period value dips below its running peak and closes the period it
    regains that peak. Returns a list of dicts with peak/trough dates+values and recovery date
    (None if still underwater at the end of the data).
    """
    episodes = []
    running_peak = -np.inf
    running_peak_date = None
    in_dd = False
    peak_date = peak_val = trough_date = trough_val = None
    for d, v in value.items():
        if v >= running_peak:
            running_peak = v
            running_peak_date = d
            if in_dd:  # value has regained the prior peak — episode recovers here
                episodes.append({"peak_date": peak_date, "peak_val": peak_val,
                                 "trough_date": trough_date, "trough_val": trough_val,
                                 "recovery_date": d})
                in_dd = False
        else:
            if not in_dd:
                in_dd = True
                peak_date, peak_val = running_peak_date, running_peak
                trough_date, trough_val = d, v
            elif v < trough_val:
                trough_date, trough_val = d, v
    if in_dd:
        episodes.append({"peak_date": peak_date, "peak_val": peak_val,
                         "trough_date": trough_date, "trough_val": trough_val,
                         "recovery_date": None})
    return episodes


def deepest_drawdown_episode(value):
    """The episode containing the largest peak-to-trough drop (None if value never falls)."""
    episodes = underwater_episodes(value)
    if not episodes:
        return None
    return min(episodes, key=lambda e: e["trough_val"] / e["peak_val"] - 1.0)


def longest_underwater_episode(value):
    """The episode with the most calendar days from peak to recovery (or to the last date if
    still underwater). Returns (episode, days, ongoing) or None if value never falls."""
    episodes = underwater_episodes(value)
    if not episodes:
        return None
    last_date = value.index[-1]

    def span_days(e):
        end = e["recovery_date"] if e["recovery_date"] is not None else last_date
        return (end - e["peak_date"]).days

    best = max(episodes, key=span_days)
    return best, span_days(best), best["recovery_date"] is None


def downside_deviation_series(returns, annualisation_factor, risk_free_rate):
    """Annualised downside deviation of a return *series* (below the per-period risk-free MAR).

    Same definition as portfolio_downside_deviation, but for an already-built return series
    (e.g. the buy-and-hold portfolio) rather than weights × a returns dataframe.
    """
    mar_period = risk_free_rate / annualisation_factor
    downside = np.minimum(returns - mar_period, 0.0)
    return np.sqrt(np.mean(downside ** 2)) * np.sqrt(annualisation_factor)
