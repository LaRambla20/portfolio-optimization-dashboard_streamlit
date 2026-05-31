"""
Portfolio calculation functions for Efficient Frontier dashboard.
All functions are pure math operations with no side effects.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
from scipy import stats


def portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor):
    # Textbook MPT annualisation: the arithmetic mean scales linearly with the horizon
    # (mu_ann = mu * T), consistent with variance scaling linearly (sigma_ann = sigma * sqrt(T)).
    # This keeps the return and risk axes on the same (additive) convention. Note this is an
    # expected/arithmetic annual return, distinct from the geometric CAGR shown in Section 2.
    average = np.sum(mean_returns * weights) * annualisation_factor
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(annualisation_factor)
    return std, average


def portfolio_annualised_performance_VaR(weights, mean_returns, cov_matrix, alpha, annualisation_factor):
    std, average = portfolio_annualised_performance(weights, mean_returns, cov_matrix, annualisation_factor)
    var = std * abs(stats.norm.ppf(1 - alpha)) - average  # positive = loss amount
    return std, average, var


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


def random_portfolios_VaR(num_portfolios, mean_returns, cov_matrix, risk_free_rate, alpha, annualisation_factor):
    # Returns 4 rows: [std, return, sharpe, VaR]. CVaR is computed historically at display
    # time (see display_portfolio_cards) so the whole app uses one CVaR definition and sign.
    results = np.zeros((4, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        # Dirichlet(1,...,1) gives uniform coverage of the weight simplex (see random_portfolios).
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return, portfolio_var = portfolio_annualised_performance_VaR(
            weights, mean_returns, cov_matrix, alpha, annualisation_factor
        )
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
        results[3, i] = portfolio_var
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


def portfolio_info_dict(name, std_dev, ret, sharpe, alloc_df, var=None):
    d = {
        "Portfolio": name,
        "Ann. Return": f"{ret:.2%}",
        "Ann. Volatility": f"{std_dev:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
    }
    if var is not None:
        d["Value at Risk"] = f"{var:.2%}"
    return d, alloc_df

def compute_portfolio_rolling_returns(weights, returns_simple, window_periods):
    """Portfolio rolling return: (portfolio_value[t]/portfolio_value[t-window_periods])-1."""
    portfolio_ret = returns_simple.dot(weights)
    portfolio_value = (1 + portfolio_ret).cumprod()
    return portfolio_value / portfolio_value.shift(window_periods) - 1
