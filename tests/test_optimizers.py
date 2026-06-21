"""
Tests for the SLSQP frontier optimizers in portfolio_calculations.

Deterministic, no network, no running app. Pins the correctness properties that
would break if the shared `_optimize` helper miswired an objective or constraint:
fully-invested long-only weights, min-variance is the volatility floor, and the
two `efficient_*` solvers actually hit their target return / volatility.

Run:  .venv\\Scripts\\python tests\\test_optimizers.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from portfolio_calculations import (  # noqa: E402
    portfolio_annualised_performance,
    portfolio_annualised_performance_sortino,
    max_sharpe_ratio,
    minimize_volatility,
    maximize_return,
    efficient_return,
    efficient_volatility,
    max_sortino_ratio,
)

# A 3-asset toy problem: distinct expected returns, mild correlations.
MEAN = np.array([0.0004, 0.0006, 0.0009])          # per-period
COV = np.array([
    [0.00010, 0.00002, 0.00001],
    [0.00002, 0.00020, 0.00004],
    [0.00001, 0.00004, 0.00040],
])
N = 252  # annualisation factor (daily)
RF = 0.02

# A per-period returns sample (Sortino needs the series, not just mean/cov),
# drawn from the same toy mean/cov so it stays consistent. Seeded → deterministic.
_rng = np.random.default_rng(777)
RETURNS = pd.DataFrame(_rng.multivariate_normal(MEAN, COV, size=2000),
                       columns=["A", "B", "C"])


def _perf(w):
    return portfolio_annualised_performance(w, MEAN, COV, N)  # (std, ret)


def _assert_simplex(w):
    assert abs(w.sum() - 1.0) < 1e-6, w
    assert (w >= -1e-9).all() and (w <= 1.0 + 1e-9).all(), w


def test_weights_on_simplex():
    for res in (max_sharpe_ratio(MEAN, COV, RF, N),
                minimize_volatility(MEAN, COV, N),
                maximize_return(MEAN, COV, N)):
        _assert_simplex(res.x)
    print("  all solvers return fully-invested long-only weights  OK")


def test_min_variance_is_the_floor():
    mv = minimize_volatility(MEAN, COV, N).x
    sharpe = max_sharpe_ratio(MEAN, COV, RF, N).x
    mr = maximize_return(MEAN, COV, N).x
    v_mv, v_sharpe, v_mr = _perf(mv)[0], _perf(sharpe)[0], _perf(mr)[0]
    assert v_mv <= v_sharpe + 1e-9, (v_mv, v_sharpe)
    assert v_mv <= v_mr + 1e-9, (v_mv, v_mr)
    # maximize_return lands on the single highest-return asset (asset 2 here).
    assert mr.argmax() == 2 and mr[2] > 0.99, mr
    print(f"  min-vol {v_mv:.4f} <= sharpe {v_sharpe:.4f}, <= max-ret {v_mr:.4f}  OK")


def test_efficient_return_hits_target():
    target = 0.18  # annualised, between min-vol and max-ret returns
    w = efficient_return(MEAN, COV, target, N).x
    _assert_simplex(w)
    assert abs(_perf(w)[1] - target) < 1e-6, _perf(w)[1]
    print(f"  efficient_return hits target ret {target:.4f}  OK")


def test_efficient_volatility_hits_target():
    target = 0.20  # annualised vol, achievable on the frontier
    w = efficient_volatility(MEAN, COV, target, N).x
    _assert_simplex(w)
    assert abs(_perf(w)[0] - target) < 1e-6, _perf(w)[0]
    print(f"  efficient_volatility hits target vol {target:.4f}  OK")


def test_max_sortino_beats_equal_weight():
    w = max_sortino_ratio(MEAN, COV, RETURNS, RF, N).x
    _assert_simplex(w)
    n = len(MEAN)
    eq = np.full(n, 1.0 / n)
    s_opt = portfolio_annualised_performance_sortino(w, MEAN, COV, RETURNS, RF, N)[2]
    s_eq = portfolio_annualised_performance_sortino(eq, MEAN, COV, RETURNS, RF, N)[2]
    assert s_opt >= s_eq - 1e-9, (s_opt, s_eq)
    print(f"  max-sortino {s_opt:.4f} >= equal-weight {s_eq:.4f}  OK")


if __name__ == "__main__":
    print("SLSQP frontier optimizers")
    test_weights_on_simplex()
    test_min_variance_is_the_floor()
    test_efficient_return_hits_target()
    test_efficient_volatility_hits_target()
    test_max_sortino_beats_equal_weight()
    print("All tests passed.")
