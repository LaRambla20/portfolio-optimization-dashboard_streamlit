"""
Tests for portfolio_calculations.real_deflator / to_real (real-terms toggle).

Deterministic, no network. Pins the properties the inflation feature relies on:
toggle-off is an exact no-op, a constant rate shifts mean returns down by ~pi while
leaving volatility ~unchanged, and real drawdowns are at least as deep as nominal.

Run:  .venv\\Scripts\\python tests\\test_real_terms.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from portfolio_calculations import real_deflator, to_real, max_drawdown  # noqa: E402


def _monthly(prices):
    idx = pd.date_range("2000-01-01", periods=len(prices), freq="MS")
    return pd.Series(prices, index=idx)


def test_zero_rate_is_noop():
    s = _monthly([100.0, 101.0, 99.0, 105.0, 110.0])
    d = real_deflator(s.index, 0.0)
    assert np.allclose(d.values, 1.0), d.values
    assert np.allclose(to_real(s, 0.0).values, s.values), "0% inflation must not change the series"
    print("  0% inflation is an exact no-op  OK")


def test_deflator_grows_with_calendar_time():
    # One year apart -> exactly (1+pi); D_0 = 1.
    idx = pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"])
    d = real_deflator(idx, 0.10)
    assert abs(d.iloc[0] - 1.0) < 1e-12
    # 366 days (leap year 2000) / 365.25 ~ 1.002 years, so allow a small calendar tolerance.
    assert abs(d.iloc[1] - 1.10) < 2e-3, d.iloc[1]
    assert d.iloc[2] > d.iloc[1] > d.iloc[0]
    print(f"  deflator grows with calendar time: D=[{d.iloc[0]:.4f}, {d.iloc[1]:.4f}, {d.iloc[2]:.4f}]  OK")


def test_mean_shifts_but_vol_unchanged():
    # A noisy monthly nominal series over several years.
    rng = np.random.default_rng(0)
    n = 240
    nom_ret = 0.008 + 0.04 * rng.standard_normal(n)        # ~9.6%/yr drift, monthly vol 4%
    nom_level = _monthly(np.concatenate([[100.0], 100.0 * np.cumprod(1 + nom_ret)]))

    pi = 0.03
    real_level = to_real(nom_level, pi)
    r_nom = nom_level.pct_change().dropna()
    r_real = real_level.pct_change().dropna()

    # Mean drops by ~ the per-period inflation (pi/12 for monthly), via the Fisher relation.
    pi_period = (1 + pi) ** (1 / 12) - 1
    assert abs((r_nom.mean() - r_real.mean()) - pi_period) < 5e-4, (r_nom.mean(), r_real.mean())
    # Volatility is ~unchanged: subtracting a near-constant per-period offset can't change the spread.
    # (Calendar-time deflation isn't *exactly* constant per period — months differ in length — so a
    # tiny residual remains; assert it's negligible relative to the ~4%/period volatility.)
    assert abs(r_nom.std() - r_real.std()) / r_nom.std() < 0.01, (r_nom.std(), r_real.std())
    print(f"  mean shifts by ~pi/12 ({pi_period:.5f}); vol unchanged "
          f"(d_std={abs(r_nom.std() - r_real.std()):.2e})  OK")


def test_real_drawdown_at_least_as_deep():
    # A run-up then a long flat-nominal plateau: real value erodes, so real DD must exceed nominal.
    nom_level = _monthly([100, 130, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160])
    r_nom = nom_level.pct_change().dropna()
    r_real = to_real(nom_level, 0.05).pct_change().dropna()
    mdd_nom = max_drawdown(r_nom)
    mdd_real = max_drawdown(r_real)
    assert mdd_real >= mdd_nom - 1e-12, (mdd_nom, mdd_real)
    assert mdd_real > mdd_nom, "flat-nominal plateau should create a real drawdown"
    print(f"  real drawdown deeper than nominal: {mdd_nom:.4f} -> {mdd_real:.4f}  OK")


if __name__ == "__main__":
    print("real_deflator / to_real")
    test_zero_rate_is_noop()
    test_deflator_grows_with_calendar_time()
    test_mean_shifts_but_vol_unchanged()
    test_real_drawdown_at_least_as_deep()
    print("All tests passed.")
