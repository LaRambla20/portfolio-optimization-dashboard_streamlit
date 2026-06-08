"""
Tests for ui_components._geometric_annual_return (Issue 6) — the compound (CAGR)
figure shown beside the arithmetic mean×N on the §7/§8 portfolio cards.

Deterministic, no network. Confirms the arithmetic-vs-geometric relationship
(intra-period compounding vs volatility drag) and the NaN guards.

Run:  .venv\\Scripts\\python tests\\test_geometric_return.py
"""

import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from ui_components import _geometric_annual_return  # noqa: E402

N = 12  # monthly periods per year


def test_zero_vol_compounds_above_arithmetic():
    # Constant positive per-period return: geometric = (1+r)^N - 1 exceeds the linear r*N,
    # because the linear mean omits intra-period compounding.
    r = pd.Series([0.01] * 240)
    geo = _geometric_annual_return(r, N)
    arith = r.mean() * N
    assert geo > arith, (geo, arith)
    assert abs(geo - (1.01 ** 12 - 1)) < 1e-12, geo
    print(f"  zero-vol: geo={geo:.4%} > arith={arith:.4%} (intra-period compounding)  OK")


def test_high_vol_drag_below_arithmetic():
    rng = np.random.default_rng(0)
    r = pd.Series(0.008 + 0.06 * rng.standard_normal(2400))   # high monthly volatility
    geo = _geometric_annual_return(r, N)
    arith = r.mean() * N
    assert geo < arith, (geo, arith)
    # The gap is on the order of the annualised variance / 2 (volatility drag).
    half_var_ann = (r.std() ** 2) * N / 2.0
    gap = arith - geo
    assert 0.3 * half_var_ann < gap < 2.0 * half_var_ann, (gap, half_var_ann)
    print(f"  high-vol: geo={geo:.4%} < arith={arith:.4%}, gap={gap:.4%} ~ sigma^2/2={half_var_ann:.4%}  OK")


def test_nan_guards():
    # Empty series -> NaN.
    assert math.isnan(_geometric_annual_return(pd.Series([], dtype=float), N))
    # A -100% period wipes out wealth (cum == 0) -> NaN, never a complex/invalid power.
    assert math.isnan(_geometric_annual_return(pd.Series([-1.0, 0.2]), N))
    # Cumulative wealth driven negative -> NaN.
    assert math.isnan(_geometric_annual_return(pd.Series([0.1, -1.5]), N))
    print("  NaN guards: empty + wealth<=0  OK")


if __name__ == "__main__":
    print("_geometric_annual_return")
    test_zero_vol_compounds_above_arithmetic()
    test_high_vol_drag_below_arithmetic()
    test_nan_guards()
    print("All tests passed.")
