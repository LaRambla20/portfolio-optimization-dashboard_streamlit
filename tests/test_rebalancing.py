"""
Tests for portfolio_calculations.rebalanced_value_series.

Deterministic, no network. Pins the two equivalences the feature relies on
(Never == buy-and-hold, Every period == constant-weight compounded) plus a
hand-checked periodic-rebalance case verifying the reset and continuity.

Run:  .venv\\Scripts\\python tests\\test_rebalancing.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from portfolio_calculations import (  # noqa: E402
    rebalanced_value_series,
    buy_and_hold_value_series,
)


def _merged(prices_by_ticker):
    """Build a merged_df (date column + one column per ticker) from a dict of price lists."""
    n = len(next(iter(prices_by_ticker.values())))
    df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=n, freq="MS")})
    for t, p in prices_by_ticker.items():
        df[t] = p
    return df


# A small 2-asset book used across the cases.
PRICES = {
    "A": [100.0, 110.0, 121.0, 121.0, 133.1],
    "B": [100.0, 100.0, 100.0, 120.0, 120.0],
}
TICKERS = ["A", "B"]
W = np.array([0.5, 0.5])


def test_never_equals_buy_and_hold():
    merged = _merged(PRICES)
    never = rebalanced_value_series(merged, TICKERS, W, rebalance_every_periods=None)
    bh = buy_and_hold_value_series(merged, TICKERS, W)
    # Closed-form buy-and-hold: V_t = sum_i w_i * P_it / P_i0
    prices = merged.set_index("date")[TICKERS]
    closed = (prices / prices.iloc[0]).mul(W, axis=1).sum(axis=1)
    assert np.allclose(never.values, closed.values), never.values
    assert np.allclose(never.values, bh.values), "wrapper must equal K=None"
    # Hand value at the last row: 0.5*1.331 + 0.5*1.20 = 1.2655
    assert abs(never.iloc[-1] - 1.2655) < 1e-12, never.iloc[-1]
    print(f"  Never == buy-and-hold (closed form): V_T={never.iloc[-1]:.6f}  OK")


def test_every_period_equals_constant_weight():
    merged = _merged(PRICES)
    every = rebalanced_value_series(merged, TICKERS, W, rebalance_every_periods=1)
    # Constant-weight rebalanced series: (1 + returns.dot(w)).cumprod(), V_0 = 1.
    rets = merged.set_index("date")[TICKERS].pct_change().dropna()
    expected = (1.0 + rets.dot(W)).cumprod()
    expected = pd.concat([pd.Series([1.0], index=[merged["date"].iloc[0]]), expected])
    assert np.allclose(every.values, expected.values), every.values
    print(f"  Every period == constant-weight compounded: V_T={every.iloc[-1]:.6f}  OK")


def test_periodic_rebalance_hand_checked():
    merged = _merged(PRICES)
    K2 = rebalanced_value_series(merged, TICKERS, W, rebalance_every_periods=2)
    # Resets at rows 0, 2, 4. Segment 0 (ref P_0): rows 1,2 -> 1.05, 1.105.
    # Segment 1 (ref P_2=[121,100], carry 1.105): row3 = 1.105*1.1 = 1.2155,
    #                                              row4 = 1.105*1.15 = 1.27075.
    expected = np.array([1.0, 1.05, 1.105, 1.2155, 1.27075])
    assert np.allclose(K2.values, expected), K2.values
    # Continuity: the reset row (2) is shared between segments — no flat/zero-return step.
    assert K2.iloc[2] != K2.iloc[1]
    # And it genuinely differs from both extremes at the reset boundary onward.
    every = rebalanced_value_series(merged, TICKERS, W, rebalance_every_periods=1)
    never = rebalanced_value_series(merged, TICKERS, W, rebalance_every_periods=None)
    assert not np.allclose(K2.values, every.values)
    assert not np.allclose(K2.values, never.values)
    print(f"  K=2 periodic rebalance hand-checked: V_T={K2.iloc[-1]:.6f}  OK")


def test_large_k_falls_back_to_never():
    merged = _merged(PRICES)
    big = rebalanced_value_series(merged, TICKERS, W, rebalance_every_periods=999)
    never = rebalanced_value_series(merged, TICKERS, W, rebalance_every_periods=None)
    assert np.allclose(big.values, never.values), big.values
    print("  K >= len falls back to never (single segment)  OK")


if __name__ == "__main__":
    print("rebalanced_value_series")
    test_never_equals_buy_and_hold()
    test_every_period_equals_constant_weight()
    test_periodic_rebalance_hand_checked()
    test_large_k_falls_back_to_never()
    print("All tests passed.")
