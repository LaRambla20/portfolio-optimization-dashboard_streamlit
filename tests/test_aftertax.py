"""Unit tests for rebalanced_value_aftertax (capital-gains tax / net-liquidation overlay, feature T).

Plain assert script, no pytest. Run: .venv\\Scripts\\python tests\\test_aftertax.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from portfolio_calculations import rebalanced_value_aftertax, rebalanced_value_series  # noqa: E402


def _merged(prices_by_ticker):
    """prices_by_ticker: dict ticker -> list of prices. Returns a merged_df with a 'date' column."""
    n = len(next(iter(prices_by_ticker.values())))
    dates = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.DataFrame({"date": dates, **prices_by_ticker})


def test_zero_tax_is_noop():
    rng = np.random.default_rng(0)
    df = _merged({"A": np.cumprod(1 + rng.normal(0.01, 0.05, 60)),
                  "B": np.cumprod(1 + rng.normal(0.005, 0.04, 60))})
    tickers, w = ["A", "B"], [0.6, 0.4]
    for K in (None, 1, 6):
        no_tax = rebalanced_value_series(df, tickers, w, K)
        with_tax, netliq = rebalanced_value_aftertax(df, tickers, w, K, tax_rate=0.0)
        assert np.allclose(with_tax.values, no_tax.values), f"with_tax != no-tax at K={K}"
        assert np.allclose(netliq.values, with_tax.values), f"netliq != with_tax at K={K}"
    print("test_zero_tax_is_noop OK")


def test_monotonicity():
    rng = np.random.default_rng(1)
    df = _merged({"A": np.cumprod(1 + rng.normal(0.01, 0.06, 80)),
                  "B": np.cumprod(1 + rng.normal(0.008, 0.05, 80))})
    tickers, w, K = ["A", "B"], [0.5, 0.5], 6
    no_tax = rebalanced_value_series(df, tickers, w, K)
    with_tax, netliq = rebalanced_value_aftertax(df, tickers, w, K, tax_rate=0.26)
    assert (with_tax.values <= no_tax.values + 1e-12).all(), "with_tax exceeds no-tax"
    assert (netliq.values <= with_tax.values + 1e-12).all(), "netliq exceeds with_tax"
    print("test_monotonicity OK")


def test_single_asset_only_final_liquidation_tax():
    # One asset: never overweight relative to a 100% target, so no rebalance ever realizes a gain.
    # Only the net-liquidation line carries tax, on the whole gain: netliq = V - rate*(V-1).
    prices = np.cumprod(1 + np.full(40, 0.01))  # steady riser, V_end > 1
    df = _merged({"A": prices})
    rate = 0.26
    with_tax, netliq = rebalanced_value_aftertax(df, ["A"], [1.0], rebalance_every_periods=6, tax_rate=rate)
    no_tax = rebalanced_value_series(df, ["A"], [1.0], 6)
    assert np.allclose(with_tax.values, no_tax.values), "single asset paid rebalance tax"
    v_end = with_tax.iloc[-1]
    assert np.isclose(netliq.iloc[-1], v_end - rate * (v_end - 1.0)), "final net-liq tax wrong"
    print("test_single_asset_only_final_liquidation_tax OK")


def _gross_end_value(df, tickers, w, K, rate):
    """Reimplementation taxing GROSS gains (the export's B1 behaviour) — for the netting comparison."""
    prices = df.set_index("date")[list(tickers)]
    wv = np.asarray(w, float) / np.sum(w)
    pv = prices.to_numpy(float)
    n = len(prices)
    resets = set(range(K, n, K))
    v, k = wv.copy(), wv.copy()
    for t in range(1, n):
        v = v * (pv[t] / pv[t - 1])
        if t in resets:
            V = v.sum(); tgt = V * wv
            sell = np.maximum(0.0, v - tgt)
            bf = np.divide(k, v, out=np.ones_like(k), where=v > 0)
            realized = sell * (1.0 - bf)
            tax = rate * realized[realized > 0].sum()      # GROSS: losses ignored
            k = np.where(v > tgt, k * np.divide(tgt, v, out=np.ones_like(v), where=v > 0), k + (tgt - v))
            v = tgt.copy()
            f = tax / V if V > 0 else 0.0
            v *= 1.0 - f; k *= 1.0 - f
    return v.sum()


def test_within_period_netting_beats_gross():
    # Down market with a mixed-sign rebalance: A overweight at a gain, B overweight at a loss,
    # C underweight. Netting the loss against the gain taxes less → higher ending value than gross.
    df = _merged({"A": [1.0, 1.3], "B": [1.0, 0.8], "C": [1.0, 0.2]})
    tickers, w, K, rate = ["A", "B", "C"], [1 / 3, 1 / 3, 1 / 3], 1, 0.26
    with_tax, _ = rebalanced_value_aftertax(df, tickers, w, K, tax_rate=rate)
    net_end = with_tax.iloc[-1]
    gross_end = _gross_end_value(df, tickers, w, K, rate)
    assert net_end > gross_end + 1e-9, f"netting not applied: net={net_end} gross={gross_end}"
    print(f"test_within_period_netting_beats_gross OK (net {net_end:.6f} > gross {gross_end:.6f})")


if __name__ == "__main__":
    test_zero_tax_is_noop()
    test_monotonicity()
    test_single_asset_only_final_liquidation_tax()
    test_within_period_netting_beats_gross()
    print("\nAll aftertax tests passed.")
