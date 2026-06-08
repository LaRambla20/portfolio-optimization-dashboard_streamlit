"""
Tests for the §2 per-asset look-back machinery (Issues 1 & 3):
- data_handling.evaluate_simple_return / evaluate_CAGR (empty / zero-span guards)
- data_handling.load_asset_series (own-history loader, date filter, usecols)
- ui_components._lookback_year_windows / _full_history_label (window selection + label)

Deterministic, no network. Uses synthetic price Series and tempdir CSVs.

Run:  .venv\\Scripts\\python tests\\test_lookback_windows.py
"""

import os
import sys
import math
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from data_handling import (  # noqa: E402
    evaluate_simple_return,
    evaluate_CAGR,
    load_asset_series,
)
from ui_components import _lookback_year_windows, _full_history_label  # noqa: E402


def _prices(values, start="2010-01-01", freq="MS"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="adj close")


def test_evaluate_simple_return_basic_and_empty():
    p = _prices([100.0, 110.0, 121.0])
    sr = evaluate_simple_return(p, p.index[0], p.index[-1])
    assert abs(sr - 0.21) < 1e-12, sr
    # Empty slice (window entirely after the data) -> NaN, not an IndexError.
    empty = evaluate_simple_return(p, pd.Timestamp("2099-01-01"), pd.Timestamp("2099-02-01"))
    assert math.isnan(empty), empty
    print("  evaluate_simple_return: value + empty->NaN  OK")


def test_evaluate_cagr_basic_empty_zerospan():
    # 10%/yr compounded monthly for ~2 years -> CAGR ~10%.
    n = 25
    monthly = 1.10 ** (1 / 12)
    p = _prices([100.0 * monthly ** i for i in range(n)])
    cagr = evaluate_CAGR(p, p.index[0], p.index[-1])
    assert abs(cagr - 0.10) < 5e-3, cagr   # calendar-day annualisation tolerance
    # Empty slice -> NaN.
    assert math.isnan(evaluate_CAGR(p, pd.Timestamp("2099-01-01"), pd.Timestamp("2099-02-01")))
    # Zero-span (single row, start == end) -> NaN (n_years <= 0 guard), no ZeroDivision.
    one = _prices([100.0])
    assert math.isnan(evaluate_CAGR(one, one.index[0], one.index[0]))
    print("  evaluate_CAGR: value + empty->NaN + zero-span->NaN  OK")


def test_load_asset_series_own_history_and_usecols():
    suffix = "_data_monthly.csv"
    with tempfile.TemporaryDirectory() as d:
        # An _EXT-style file carrying the extra columns the loader must ignore.
        frame = pd.DataFrame({
            "date": ["2015-01-01", "2015-02-01", "2015-03-01", "2030-01-01"],
            "adj close": [100.0, 105.0, 110.0, 999.0],
            "synthetic": [True, True, False, False],
            "recon_yield": [0.02, 0.02, np.nan, np.nan],
            "currency": ["EUR", "EUR", "EUR", "EUR"],
        })
        frame.to_csv(os.path.join(d, "EXTASSET" + suffix), index=False)
        s = load_asset_series(d, "EXTASSET", suffix, pd.Timestamp("2020-12-31"))
        # Date-indexed Series of adj close, own history, filtered to <= filter_date.
        assert isinstance(s, pd.Series), type(s)
        assert s.name == "adj close", s.name
        assert list(s.values) == [100.0, 105.0, 110.0], s.values   # 2030 row excluded by filter
        assert s.index[0] == pd.Timestamp("2015-01-01")
        assert s.index.is_monotonic_increasing
    print("  load_asset_series: own history, date filter, usecols ignores extras  OK")


def test_lookback_year_windows_fit_and_full():
    end = pd.Timestamp("2026-06-01")

    # 3.2y history: the 5y window must be dropped; 1y/3y kept; Full row last.
    start_32 = end - pd.Timedelta(days=int(3.2 * 365.25))
    wins = _lookback_year_windows(end, start_32)
    labels = [lbl for lbl, _ in wins]
    assert "1y" in labels and "3y" in labels and "5y" not in labels, labels
    assert labels[-1].startswith("Full ("), labels
    assert wins[-1][1] == start_32, wins[-1]      # Full row starts at data_start

    # 10y history: 1/3/5y all kept (in order), Full last.
    start_10 = end - pd.Timedelta(days=int(10 * 365.25))
    labels10 = [lbl for lbl, _ in _lookback_year_windows(end, start_10)]
    assert labels10[:3] == ["1y", "3y", "5y"], labels10

    # Boundary: data_start exactly == the 5y window start -> 5y included.
    start_5 = end - pd.DateOffset(years=5)
    labels5 = [lbl for lbl, _ in _lookback_year_windows(end, start_5)]
    assert "5y" in labels5, labels5
    print("  _lookback_year_windows: drops over-long windows, keeps fitting + Full  OK")


def test_full_history_label_formatting():
    base = pd.Timestamp("2010-01-01")
    # ~8y 4m span.
    lbl = _full_history_label(base, base + pd.Timedelta(days=int(8 * 365.25 + 4 * 30.4)))
    assert lbl.startswith("Full (8y "), lbl
    # Zero / negative span -> Full (0y 0m).
    assert _full_history_label(base, base) == "Full (0y 0m)"
    assert _full_history_label(base, base - pd.Timedelta(days=10)) == "Full (0y 0m)"
    # 12-month rollover: a ~0.99y span rounds months to 12, which must roll to (1y 0m) —
    # the output must never contain "12m".
    span = pd.Timedelta(days=int(round(0.99 * 365.25)))
    lbl2 = _full_history_label(base, base + span)
    assert " 12m)" not in lbl2, lbl2
    print(f"  _full_history_label: formatting + rollover (e.g. {lbl!r}, {lbl2!r})  OK")


if __name__ == "__main__":
    print("§2 look-back machinery")
    test_evaluate_simple_return_basic_and_empty()
    test_evaluate_cagr_basic_empty_zerospan()
    test_load_asset_series_own_history_and_usecols()
    test_lookback_year_windows_fit_and_full()
    test_full_history_label_formatting()
    print("All tests passed.")
