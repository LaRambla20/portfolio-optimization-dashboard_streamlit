"""
Tests for data_handling.synthesize_total_return.

Deterministic tests (no network) verify the calibration math, the splice
continuity, and the FX-conversion caveat. A final live test downloads the S&P
price index (^GSPC) and its total-return counterpart (^SP500TR) and checks the
recovered yield lands in the realistic dividend-yield band (~1.8-2.0%); it SKIPs
(does not fail) if the network/SSL is unavailable.

Run:  .venv\\Scripts\\python tests\\test_total_return_synthesis.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from data_handling import (  # noqa: E402
    synthesize_total_return,
    build_reconstructed_frame,
    read_synthetic_info,
    build_merged_dataframe,
    apply_eur_conversion,
    read_currency_info,
    convert_series_to_eur,
)


def _index_series(n, periods_per_year, seed=0, drift=0.0):
    """A noisy random-walk price *level* series on a regular calendar."""
    rng = np.random.default_rng(seed)
    if periods_per_year == 12:
        freq = "MS"
    elif periods_per_year == 52:
        freq = "W"
    else:
        freq = "B"
    dates = pd.date_range("2000-01-01", periods=n, freq=freq)
    step = rng.normal(drift / periods_per_year, 0.04, n)
    level = 100.0 * np.cumprod(1.0 + step)
    return pd.Series(level, index=dates)


def _gross_up(level, annual_yield, periods_per_year):
    """Turn a price-return level series into a total-return one at a known yield."""
    per_step = (1.0 + annual_yield) ** (1.0 / periods_per_year)
    factor = per_step ** np.arange(len(level))
    return level * factor


def test_recovers_known_yield():
    N = 12
    Q = 0.02
    idx = _index_series(240, N, seed=1)
    tr = _gross_up(idx, Q, N)
    out = synthesize_total_return(idx, tr, eurusd=None, periods_per_year=N)
    # Tolerance ~1e-4, not machine epsilon: CAGR annualises on calendar days
    # (days/365.25) while the gross-up compounds over unequal-length months.
    assert abs(out["q_hat"] - Q) < 1e-4, out["q_hat"]
    # Full overlap → nothing to reconstruct; series is just the ETF.
    assert out["synthetic_start"] is None
    print(f"  recovers known yield: q_hat={out['q_hat']:.6f} (true {Q})  OK")


def test_splice_is_continuous():
    N = 12
    Q = 0.025
    idx = _index_series(240, N, seed=2)
    full_tr = _gross_up(idx, Q, N)
    etf = full_tr.iloc[120:]  # ETF "launches" halfway through the index history
    out = synthesize_total_return(idx, etf, eurusd=None, periods_per_year=N)

    assert abs(out["q_hat"] - Q) < 1e-4, out["q_hat"]
    assert out["join_date"] == etf.index[0]
    assert out["synthetic_start"] == idx.index[0]
    # The reconstructed history must reproduce the (withheld) true TR everywhere,
    # confirming returns were chained and scaled to meet the ETF at the join.
    spliced = out["series"].reindex(full_tr.index)
    assert not spliced.isna().any()
    rel = (spliced / full_tr - 1.0).abs().max()
    assert rel < 1e-3, rel
    print(f"  splice continuous: max rel error vs true TR = {rel:.2e}  OK")


def test_fx_conversion_caveat():
    N = 12
    Q = 0.02
    g_fx = 0.015  # EUR strengthens 1.5%/yr vs USD
    idx_usd = _index_series(180, N, seed=3)
    per_fx = (1.0 + g_fx) ** (1.0 / N)
    fx = pd.Series(1.10 * per_fx ** np.arange(len(idx_usd)), index=idx_usd.index)  # USD per EUR
    etf_eur = _gross_up(idx_usd / fx, Q, N)

    with_fx = synthesize_total_return(idx_usd, etf_eur, eurusd=fx, periods_per_year=N)["q_hat"]
    no_fx = synthesize_total_return(idx_usd, etf_eur, eurusd=None, periods_per_year=N)["q_hat"]

    # Correct conversion recovers the true dividend yield...
    assert abs(with_fx - Q) < 1e-4, with_fx
    # ...skipping it folds FX drift into q_hat: (1+q) = (1+Q)/(1+g_fx).
    assert abs((1.0 + no_fx) - (1.0 + Q) / (1.0 + g_fx)) < 1e-4, no_fx
    print(f"  fx caveat: with_fx={with_fx:.6f} (true {Q}), "
          f"without_fx={no_fx:.6f} (FX drift folded in)  OK")


def test_reconstructed_frame_schema():
    N = 12
    Q = 0.02
    idx = _index_series(240, N, seed=5)
    full_tr = _gross_up(idx, Q, N)
    etf = full_tr.iloc[120:]  # ETF launches halfway
    frame, meta = build_reconstructed_frame(idx, etf, None, N)

    assert list(frame.columns) == ["date", "adj close", "synthetic", "recon_yield"]
    # Synthetic flag splits exactly at the ETF's first date.
    join = meta["join_date"]
    dates = pd.to_datetime(frame["date"])
    assert (frame.loc[dates < join, "synthetic"] == True).all()       # noqa: E712
    assert (frame.loc[dates >= join, "synthetic"] == False).all()     # noqa: E712
    # recon_yield carries q_hat on synthetic rows, blank on real ones.
    assert frame.loc[dates < join, "recon_yield"].notna().all()
    assert frame.loc[dates >= join, "recon_yield"].isna().all()
    assert abs(frame.loc[0, "recon_yield"] - meta["q_hat"]) < 1e-12
    # Real rows reproduce the ETF exactly; dates are ISO strings.
    real = frame[dates >= join].reset_index(drop=True)
    assert np.allclose(real["adj close"].values, etf.values)
    assert frame["date"].iloc[0] == idx.index[0].strftime("%Y-%m-%d")
    print(f"  reconstructed frame schema: {len(frame)} rows, "
          f"{int(frame['synthetic'].sum())} synthetic, q={meta['q_hat']*100:.2f}%  OK")


def test_save_read_roundtrip(tmp_suffix="_data_monthly.csv"):
    import os
    import tempfile

    N = 12
    Q = 0.022
    idx = _index_series(180, N, seed=6)
    full_tr = _gross_up(idx, Q, N)
    etf = full_tr.iloc[90:]
    frame, meta = build_reconstructed_frame(idx, etf, None, N)

    with tempfile.TemporaryDirectory() as d:
        etf_name = "WORLD"
        frame.to_csv(os.path.join(d, f"{etf_name}_EXT{tmp_suffix}"), index=False)
        filter_date = "2099-12-31"

        # §1 reader recovers the metadata.
        info = read_synthetic_info([f"{etf_name}_EXT"], d, tmp_suffix, filter_date)
        assert f"{etf_name}_EXT" in info
        m = info[f"{etf_name}_EXT"]
        assert m["n_synthetic"] == int(frame["synthetic"].sum())
        assert abs(m["q_hat"] - meta["q_hat"]) < 1e-9
        assert m["join_date"] < meta["join_date"]  # last synthetic row precedes join

        # Existing 2-column reader ignores the extra columns and reads adj close.
        merged = build_merged_dataframe([f"{etf_name}_EXT"], d, tmp_suffix, filter_date)
        assert list(merged.columns) == ["date", f"{etf_name}_EXT"]
        assert len(merged) == len(frame)
        assert np.allclose(merged[f"{etf_name}_EXT"].values, frame["adj close"].values)

        # A plain ticker (no synthetic column) is correctly omitted from §1 info.
        plain = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "adj close": [1.0, 1.1]})
        plain.to_csv(os.path.join(d, f"PLAIN{tmp_suffix}"), index=False)
        info2 = read_synthetic_info(["PLAIN"], d, tmp_suffix, filter_date)
        assert info2 == {}

    print("  save/read roundtrip: §1 reader + 2-col reader both OK")


def test_eur_conversion_scales_prices():
    # Three business days of an asset priced in USD; FX has a holiday gap on day 2.
    data = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "open": [10.0, 11.0, 12.0],
        "adj close": [10.0, 11.0, 12.0],
        "volume": [100, 200, 300],          # must NOT be scaled
    })
    fx = pd.Series(
        [0.90, 0.92],
        index=pd.to_datetime(["2020-01-01", "2020-01-03"]),  # 2020-01-02 missing
    )
    out = apply_eur_conversion(data, fx)
    # Day 1 -> 0.90; day 2 has no FX, takes nearest-prior 0.90; day 3 -> 0.92.
    assert np.allclose(out["adj close"].values, [10.0 * 0.90, 11.0 * 0.90, 12.0 * 0.92])
    assert np.allclose(out["open"].values, [10.0 * 0.90, 11.0 * 0.90, 12.0 * 0.92])
    assert list(out["volume"]) == [100, 200, 300]  # untouched
    print("  eur conversion: price cols scaled, volume untouched, FX gap nearest-prior  OK")


def test_currency_info_reads_stored_column():
    import os
    import tempfile
    suffix = "_data_daily.csv"
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame({"date": ["2020-01-01"], "adj close": [1.0], "currency": ["EUR"]}) \
            .to_csv(os.path.join(d, "AAA" + suffix), index=False)
        pd.DataFrame({"date": ["2020-01-01"], "adj close": [1.0], "currency": ["USD"]}) \
            .to_csv(os.path.join(d, "BBB" + suffix), index=False)
        info = read_currency_info(["AAA", "BBB"], d, suffix)
        assert info == {"AAA": "EUR", "BBB": "USD"}, info
    print("  currency info: reads stored column offline (no sniff)  OK")


def test_convert_series_to_eur():
    """The reconstruction path's ETF leg converts exactly like the download path's frame."""
    prices = pd.Series(
        [10.0, 11.0, 12.0],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    fx = pd.Series([0.90, 0.92], index=pd.to_datetime(["2020-01-01", "2020-01-03"]))
    out = convert_series_to_eur(prices, fx)
    # Same nearest-prior alignment as apply_eur_conversion (day 2 has no FX -> 0.90).
    assert np.allclose(out.values, [10.0 * 0.90, 11.0 * 0.90, 12.0 * 0.92])
    assert out.index.equals(prices.index), "index must survive the round-trip"
    print("  convert_series_to_eur: scales series, keeps index, nearest-prior FX  OK")


def test_reconstruction_currency_column():
    """`currency` is appended last and every price reader ignores it."""
    N = 12
    idx = _index_series(240, N, seed=11)
    etf = _gross_up(idx, 0.02, N).iloc[120:]

    # Default (None) stays on the pre-existing 4-column schema, byte-for-byte.
    frame_plain, _ = build_reconstructed_frame(idx, etf, None, N)
    assert list(frame_plain.columns) == ["date", "adj close", "synthetic", "recon_yield"]

    frame_cur, _ = build_reconstructed_frame(idx, etf, None, N, currency="EUR")
    assert list(frame_cur.columns) == ["date", "adj close", "synthetic", "recon_yield", "currency"]
    assert (frame_cur["currency"] == "EUR").all()
    # Adding the column changes nothing else.
    for col in frame_plain.columns:
        assert frame_plain[col].equals(frame_cur[col]), col

    # End to end: §1's currency check reads it offline, and the price reader ignores it.
    import os
    import tempfile
    suffix = "_data_monthly.csv"
    with tempfile.TemporaryDirectory() as d:
        frame_cur.to_csv(os.path.join(d, "XXX_EXT" + suffix), index=False)
        assert read_currency_info(["XXX_EXT"], d, suffix) == {"XXX_EXT": "EUR"}
        merged = build_merged_dataframe(["XXX_EXT"], d, suffix, pd.Timestamp("2100-01-01"))
        assert list(merged.columns) == ["date", "XXX_EXT"]
        assert len(merged) == len(frame_cur)
    print("  reconstruction currency column: appended last, read offline, price reader OK")


def test_real_sp500_yield():
    try:
        import yfinance as yf
        from curl_cffi import requests as cr
        import warnings
        warnings.filterwarnings("ignore")
        s = cr.Session(impersonate="chrome", verify=False)

        def fetch(t):
            h = yf.Ticker(t, session=s).history(
                start="2010-01-01", end="2024-12-31", interval="1mo", auto_adjust=False
            )
            return h["Close"].tz_localize(None)

        gspc = fetch("^GSPC")      # price-return
        sptr = fetch("^SP500TR")   # total-return
        if len(gspc) < 24 or len(sptr) < 24:
            raise RuntimeError("insufficient rows")
    except Exception as e:  # network / SSL / yfinance schema drift
        print(f"  real S&P validation: SKIP ({str(e)[:60]})")
        return

    out = synthesize_total_return(gspc, sptr, eurusd=None, periods_per_year=12)
    q = out["q_hat"]
    assert 0.012 < q < 0.025, f"q_hat={q:.4f} outside realistic S&P dividend band"
    print(f"  real S&P validation: q_hat={q*100:.2f}%/yr "
          f"(overlap {out['overlap_start'].date()}..{out['overlap_end'].date()})  OK")


if __name__ == "__main__":
    print("synthesize_total_return")
    test_recovers_known_yield()
    test_splice_is_continuous()
    test_fx_conversion_caveat()
    test_reconstructed_frame_schema()
    test_save_read_roundtrip()
    test_eur_conversion_scales_prices()
    test_currency_info_reads_stored_column()
    test_convert_series_to_eur()
    test_reconstruction_currency_column()
    test_real_sp500_yield()
    print("All tests passed.")
