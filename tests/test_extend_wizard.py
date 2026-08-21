"""
Tests for the guided extend wizard's pure validation helpers.

Deterministic, no network: the FX-pair direction (getting it backwards silently inverts the
conversion), the ETF-name -> index-name reduction that seeds the Yahoo search, and the
recovered-yield band that gates the Reconstruct button.

Run:  .venv/bin/python tests/test_extend_wizard.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "efficient_frontier_app"))
from data_handling import (  # noqa: E402
    suggest_fx_ticker,
    index_query_from_name,
    q_hat_verdict,
    default_q_regime,
    CURATED_INDEX_HINTS,
    Q_BANDS,
    SEARCH_QUOTE_TYPES,
    _search_queries,
)


def test_suggest_fx_ticker_direction():
    # synthesize_total_return divides (index / fx), and Yahoo's ABCDEF=X is DEF per 1 ABC.
    # A USD index with an EUR fund therefore needs EURUSD=X (USD per EUR), NOT USDEUR=X:
    # index_USD / (USD per EUR) = EUR. The reverse would multiply the error instead.
    assert suggest_fx_ticker("EUR", "USD") == "EURUSD=X"
    assert suggest_fx_ticker("USD", "EUR") == "USDEUR=X"
    assert suggest_fx_ticker("eur", "usd") == "EURUSD=X", "must normalise case"
    print("  suggest_fx_ticker: EUR fund + USD index -> EURUSD=X (divisor direction)  OK")


def test_suggest_fx_ticker_no_conversion_cases():
    assert suggest_fx_ticker("EUR", "EUR") is None, "same currency needs no rate"
    assert suggest_fx_ticker("EUR", None) is None, "unknown index currency -> no suggestion"
    assert suggest_fx_ticker(None, "USD") is None
    assert suggest_fx_ticker("", "") is None
    print("  suggest_fx_ticker: same-currency and unknown-currency both yield None  OK")


def test_index_query_from_name():
    assert index_query_from_name("Vanguard FTSE All-World UCITS ETF USD Accumulation") == "FTSE All-World"
    assert index_query_from_name("iShares Core S&P 500 UCITS ETF USD (Acc)") == "S&P 500"
    assert index_query_from_name("") == ""
    assert index_query_from_name(None) == ""
    # An already-clean index name must survive untouched.
    assert index_query_from_name("MSCI World") == "MSCI World"
    print("  index_query_from_name: strips issuer + wrapper words, keeps the index name  OK")


def test_q_hat_verdict_band():
    # Inside the band: a plausible dividend yield.
    ok, msg = q_hat_verdict(0.019)
    assert ok and "1.90%" in msg, msg
    # Boundaries are inclusive.
    assert q_hat_verdict(0.0)[0] is True
    assert q_hat_verdict(0.06)[0] is True
    # Just outside, both directions.
    assert q_hat_verdict(-1e-6)[0] is False
    assert q_hat_verdict(0.0601)[0] is False
    print("  q_hat_verdict: 0-6%% inclusive, rejects just outside either edge  OK")


def test_q_hat_verdict_rejects_real_mismatch():
    # The pairing CLAUDE.md calls nonsensical (^GSPC -> VWCE.MI) recovers about -1.2%/yr.
    ok, msg = q_hat_verdict(-0.012)
    assert not ok and "negative" in msg.lower(), msg
    # A wildly high value is the other failure mode (wrong market / wrong FX).
    ok_hi, msg_hi = q_hat_verdict(0.25)
    assert not ok_hi and "dividend" in msg_hi.lower(), msg_hi
    # Guards.
    assert q_hat_verdict(None)[0] is False
    assert q_hat_verdict(float("nan"))[0] is False
    print("  q_hat_verdict: rejects the real -1.2%/yr mismatch + NaN/None guards  OK")


def _probe(quote_type, pays_dividends=False, ok=True):
    """The subset of a probe_ticker() result that default_q_regime actually reads."""
    return {"ok": ok, "quote_type": quote_type, "pays_dividends": pays_dividends}


def test_default_q_regime_classification():
    # A price index omits the dividends the fund collects -> expect a dividend-sized gap.
    assert default_q_regime(_probe("INDEX")) == "price_index"
    # A distributing fund's adjusted prices already carry the income -> expect ~0.
    assert default_q_regime(_probe("ETF", pays_dividends=True)) == "same_income"
    # Futures/FX have no income at all -> expect ~0. This is the GC=F -> SGLD.MI case.
    assert default_q_regime(_probe("FUTURE")) == "same_income"
    assert default_q_regime(_probe("CURRENCY")) == "same_income"
    # A fund with no distributions (accumulating, or a no-income asset like gold) is still
    # total-return through Adj Close -> expect ~0. This is IAU / GLD.
    assert default_q_regime(_probe("ETF")) == "same_income"
    # An unusable probe must not claim to know; fall back to the historical default.
    assert default_q_regime(_probe("INDEX", ok=False)) == "price_index"
    assert default_q_regime(None) == "price_index"
    print("  default_q_regime: INDEX->price_index, ETF/FUTURE/dividend-payers->same_income  OK")


def test_gate_on_real_measured_pairings():
    """The measured q̂ values that motivated two bands, gated under their detected regime."""
    good = [("^GSPC->SXR8.DE",  0.0025, "price_index"),
            ("VT->VWCE.MI",     0.0010, "same_income"),
            ("ACWI->VWCE.MI",   0.0005, "same_income"),
            ("GC=F->SGLD.MI",  -0.0024, "same_income"),   # was wrongly BLOCKED before this change
            ("IAU->SGLD.MI",   -0.0012, "same_income"),   # ditto
            ("GLD->SGLD.MI",    0.0003, "same_income")]
    bad  = [("^GSPC->VWCE.MI", -0.0049, "price_index"),
            ("VT->SGLD.MI",     0.0092, "same_income"),
            ("GLD->VWCE.MI",   -0.0288, "same_income"),
            ("IAU->SXR8.DE",    0.0449, "same_income"),
            ("VT->SXR8.DE",     0.0150, "same_income")]
    for name, q, regime in good:
        ok, msg = q_hat_verdict(q, regime)
        assert ok, f"{name} (q={q}) should PASS under {regime}: {msg}"
    for name, q, regime in bad:
        ok, msg = q_hat_verdict(q, regime)
        assert not ok, f"{name} (q={q}) should BLOCK under {regime}: {msg}"
    print(f"  real pairings: {len(good)} good pass, {len(bad)} bad block, under detected regimes  OK")


def test_same_income_band_is_symmetric_and_tight():
    lo, hi = Q_BANDS["same_income"]
    assert (lo, hi) == (-0.005, 0.005)
    assert q_hat_verdict(lo, "same_income")[0] and q_hat_verdict(hi, "same_income")[0]
    assert not q_hat_verdict(lo - 1e-6, "same_income")[0]
    assert not q_hat_verdict(hi + 1e-6, "same_income")[0]
    # A negative miss should name the fee as the likely cause, not just say "wrong".
    assert "fee" in q_hat_verdict(-0.02, "same_income")[1].lower()
    print("  same_income band: ±0.5% inclusive, negative miss blames the fee  OK")


def test_default_regime_preserves_old_behaviour():
    """Calling without a regime must still be the original 0-6% gate."""
    for q in (0.0, 0.019, 0.06):
        assert q_hat_verdict(q)[0], q
    for q in (-1e-6, -0.012, 0.0601):
        assert not q_hat_verdict(q)[0], q
    assert q_hat_verdict(0.019) == q_hat_verdict(0.019, "price_index")
    print("  no-regime call still behaves as the original 0-6% gate  OK")


def test_search_widening():
    # FUTURE is what makes GC=F (gold, back to 2000) reachable; EQUITY would flood the list.
    assert "FUTURE" in SEARCH_QUOTE_TYPES and "CURRENCY" in SEARCH_QUOTE_TYPES
    assert "EQUITY" not in SEARCH_QUOTE_TYPES
    # Most specific first: "Gold index" finds gold *miners*, plain "Gold" finds bullion futures,
    # so the full name must be tried before the shortened fallbacks.
    qs = _search_queries("Physical Gold")
    assert qs[0] == "Physical Gold index" and qs[1] == "Physical Gold" and "Gold" in qs
    assert qs.index("Physical Gold") < qs.index("Gold"), "specific query must come first"
    assert _search_queries("") == []
    print("  search: FUTURE/CURRENCY accepted, EQUITY excluded, specific query ordered first  OK")


def test_curated_covers_gold():
    gold = [sym for fam, entries in CURATED_INDEX_HINTS.items() if "gold" in fam
            for sym, _ in entries]
    assert "GC=F" in gold, "Yahoo search alone never surfaces bullion for 'Invesco Physical Gold ETC'"
    print("  curated hints: gold family maps to GC=F  OK")


def test_curated_hints_shape():
    # Hints are (symbol, label) pairs keyed by a lowercase family substring, so the lookup
    # against a lowercased longName can never miss on case.
    for family, entries in CURATED_INDEX_HINTS.items():
        assert family == family.lower(), family
        for sym, label in entries:
            assert sym.strip() and label.strip(), family
    assert any(s == "VT" for e in CURATED_INDEX_HINTS.values() for s, _ in e), \
        "FTSE All-World needs its proxy: no real FTSE All-World index has Yahoo history"
    print(f"  curated hints: {len(CURATED_INDEX_HINTS)} families, all lowercase keys  OK")


if __name__ == "__main__":
    print("extend wizard helpers")
    test_suggest_fx_ticker_direction()
    test_suggest_fx_ticker_no_conversion_cases()
    test_index_query_from_name()
    test_q_hat_verdict_band()
    test_q_hat_verdict_rejects_real_mismatch()
    test_curated_hints_shape()
    test_default_q_regime_classification()
    test_gate_on_real_measured_pairings()
    test_same_income_band_is_symmetric_and_tight()
    test_default_regime_preserves_old_behaviour()
    test_search_widening()
    test_curated_covers_gold()
    print("All tests passed.")
