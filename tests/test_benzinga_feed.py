"""
Tests for the Benzinga earnings feed (Q-005).

RECORDED-ONLY observability client.  These tests pin:

* the field mapping (live-confirmed names, Q-005 STEP 0 2026-05-29),
  including the ×100 surprise-unit conversion and the verbatim
  eps_method capture;
* the eps_time derivation (BMO / AMC / DMH boundaries + null);
* most-recent-*reported* selection (skips future/projected records and
  records with actual_eps = None);
* empty-results → None (a normal miss, not an exception).

No network: ``requests.get`` is monkeypatched with a fake response.
"""

from __future__ import annotations

import json

import pytest

from data import benzinga_feed
from data.benzinga_feed import _derive_eps_time, fetch_latest_reported_earnings


# ── Fake HTTP plumbing ───────────────────────────────────────────────


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _patch_results(monkeypatch, results: list[dict]):
    """Patch benzinga_feed.requests.get to return *results*."""
    captured = {}

    def _fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return _FakeResponse({"results": results})

    monkeypatch.setattr(benzinga_feed.requests, "get", _fake_get)
    return captured


# A live-shaped reported record (values from Q-005 STEP 0, AAPL 2026-04-30).
def _reported(**overrides) -> dict:
    rec = {
        "currency": "USD",
        "date_status": "confirmed",
        "actual_eps": 2.01,
        "estimated_eps": 1.94,
        "previous_eps": 1.65,
        "eps_surprise": 0.07,
        "eps_surprise_percent": 0.0361,   # FRACTION → 3.61%
        "eps_method": "gaap",
        "importance": 5,
        "company_name": "Apple",
        "ticker": "AAPL",
        "date": "2026-04-30",
        "time": "16:30:00",
    }
    rec.update(overrides)
    return rec


# A projected/future record: no actual_eps, no surprise fields.
def _projected(**overrides) -> dict:
    rec = {
        "currency": "USD",
        "date_status": "projected",
        "estimated_eps": 2.2,
        "previous_eps": 2.01,
        "importance": 5,
        "company_name": "Apple",
        "ticker": "AAPL",
        "date": "2099-04-29",
        "time": "16:00:00",
    }
    rec.update(overrides)
    return rec


# ── eps_time derivation ──────────────────────────────────────────────


class TestDeriveEpsTime:
    def test_before_market_open_is_bmo(self):
        assert _derive_eps_time("06:00:00") == "BMO"
        assert _derive_eps_time("09:29:59") == "BMO"

    def test_after_market_close_is_amc(self):
        assert _derive_eps_time("16:00:00") == "AMC"
        assert _derive_eps_time("16:30:00") == "AMC"
        assert _derive_eps_time("20:00:00") == "AMC"

    def test_during_market_hours_is_dmh(self):
        # 09:30:00 is NOT < "09:30:00", and NOT >= "16:00:00" → DMH.
        assert _derive_eps_time("09:30:00") == "DMH"
        assert _derive_eps_time("12:00:00") == "DMH"
        assert _derive_eps_time("15:59:59") == "DMH"

    def test_null_or_empty_is_none(self):
        assert _derive_eps_time(None) is None
        assert _derive_eps_time("") is None
        assert _derive_eps_time("   ") is None


# ── Mapping ──────────────────────────────────────────────────────────


class TestMapping:
    def test_reported_record_maps_to_capture_fields(self, monkeypatch):
        captured = _patch_results(monkeypatch, [_reported()])
        out = fetch_latest_reported_earnings("AAPL", api_key="k", timeout=4.0)

        assert out == {
            "announce_date": "2026-04-30",
            "surprise_pct":  pytest.approx(3.61),   # 0.0361 × 100
            "actual_eps":    2.01,
            "estimate_eps":  1.94,                   # mapped from estimated_eps
            "eps_method":    "gaap",                 # raw literal, verbatim
            "date_status":   "confirmed",
            "importance":    5,
            "eps_time":      "AMC",                  # 16:30:00
        }

    def test_request_uses_expected_params(self, monkeypatch):
        captured = _patch_results(monkeypatch, [_reported()])
        fetch_latest_reported_earnings("AAPL", api_key="secret", timeout=2.5)
        p = captured["params"]
        assert p["ticker"] == "AAPL"
        assert p["sort"] == "date.desc"
        assert p["limit"] == 10
        assert p["apiKey"] == "secret"
        assert captured["timeout"] == 2.5

    def test_eps_method_captured_verbatim_not_normalised(self, monkeypatch):
        # ffo / adj must pass through unchanged (REIT / adjusted reporters).
        for method in ("gaap", "ffo", "adj", "FFO_weird"):
            _patch_results(monkeypatch, [_reported(eps_method=method)])
            out = fetch_latest_reported_earnings("X", api_key="k")
            assert out["eps_method"] == method

    def test_surprise_pct_null_guard(self, monkeypatch):
        # Some reported rows may lack eps_surprise_percent → store None,
        # never None * 100.
        _patch_results(monkeypatch, [_reported(eps_surprise_percent=None)])
        out = fetch_latest_reported_earnings("X", api_key="k")
        assert out["surprise_pct"] is None
        # Other fields still populated.
        assert out["actual_eps"] == 2.01

    def test_surprise_pct_is_percent_units(self, monkeypatch):
        # A 6.77% surprise (yfinance would store ~6.77, not 0.0677).
        _patch_results(monkeypatch, [_reported(eps_surprise_percent=0.0677)])
        out = fetch_latest_reported_earnings("X", api_key="k")
        assert out["surprise_pct"] == pytest.approx(6.77)


# ── Most-recent-reported selection ───────────────────────────────────


class TestSelection:
    def test_skips_projected_future_records(self, monkeypatch):
        # date.desc order: projected futures first, then reported.
        results = [
            _projected(date="2099-04-29"),
            _projected(date="2099-01-29"),
            _reported(date="2026-04-30", actual_eps=2.01),
            _reported(date="2026-01-29", actual_eps=2.84),
        ]
        _patch_results(monkeypatch, results)
        out = fetch_latest_reported_earnings("AAPL", api_key="k")
        # Most recent REPORTED record wins (2026-04-30), not the projected ones.
        assert out["announce_date"] == "2026-04-30"
        assert out["actual_eps"] == 2.01

    def test_skips_records_with_null_actual_eps(self, monkeypatch):
        # A record with actual_eps None (even if dated in the past) is not
        # yet "reported" → skipped.
        results = [
            _reported(date="2026-05-01", actual_eps=None),  # past but no actual
            _reported(date="2026-04-30", actual_eps=2.01),
        ]
        _patch_results(monkeypatch, results)
        out = fetch_latest_reported_earnings("AAPL", api_key="k")
        assert out["announce_date"] == "2026-04-30"

    def test_skips_future_dated_record_even_with_actual(self, monkeypatch):
        # Defensive: a future date is skipped regardless of actual_eps.
        results = [
            _reported(date="2099-12-31", actual_eps=9.99),  # future
            _reported(date="2026-04-30", actual_eps=2.01),
        ]
        _patch_results(monkeypatch, results)
        out = fetch_latest_reported_earnings("AAPL", api_key="k")
        assert out["announce_date"] == "2026-04-30"

    def test_picks_max_date_among_reported(self, monkeypatch):
        # Out-of-order input — selection takes the most recent by date,
        # not by list position.
        results = [
            _reported(date="2025-10-30", actual_eps=1.85),
            _reported(date="2026-04-30", actual_eps=2.01),
            _reported(date="2026-01-29", actual_eps=2.84),
        ]
        _patch_results(monkeypatch, results)
        out = fetch_latest_reported_earnings("AAPL", api_key="k")
        assert out["announce_date"] == "2026-04-30"


# ── Misses ───────────────────────────────────────────────────────────


class TestMiss:
    def test_empty_results_returns_none(self, monkeypatch):
        _patch_results(monkeypatch, [])
        assert fetch_latest_reported_earnings("EU.DE", api_key="k") is None

    def test_only_projected_records_returns_none(self, monkeypatch):
        # EU-style ticker: Benzinga (US-focused) returns only projected /
        # no reported records → None (a normal miss).
        _patch_results(monkeypatch, [_projected(), _projected(date="2099-01-29")])
        assert fetch_latest_reported_earnings("SAP.DE", api_key="k") is None

    def test_network_error_propagates(self, monkeypatch):
        # Network errors MUST NOT be swallowed here — the caller's
        # fail-safe wrapper handles them (keeps the forced-exception test
        # honest).
        import requests

        def _boom(url, params=None, timeout=None):
            raise requests.ConnectionError("down")

        monkeypatch.setattr(benzinga_feed.requests, "get", _boom)
        with pytest.raises(requests.ConnectionError):
            fetch_latest_reported_earnings("AAPL", api_key="k")
