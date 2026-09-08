"""Unit tests for the daily OHLC ingest pipeline.

Mocks the Polygon and Alpaca HTTP layers — no network, no real key needed.

Covers:
  - schema: daily_ohlc table is created on Database init
  - upsert_daily_ohlc: idempotent (insert same (ticker,date) twice = 1 row,
    fields are updated to the new values)
  - hygiene: OHLC-inconsistent bars are STORED with quality_flag='OHLC_INCONSISTENT';
    >50% jumps get quality_flag='EXTREME_MOVE'; clean rows get NULL
  - both close (raw) and adj_close persist
  - universe builder returns exactly 20 uppercase US tickers, no ".", contains
    expected names (e.g. PBR), excludes EU ones (VNA.DE, COFA.PA)

Run:
    python3 -m pytest tests/test_ohlc_ingest.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.polygon_feed import PolygonFeed
from scripts.ingest_ohlc import (
    build_us20_universe,
    flag_bars,
)
from storage.database import Database


# ──────────────────────────────────────────────────────────────────────────────
# Schema + upsert
# ──────────────────────────────────────────────────────────────────────────────

def test_daily_ohlc_table_created(tmp_db):
    """Schema-init creates the daily_ohlc table with the expected columns."""
    with tmp_db._connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='daily_ohlc'"
        ).fetchall()
        assert len(rows) == 1, "daily_ohlc table should exist after init"

        cols = {r["name"] for r in conn.execute("PRAGMA table_info(daily_ohlc)").fetchall()}
        expected = {
            "ticker", "date", "open", "high", "low", "close", "adj_close",
            "volume", "source", "quality_flag", "ingested_at",
        }
        assert expected.issubset(cols), f"missing columns: {expected - cols}"


def _bar(ticker="AAPL", date="2026-05-20", o=100.0, h=101.0, l=99.0, c=100.5,
         adj=None, vol=1_000_000, src="polygon", flag=None) -> dict:
    return {
        "ticker": ticker, "date": date, "open": o, "high": h, "low": l, "close": c,
        "adj_close": adj if adj is not None else c, "volume": vol,
        "source": src, "quality_flag": flag,
    }


def test_upsert_is_idempotent(tmp_db):
    """Inserting the same (ticker, date) twice yields exactly one row,
    and the second write overwrites the fields of the first."""
    tmp_db.upsert_daily_ohlc([_bar(c=100.0, adj=99.5)])
    tmp_db.upsert_daily_ohlc([_bar(c=105.0, adj=104.5)])

    rows = tmp_db.get_daily_ohlc("AAPL", "2026-05-19", "2026-05-21")
    assert len(rows) == 1
    assert rows[0]["close"] == 105.0
    assert rows[0]["adj_close"] == 104.5


def test_upsert_persists_raw_and_adj(tmp_db):
    """`close` is the raw close; `adj_close` is the adjusted close;
    both are stored and round-trip cleanly."""
    tmp_db.upsert_daily_ohlc([_bar(c=100.0, adj=98.7)])
    rows = tmp_db.get_daily_ohlc("AAPL", "2026-05-20", "2026-05-20")
    assert rows[0]["close"] == 100.0
    assert rows[0]["adj_close"] == 98.7


# ──────────────────────────────────────────────────────────────────────────────
# Hygiene
# ──────────────────────────────────────────────────────────────────────────────

def _polygon_bar(date, o, h, l, c, vol=1_000_000):
    """Build a bar shaped like what PolygonFeed returns (no quality_flag)."""
    return {
        "date": date, "open": o, "high": h, "low": l, "close": c,
        "adj_close": c, "volume": vol, "ticker": "TST", "source": "polygon",
    }


def test_hygiene_clean_row_unflagged(tmp_db):
    bars = [_polygon_bar("2026-05-20", 100, 101, 99, 100.5)]
    flagged, _ = flag_bars(bars, extreme_pct=0.50)
    assert flagged[0]["quality_flag"] is None

    tmp_db.upsert_daily_ohlc(flagged)
    row = tmp_db.get_daily_ohlc("TST", "2026-05-20", "2026-05-20")[0]
    assert row["quality_flag"] is None


def test_hygiene_inconsistent_is_stored_with_flag(tmp_db):
    """An OHLC-inconsistent bar (low > open) must STILL be stored,
    just with quality_flag='OHLC_INCONSISTENT'."""
    bad = _polygon_bar("2026-05-20", o=100, h=101, l=105, c=100.5)  # low > open
    flagged, _ = flag_bars([bad], extreme_pct=0.50)
    assert flagged[0]["quality_flag"] == "OHLC_INCONSISTENT"

    n = tmp_db.upsert_daily_ohlc(flagged)
    assert n == 1, "bad rows must still be stored (flagged, not dropped)"

    row = tmp_db.get_daily_ohlc("TST", "2026-05-20", "2026-05-20")[0]
    assert row["quality_flag"] == "OHLC_INCONSISTENT"


def test_hygiene_extreme_move_flagged(tmp_db):
    """A >50% close-over-close jump gets quality_flag='EXTREME_MOVE'."""
    bars = [
        _polygon_bar("2026-05-19", 100, 101, 99, 100.0),
        _polygon_bar("2026-05-20", 100, 200, 99, 160.0),   # +60%
    ]
    flagged, _ = flag_bars(bars, extreme_pct=0.50)
    assert flagged[0]["quality_flag"] is None
    assert flagged[1]["quality_flag"] == "EXTREME_MOVE"

    tmp_db.upsert_daily_ohlc(flagged)
    rows = tmp_db.get_daily_ohlc("TST", "2026-05-19", "2026-05-20")
    flags = {r["date"]: r["quality_flag"] for r in rows}
    assert flags["2026-05-19"] is None
    assert flags["2026-05-20"] == "EXTREME_MOVE"


def test_hygiene_non_positive_values_flagged():
    bars = [_polygon_bar("2026-05-20", o=0.0, h=101, l=99, c=100.5)]
    flagged, _ = flag_bars(bars, extreme_pct=0.50)
    assert flagged[0]["quality_flag"] == "OHLC_INCONSISTENT"


# ──────────────────────────────────────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────────────────────────────────────

def test_us20_universe_shape():
    u = build_us20_universe()
    assert len(u) == 20, f"expected 20 tickers, got {len(u)}: {u}"
    assert all(t == t.upper() for t in u), "all tickers must be uppercase"
    assert len(set(u)) == 20, "tickers must be unique"


def test_us20_excludes_eu_names():
    u = build_us20_universe()
    assert all("." not in t for t in u), (
        f"EU/exchange-suffixed tickers leaked into US-20: "
        f"{[t for t in u if '.' in t]}"
    )
    # Spot-check: EU names that exist in PEAD_TICKERS must NOT be in US-20
    for eu in ("VNA.DE", "HOT.DE", "COFA.PA", "VIE.PA", "FNTN.DE", "LEG.DE"):
        assert eu not in u


def test_us20_contains_pbr_and_watchlist_core():
    """PBR is the US PEAD mid-cap; META is the canonical watchlist core."""
    u = build_us20_universe()
    assert "PBR" in u, "US PEAD ticker PBR must be in US-20"
    assert "META" in u, "watchlist core META must be in US-20"


# ──────────────────────────────────────────────────────────────────────────────
# Polygon feed: HTTP mocked
# ──────────────────────────────────────────────────────────────────────────────

def _mock_polygon_response(bars: list[tuple[str, float, float, float, float, int]]):
    """Build a fake requests.Response for /v2/aggs/...

    `bars` is a list of (date_iso, o, h, l, c, v).
    """
    import time as _t
    from unittest.mock import MagicMock

    results = []
    for d, o, h, l, c, v in bars:
        # Convert YYYY-MM-DD → ms epoch
        ts = int(_t.mktime(_t.strptime(d, "%Y-%m-%d"))) * 1000
        results.append({"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v})

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"results": results, "resultsCount": len(results)}
    return resp


def test_polygon_feed_parses_and_pairs_raw_adj(monkeypatch):
    """Both raw and adjusted close land on the returned dict."""
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        # First call = raw (adjusted=false), second = adjusted (adjusted=true)
        is_adj = params.get("adjusted") == "true"
        if is_adj:
            return _mock_polygon_response([("2026-05-20", 100, 101, 99, 99.5, 1_000_000)])
        return _mock_polygon_response([("2026-05-20", 100, 101, 99, 100.5, 1_000_000)])

    # Bypass rate-limit sleep
    monkeypatch.setattr(PolygonFeed, "_respect_rate_limit", lambda self: None)
    monkeypatch.setattr("data.polygon_feed.requests.get", fake_get)

    feed = PolygonFeed(api_key="test")
    bars = feed.get_daily_aggs("AAPL", "2026-05-20", "2026-05-20")
    assert len(bars) == 1
    assert bars[0]["close"] == 100.5   # raw
    assert bars[0]["adj_close"] == 99.5  # adjusted


def test_polygon_feed_survives_adjusted_failure(monkeypatch):
    """If the adjusted call fails, raw bars still return with adj_close=None."""
    def fake_get(url, params=None, timeout=None):
        if params.get("adjusted") == "true":
            raise RuntimeError("simulated adjusted-call failure")
        return _mock_polygon_response([("2026-05-20", 100, 101, 99, 100.5, 1_000_000)])

    monkeypatch.setattr(PolygonFeed, "_respect_rate_limit", lambda self: None)
    monkeypatch.setattr("data.polygon_feed.requests.get", fake_get)

    feed = PolygonFeed(api_key="test")
    bars = feed.get_daily_aggs("AAPL", "2026-05-20", "2026-05-20")
    assert len(bars) == 1
    assert bars[0]["close"] == 100.5
    assert bars[0]["adj_close"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Window end + freshness gate
# ──────────────────────────────────────────────────────────────────────────────

from datetime import date, datetime, timezone

import scripts.ingest_ohlc as ingest
from data.market_calendar import last_us_trading_day


def test_window_end_same_day_after_cutoff():
    """After 22:00 UTC the just-closed session is included in the window."""
    late = datetime(2026, 8, 31, 22, 30, tzinfo=timezone.utc)
    assert ingest._window_end(late) == date(2026, 8, 31)


def test_window_end_yesterday_before_cutoff():
    """Before the cutoff (market possibly still open) the window ends yesterday
    so a partial in-progress bar can never be ingested."""
    midday = datetime(2026, 8, 31, 14, 0, tzinfo=timezone.utc)
    assert ingest._window_end(midday) == date(2026, 8, 30)


def test_check_freshness_passes_when_current(tmp_db):
    tmp_db.upsert_daily_ohlc([_bar(ticker="AAPL", date="2026-08-31")])
    expected, stale = ingest.check_freshness(tmp_db, ["AAPL"], "2026-08-31")
    assert expected == "2026-08-31"
    assert stale == []


def test_check_freshness_rolls_expectation_over_weekend(tmp_db):
    """Window ending Sunday only expects Friday's bar — weekend is no lag."""
    tmp_db.upsert_daily_ohlc([_bar(ticker="AAPL", date="2026-08-28")])
    expected, stale = ingest.check_freshness(tmp_db, ["AAPL"], "2026-08-30")
    assert expected == "2026-08-28"
    assert stale == []


def test_check_freshness_flags_stale_and_missing(tmp_db):
    """The 31.08 incident shape: store stuck on Friday while Monday closed."""
    tmp_db.upsert_daily_ohlc([_bar(ticker="AAPL", date="2026-08-28")])
    expected, stale = ingest.check_freshness(tmp_db, ["AAPL", "MSFT"], "2026-08-31")
    assert expected == "2026-08-31"
    assert ("AAPL", "2026-08-28") in stale
    assert ("MSFT", None) in stale


def _run_with_fake_feed(tmp_db, monkeypatch, bar_date: str, alerts: list):
    """Drive ingest.run('incremental') with a feed that returns one bar per
    ticker dated `bar_date`. Returns the exit code."""
    class FakeFeed:
        def get_daily_aggs(self, ticker, start, end):
            return [{
                "date": bar_date, "open": 100.0, "high": 101.0,
                "low": 99.0, "close": 100.5, "adj_close": 100.5,
                "volume": 1_000_000,
            }]

    monkeypatch.setattr(ingest, "build_feed", lambda source: FakeFeed())
    monkeypatch.setattr(ingest, "Database", lambda: tmp_db)
    monkeypatch.setattr(ingest, "_alert_failure", alerts.append)
    return ingest.run("incremental")


def test_run_fails_on_stale_data(tmp_db, monkeypatch):
    """A run whose freshest bar is older than the expected session exits 1
    and alerts — a silent 'ok' on zero new rows is no longer possible."""
    stale_day = "2020-01-02"  # far older than any expected session
    rc = _run_with_fake_feed(tmp_db, monkeypatch, stale_day, alerts := [])
    assert rc == 1
    assert alerts and "freshness gate" in alerts[0]


def test_run_passes_on_fresh_data(tmp_db, monkeypatch):
    fresh_day = last_us_trading_day(ingest._window_end()).isoformat()
    rc = _run_with_fake_feed(tmp_db, monkeypatch, fresh_day, alerts := [])
    assert rc == 0
    assert alerts == []


# ──────────────────────────────────────────────────────────────────────────────
# Alpaca feed: HTTP mocked (default source since 2026-09-08)
# ──────────────────────────────────────────────────────────────────────────────

from data.alpaca_ohlc_feed import AlpacaOHLCFeed


def _mock_alpaca_response(symbol, bars, next_page_token=None):
    """Fake requests.Response for /v2/stocks/bars; `bars` = [(date, o, h, l, c, v)]."""
    from unittest.mock import MagicMock
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "bars": {symbol: [
            {"t": f"{d}T04:00:00Z", "o": o, "h": h, "l": l, "c": c, "v": v, "n": 1, "vw": c}
            for d, o, h, l, c, v in bars
        ]},
        "next_page_token": next_page_token,
    }
    return resp


def test_alpaca_feed_parses_and_pairs_raw_split(monkeypatch):
    """Raw bars carry the store fields; adj_close comes from adjustment=split."""
    seen = []

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.append(dict(params))
        if params["adjustment"] == "split":
            return _mock_alpaca_response("AAPL", [("2026-09-04", 100, 101, 99, 99.5, 1_000)])
        return _mock_alpaca_response("AAPL", [("2026-09-04", 100, 101, 99, 100.5, 1_000)])

    monkeypatch.setattr(AlpacaOHLCFeed, "_respect_rate_limit", lambda self: None)
    monkeypatch.setattr("data.alpaca_ohlc_feed.requests.get", fake_get)

    feed = AlpacaOHLCFeed(api_key="k", secret_key="s")
    bars = feed.get_daily_aggs("AAPL", "2026-09-01", "2026-09-04")
    assert bars == [{
        "date": "2026-09-04", "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.5, "volume": 1_000, "adj_close": 99.5,
    }]
    # consolidated tape, raw first, then split-only (never dividend-adjusted)
    assert [p["adjustment"] for p in seen] == ["raw", "split"]
    assert all(p["feed"] == "sip" and p["timeframe"] == "1Day" for p in seen)


def test_alpaca_feed_survives_split_failure(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if params["adjustment"] == "split":
            raise RuntimeError("simulated split-call failure")
        return _mock_alpaca_response("AAPL", [("2026-09-04", 100, 101, 99, 100.5, 1_000)])

    monkeypatch.setattr(AlpacaOHLCFeed, "_respect_rate_limit", lambda self: None)
    monkeypatch.setattr("data.alpaca_ohlc_feed.requests.get", fake_get)

    bars = AlpacaOHLCFeed(api_key="k", secret_key="s").get_daily_aggs("AAPL", "2026-09-01", "2026-09-04")
    assert len(bars) == 1 and bars[0]["close"] == 100.5 and bars[0]["adj_close"] is None


def test_alpaca_feed_follows_page_token(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params.get("page_token"))
        if params["adjustment"] == "split":
            return _mock_alpaca_response("AAPL", [])
        if params.get("page_token") is None:
            return _mock_alpaca_response("AAPL", [("2026-09-03", 1, 2, 1, 1.5, 10)], next_page_token="p2")
        return _mock_alpaca_response("AAPL", [("2026-09-04", 1, 2, 1, 1.6, 10)])

    monkeypatch.setattr(AlpacaOHLCFeed, "_respect_rate_limit", lambda self: None)
    monkeypatch.setattr("data.alpaca_ohlc_feed.requests.get", fake_get)

    bars = AlpacaOHLCFeed(api_key="k", secret_key="s").get_daily_aggs("AAPL", "2026-09-01", "2026-09-04")
    assert [b["date"] for b in bars] == ["2026-09-03", "2026-09-04"]
    assert calls[:2] == [None, "p2"]


def test_alpaca_feed_non_retryable_4xx_raises(monkeypatch):
    """The free-plan SIP recency 403 (or bad credentials) must fail the ticker
    immediately — no retry loop, no silent empty result."""
    from unittest.mock import MagicMock
    resp = MagicMock(); resp.status_code = 403
    resp.text = '{"message":"subscription does not permit querying recent SIP data"}'
    n = {"calls": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        n["calls"] += 1
        return resp

    monkeypatch.setattr(AlpacaOHLCFeed, "_respect_rate_limit", lambda self: None)
    monkeypatch.setattr("data.alpaca_ohlc_feed.requests.get", fake_get)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        AlpacaOHLCFeed(api_key="k", secret_key="s").get_daily_aggs("AAPL", "2026-09-01", "2026-09-04")
    assert n["calls"] == 1


def test_alpaca_end_clamped_to_sip_recency_margin():
    """A window ending today is asked for as now - margin (>= 15 min back);
    a window ending in the past keeps its end-of-day timestamp. At the
    nightly 22:30 UTC run this yields 22:10 UTC, after the US close."""
    now = datetime(2026, 9, 8, 22, 30, tzinfo=timezone.utc)
    assert AlpacaOHLCFeed._clamp_end("2026-09-08", now) == "2026-09-08T22:10:00Z"
    assert AlpacaOHLCFeed._clamp_end("2026-09-04", now) == "2026-09-04T23:59:59Z"


def test_alpaca_feed_requires_credentials():
    feed = AlpacaOHLCFeed(api_key="", secret_key="")
    assert not feed.available
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        feed.get_daily_aggs("AAPL", "2026-09-01", "2026-09-04")


# ──────────────────────────────────────────────────────────────────────────────
# Source selection
# ──────────────────────────────────────────────────────────────────────────────

def test_build_feed_alpaca_default(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    assert isinstance(ingest.build_feed("alpaca"), AlpacaOHLCFeed)


def test_build_feed_missing_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        ingest.build_feed("alpaca")
    monkeypatch.setattr(ingest, "POLYGON_API_KEY", "")
    with pytest.raises(RuntimeError, match="POLYGON_API_KEY"):
        ingest.build_feed("polygon")


def test_build_feed_polygon_rollback(monkeypatch):
    monkeypatch.setattr(ingest, "POLYGON_API_KEY", "test-key")
    assert isinstance(ingest.build_feed("polygon"), PolygonFeed)


def test_build_feed_unknown_source():
    with pytest.raises(RuntimeError, match="unknown OHLC_SOURCE"):
        ingest.build_feed("yahoo")


def test_run_aborts_and_alerts_without_credentials(tmp_db, monkeypatch):
    """Missing credentials: exit 1 + Telegram alert, nothing written."""
    alerts = []
    monkeypatch.setattr(ingest, "OHLC_SOURCE", "alpaca")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setattr(ingest, "Database", lambda: tmp_db)
    monkeypatch.setattr(ingest, "_alert_failure", alerts.append)
    assert ingest.run("incremental") == 1
    assert alerts and "ALPACA_API_KEY" in alerts[0]
    assert tmp_db.get_daily_ohlc_max_dates(["AAPL"]).get("AAPL") is None


def test_run_stamps_source_on_rows(tmp_db, monkeypatch):
    """Rows written by the run carry OHLC_SOURCE in daily_ohlc.source."""
    fresh_day = last_us_trading_day(ingest._window_end()).isoformat()
    monkeypatch.setattr(ingest, "OHLC_SOURCE", "alpaca")
    rc = _run_with_fake_feed(tmp_db, monkeypatch, fresh_day, alerts := [])
    assert rc == 0 and alerts == []
    row = tmp_db.get_daily_ohlc("AAPL", fresh_day, fresh_day)[0]
    assert row["source"] == "alpaca"
