"""Unit tests for the Polygon OHLC ingest pipeline.

Mocks the Polygon HTTP layer — no network, no real key needed.

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
