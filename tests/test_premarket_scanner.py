"""
Unit tests for D9 pre-market scanner (scripts/premarket_scanner.py).

Covers:
    - Ranking math (earnings*3 + momentum*2 + sentiment*1, tiebreak on move)
    - Sector contagion via get_peers
    - Output JSON shape
    - resolve_us_tickers fallback when the scanner is disabled
    - resolve_us_tickers layering when scanner output is fresh
    - Stale output is ignored (falls back to core)
    - log_to_signal_events is fire-and-forget (no raise on DB error)

All network I/O (S&P 500 fetch, yfinance, news aggregator, sentiment agent)
is monkeypatched — no real calls.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import premarket_scanner as scanner


# ── Ranking ───────────────────────────────────────────────────────────────


def test_rank_candidates_scores_and_orders():
    earnings = {"AAPL", "MSFT"}
    momentum = {"AAPL", "NVDA", "TSLA"}
    sentiment = {"AAPL": 0.8, "META": 0.2}
    stats = {
        "AAPL": {"pct_change_5d": 5.0},
        "MSFT": {"pct_change_5d": 1.0},
        "NVDA": {"pct_change_5d": 7.0},
        "TSLA": {"pct_change_5d": -4.0},
        "META": {"pct_change_5d": 0.5},
    }
    ranked = scanner.rank_candidates(
        earnings_flag=earnings,
        momentum_flag=momentum,
        sentiment_scores=sentiment,
        stats=stats,
    )

    # AAPL: 3 (earnings) + 2 (momentum) + 1 (sentiment) = 6
    # MSFT: 3 (earnings)                                 = 3
    # NVDA:              2 (momentum)                    = 2
    # TSLA:              2 (momentum)                    = 2  (tie, higher |Δ| wins)
    # META:                               1 (sentiment)  = 1
    assert ranked[0]["ticker"] == "AAPL"
    assert ranked[0]["score"] == 6
    assert set(ranked[0]["reasons"]) == {"earnings", "momentum", "sentiment"}
    assert ranked[1]["ticker"] == "MSFT"
    assert ranked[1]["score"] == 3
    # NVDA (|7|) outranks TSLA (|4|) at score=2
    assert ranked[2]["ticker"] == "NVDA"
    assert ranked[3]["ticker"] == "TSLA"
    assert ranked[-1]["ticker"] == "META"


def test_rank_candidates_empty():
    assert scanner.rank_candidates(
        earnings_flag=set(), momentum_flag=set(), sentiment_scores={}, stats={}
    ) == []


# ── Momentum / liquidity filters ──────────────────────────────────────────


def test_find_momentum_candidates_uses_abs_threshold():
    stats = {
        "A": {"pct_change_5d": 5.0},
        "B": {"pct_change_5d": -4.0},
        "C": {"pct_change_5d": 2.0},       # below threshold
        "D": {"pct_change_5d": -2.9},      # just below
        "E": {"pct_change_5d": 3.0},       # at threshold
    }
    flagged = scanner.find_momentum_candidates(stats)
    assert flagged == {"A", "B", "E"}


def test_filter_liquid_uses_min_volume():
    stats = {
        "A": {"avg_volume": 600_000},
        "B": {"avg_volume": 499_999},       # just below
        "C": {"avg_volume": 500_000},       # at threshold
        "D": {"avg_volume": 0},
    }
    assert set(scanner.filter_liquid(stats)) == {"A", "C"}


# ── Earnings window ───────────────────────────────────────────────────────


def test_find_earnings_candidates_window(monkeypatch):
    today = date(2026, 4, 15)

    def fake_get_earnings_date(ticker):
        return {
            "IN_FORWARD":  today + timedelta(days=3),   # in +5d window
            "IN_BACKWARD": today - timedelta(days=2),   # in -3d window
            "EDGE_FWD":    today + timedelta(days=5),   # exact edge, included
            "EDGE_BACK":   today - timedelta(days=3),   # exact edge, included
            "TOO_LATE":    today + timedelta(days=6),   # past window
            "TOO_EARLY":   today - timedelta(days=4),   # past window
            "NO_DATE":     None,
        }.get(ticker)

    monkeypatch.setattr(
        "data.events_feed.get_earnings_date", fake_get_earnings_date
    )

    flagged = scanner.find_earnings_candidates(
        ["IN_FORWARD", "IN_BACKWARD", "EDGE_FWD", "EDGE_BACK",
         "TOO_LATE", "TOO_EARLY", "NO_DATE"],
        today=today,
    )
    assert flagged == {"IN_FORWARD", "IN_BACKWARD", "EDGE_FWD", "EDGE_BACK"}


# ── Sector contagion ──────────────────────────────────────────────────────


def test_expand_with_peers_pulls_in_correlated_tickers(monkeypatch):
    fake_db = MagicMock()
    peer_map = {
        "NVDA": ["META", "VRT"],
        "META": ["NVDA", "AMZN"],
        "XYZ":  [],
    }
    fake_db.get_peers.side_effect = lambda t: peer_map.get(t, [])

    monkeypatch.setattr(
        "storage.database.Database", lambda *a, **k: fake_db
    )

    expanded = scanner.expand_with_peers({"NVDA", "XYZ"})
    # NVDA's peers come in. XYZ has none. META's transitive peers do NOT
    # — expand_with_peers is a one-hop expansion, not a BFS.
    assert "NVDA" in expanded
    assert "META" in expanded
    assert "VRT" in expanded
    assert "XYZ" in expanded
    assert "AMZN" not in expanded  # would require two hops


# ── Output format ─────────────────────────────────────────────────────────


def test_run_scanner_output_shape(monkeypatch):
    # Stub every external call so run_scanner exercises only its orchestration.
    monkeypatch.setattr(scanner, "fetch_sp500_universe",
                        lambda: ["AAPL", "NVDA", "META", "XOM"])
    monkeypatch.setattr(scanner, "fetch_liquidity_and_momentum", lambda tickers: {
        "AAPL": {"pct_change_5d": 5.0, "avg_volume": 10_000_000, "last_close": 180.0},
        "NVDA": {"pct_change_5d": 8.0, "avg_volume": 50_000_000, "last_close": 900.0},
        "META": {"pct_change_5d": 1.0, "avg_volume": 15_000_000, "last_close": 500.0},
        "XOM":  {"pct_change_5d": 0.2, "avg_volume": 20_000_000, "last_close": 120.0},
    })
    monkeypatch.setattr(scanner, "find_earnings_candidates",
                        lambda tickers, today=None: {"AAPL"})
    monkeypatch.setattr(scanner, "fetch_sentiment_candidates",
                        lambda tickers: {"NVDA": 0.85})
    monkeypatch.setattr(scanner, "expand_with_peers",
                        lambda tickers: set(tickers) | {"VRT"})

    output = scanner.run_scanner(
        core_watchlist=["META", "JPM"],
        top_n=10,
        today=date(2026, 4, 15),
    )

    # Required top-level keys
    assert set(output) == {
        "date", "generated_at", "tickers", "ranked",
        "core_watchlist", "peers_added", "stats",
    }
    assert output["date"] == "2026-04-15"
    assert isinstance(output["generated_at"], str)
    assert isinstance(output["tickers"], list)
    assert output["tickers"] == sorted(output["tickers"])  # sorted
    assert len(output["tickers"]) == len(set(output["tickers"]))  # dedup'd

    # Core watchlist is always included as safety floor
    assert "META" in output["tickers"]
    assert "JPM" in output["tickers"]

    # Peer added via expand_with_peers
    assert "VRT" in output["tickers"]

    # Ranked entries have the expected shape
    for r in output["ranked"]:
        assert set(r) >= {"ticker", "score", "reasons"}

    # Stats counters are integers and reflect the mocked sizes
    s = output["stats"]
    assert s["universe_size"] == 4
    assert s["post_liquidity"] == 4
    assert s["earnings_flagged"] == 1
    assert s["momentum_flagged"] == 2   # AAPL (5%), NVDA (8%)
    assert s["sentiment_flagged"] == 1
    assert s["final_count"] == len(output["tickers"])


# ── save / load roundtrip ─────────────────────────────────────────────────


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    output = {
        "date": scanner._today_utc().isoformat(),
        "generated_at": "2026-04-15T13:00:00+00:00",
        "tickers": ["AAPL", "META", "NVDA"],
        "ranked": [{"ticker": "AAPL", "score": 6, "reasons": ["earnings"]}],
        "core_watchlist": ["META"],
        "peers_added": ["VRT"],
        "stats": {"final_count": 3},
    }
    target = tmp_path / "scanner_output.json"
    monkeypatch.setenv("SCANNER_OUTPUT_PATH", str(target))

    scanner.save_output(output)
    assert target.exists()

    loaded = scanner.load_scanner_output()
    assert loaded == output


def test_load_scanner_output_ignores_stale(tmp_path, monkeypatch):
    stale = {
        "date": "1999-01-01",
        "tickers": ["STALE"],
    }
    target = tmp_path / "scanner_output.json"
    target.write_text(json.dumps(stale))
    monkeypatch.setenv("SCANNER_OUTPUT_PATH", str(target))

    assert scanner.load_scanner_output() is None


def test_load_scanner_output_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "SCANNER_OUTPUT_PATH", str(tmp_path / "does_not_exist.json")
    )
    assert scanner.load_scanner_output() is None


# ── resolve_us_tickers fallback semantics ─────────────────────────────────


def test_resolve_us_tickers_disabled_returns_core(monkeypatch, tmp_path):
    """ENABLE_PREMARKET_SCANNER=false → ignore any output file, return core."""
    # Even if a fresh output exists, disabling must take precedence.
    fresh = {
        "date": scanner._today_utc().isoformat(),
        "tickers": ["SCANNED1", "SCANNED2"],
    }
    target = tmp_path / "scanner_output.json"
    target.write_text(json.dumps(fresh))
    monkeypatch.setenv("SCANNER_OUTPUT_PATH", str(target))
    monkeypatch.setenv("ENABLE_PREMARKET_SCANNER", "false")

    result = scanner.resolve_us_tickers(["META", "JPM"])
    assert result == ["META", "JPM"]


def test_resolve_us_tickers_enabled_layers_on_core(monkeypatch, tmp_path):
    fresh = {
        "date": scanner._today_utc().isoformat(),
        "tickers": ["AAPL", "NVDA", "META"],  # META also in core
    }
    target = tmp_path / "scanner_output.json"
    target.write_text(json.dumps(fresh))
    monkeypatch.setenv("SCANNER_OUTPUT_PATH", str(target))
    monkeypatch.setenv("ENABLE_PREMARKET_SCANNER", "true")

    result = scanner.resolve_us_tickers(["META", "JPM"])
    # Union: scanner + core, sorted, deduplicated
    assert result == ["AAPL", "JPM", "META", "NVDA"]


def test_resolve_us_tickers_enabled_but_stale_falls_back(monkeypatch, tmp_path):
    stale = {"date": "1999-01-01", "tickers": ["STALE"]}
    target = tmp_path / "scanner_output.json"
    target.write_text(json.dumps(stale))
    monkeypatch.setenv("SCANNER_OUTPUT_PATH", str(target))
    monkeypatch.setenv("ENABLE_PREMARKET_SCANNER", "true")

    assert scanner.resolve_us_tickers(["META", "JPM"]) == ["META", "JPM"]


def test_resolve_us_tickers_enabled_but_missing_file_falls_back(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "SCANNER_OUTPUT_PATH", str(tmp_path / "nope.json")
    )
    monkeypatch.setenv("ENABLE_PREMARKET_SCANNER", "true")

    assert scanner.resolve_us_tickers(["META", "JPM"]) == ["META", "JPM"]


# ── signal_events attribution is fire-and-forget ─────────────────────────


def test_log_to_signal_events_never_raises(monkeypatch):
    """SignalLogger.log() raises unexpectedly — caller must swallow it."""
    broken_logger = MagicMock()
    broken_logger.log.side_effect = RuntimeError("boom")
    monkeypatch.setattr(
        "analytics.signal_logger.SignalLogger",
        lambda *a, **k: broken_logger,
    )
    out = {
        "tickers": ["AAPL", "NVDA"],
        "ranked": [
            {"ticker": "AAPL", "score": 6, "reasons": ["earnings", "momentum"],
             "sentiment": None},
        ],
    }
    # Must not raise
    scanner.log_to_signal_events(out)
    # Called once per ticker
    assert broken_logger.log.call_count == 2


def test_log_to_signal_events_core_only_ticker_uses_fallback_reason(monkeypatch):
    """A ticker in .tickers but not in .ranked is a 'core_watchlist' row."""
    good_logger = MagicMock()
    monkeypatch.setattr(
        "analytics.signal_logger.SignalLogger",
        lambda *a, **k: good_logger,
    )
    out = {
        "tickers": ["CORE_ONLY"],
        "ranked": [],
    }
    scanner.log_to_signal_events(out)

    good_logger.log.assert_called_once()
    payload = good_logger.log.call_args[0][0]
    assert payload["ticker"] == "CORE_ONLY"
    assert payload["strategy"] == "PreMarketScanner"
    assert payload["signal"] == "WATCH"
    assert payload["debate_outcome"] == "core_watchlist"
