"""
Cluster agreement gate (2026-09-07) — coordinator wiring and
signal_events persistence.

    (a) _fuse_signals carries the gate outcome in vote_ctx; a rejected
        solo vote is logged with the "Cluster-gate: rejected" journal line.
    (b) _cluster_gate_fields maps every fusion exit to the cluster_*
        payload (CONFLICTING recorded from the signal, non-cluster paths
        → None).
    (c) SignalLogger writes / migrates the four cluster_* columns.
    (d) run_combined end-to-end: a solo vote executes NOTHING and the
        Combined signal_events row says why (rejected_min_agreement,
        1 vote, BUY, Momentum); a 2-vote cluster still trades and its
        row says 'passed'.
    (e) Strategy rows keep NULL cluster_* columns (the gate is a
        Combined-row fact).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from analytics.signal_logger import SignalLogger
from orchestrator.cluster_detector import ClusterDetector
from storage.database import Database
from strategies.base import StrategyResult


def _vote(name: str, signal: str, confidence: float) -> StrategyResult:
    return StrategyResult(signal=signal, confidence=confidence, strategy_name=name)


SOLO = [
    _vote("Momentum", "BUY", 65.0),
    _vote("Pullback", "HOLD", 40.0),
    _vote("NewsCatalyst", "HOLD", 45.0),
]
PAIR = [
    _vote("Momentum", "BUY", 65.0),
    _vote("Pullback", "HOLD", 40.0),
    _vote("NewsCatalyst", "WEAK BUY", 55.0),
]
SPLIT = [
    _vote("Momentum", "BUY", 65.0),
    _vote("Pullback", "HOLD", 40.0),
    _vote("NewsCatalyst", "SELL", 55.0),
]


@pytest.fixture
def gate_db(tmp_path):
    return Database(str(tmp_path / "gate.db"))


def _bare_coord(db):
    from orchestrator.coordinator import Coordinator

    coord = Coordinator.__new__(Coordinator)
    coord._cluster_detector = ClusterDetector()
    coord.db = db
    return coord


def _fuse(coord, votes):
    return coord._fuse_signals(
        "NVDA", votes,
        sentiment_signal="BUY", sentiment_confidence=0.5,
        fallback_technical_signal="BUY", fallback_technical_confidence=0.6,
    )


def _combined_rows(db):
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM signal_events WHERE strategy='Combined' ORDER BY id",
        ).fetchall()
        return [dict(r) for r in rows]


def _strategy_rows(db):
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM signal_events WHERE strategy!='Combined' ORDER BY id",
        ).fetchall()
        return [dict(r) for r in rows]


# ── (a) _fuse_signals ────────────────────────────────────────────────────


def test_fuse_solo_vote_rejected_and_logged(caplog):
    coord = _bare_coord(MagicMock())
    with caplog.at_level(logging.INFO, logger="orchestrator.coordinator"):
        signal, conf, path, ctx = _fuse(coord, SOLO)

    assert signal == "HOLD"
    assert path == "CLUSTER"
    assert ctx["gate_status"] == "rejected_min_agreement"
    assert ctx["gate_votes"] == 1
    assert ctx["gate_direction"] == "BUY"
    assert ctx["gate_voters"] == "Momentum"
    # vote vector still intact — the data stream does not get thinner
    assert [v["strategy_name"] for v in ctx["strategy_votes"]] == [
        "Momentum", "Pullback", "NewsCatalyst",
    ]
    line = next(m for m in caplog.messages if m.startswith("Cluster-gate: rejected"))
    assert "ticker=NVDA" in line
    assert "direction=BUY" in line
    assert "votes=1" in line
    assert "sources=1" in line
    assert "min=2" in line
    assert "voters=Momentum" in line


def test_fuse_pair_passes_without_gate_line(caplog):
    coord = _bare_coord(MagicMock())
    with caplog.at_level(logging.INFO, logger="orchestrator.coordinator"):
        signal, conf, path, ctx = _fuse(coord, PAIR)

    assert (signal, conf, path) == ("BUY", 0.75, "CLUSTER")
    assert ctx["gate_status"] == "passed"
    assert ctx["gate_votes"] == 2
    assert ctx["gate_direction"] == "BUY"
    assert ctx["gate_voters"] == "Momentum,NewsCatalyst"
    assert not [m for m in caplog.messages if m.startswith("Cluster-gate:")]


def test_fuse_conflicting_keeps_vote_ctx_none():
    coord = _bare_coord(MagicMock())
    signal, conf, path, ctx = _fuse(coord, SPLIT)
    assert signal == "CONFLICTING"
    assert ctx is None  # A4 unchanged


def test_fuse_all_hold_reports_no_directional():
    coord = _bare_coord(MagicMock())
    signal, conf, path, ctx = _fuse(coord, [
        _vote("Momentum", "HOLD", 20.0),
        _vote("Pullback", "HOLD", 30.0),
        _vote("NewsCatalyst", "HOLD", 10.0),
    ])
    assert signal == "HOLD"
    assert ctx["gate_status"] == "no_directional"
    assert ctx["gate_votes"] == 0
    assert ctx["gate_direction"] is None
    assert ctx["gate_voters"] == ""


# ── (b) _cluster_gate_fields ─────────────────────────────────────────────


def test_gate_fields_for_every_exit():
    from orchestrator.coordinator import Coordinator

    f = Coordinator._cluster_gate_fields
    rejected_ctx = {
        "gate_status": "rejected_min_agreement", "gate_votes": 1,
        "gate_direction": "BUY", "gate_voters": "Momentum",
    }
    assert f(rejected_ctx, "HOLD", "CLUSTER") == {
        "status": "rejected_min_agreement", "votes": 1,
        "direction": "BUY", "voters": "Momentum",
    }
    assert f(rejected_ctx, "HOLD", "CLUSTER_PARTIAL")["status"] == (
        "rejected_min_agreement"
    )
    # CONFLICTING has no vote_ctx by design but is a detector exit
    assert f(None, "CONFLICTING", "CLUSTER") == {
        "status": "conflicting", "votes": None,
        "direction": None, "voters": None,
    }
    # never reached the detector → no gate fact
    assert f(None, "STRONG BUY", "FUSION_FALLBACK") is None
    assert f(rejected_ctx, "HOLD", None) is None
    # PEAD override ctx (no gate keys) on a fallback path
    pead_ctx = {"strategy_votes": [], "directional_count": None}
    assert f(pead_ctx, "BUY", "FUSION_FALLBACK") is None


# ── (c) SignalLogger columns ─────────────────────────────────────────────


def test_signal_logger_persists_cluster_columns(gate_db):
    logger = SignalLogger(gate_db)
    logger.log({
        "ticker": "NVDA", "session": "US_PRE", "strategy": "Combined",
        "signal": "HOLD", "confidence": 0.36, "signal_path": "CLUSTER",
        "cluster_gate": "rejected_min_agreement", "cluster_votes": 1,
        "cluster_direction": "BUY", "cluster_voters": "Momentum",
    })
    logger.log({
        "ticker": "NVDA", "session": "US_PRE", "strategy": "Momentum",
        "signal": "BUY", "confidence": 0.65,
    })
    combined, = _combined_rows(gate_db)
    assert combined["cluster_gate"] == "rejected_min_agreement"
    assert combined["cluster_votes"] == 1
    assert combined["cluster_direction"] == "BUY"
    assert combined["cluster_voters"] == "Momentum"
    strat, = _strategy_rows(gate_db)
    assert strat["cluster_gate"] is None
    assert strat["cluster_votes"] is None


def test_signal_logger_migrates_existing_table(gate_db):
    """Pre-gate DBs get the columns via idempotent ALTER; a second
    SignalLogger on the same DB must not raise on duplicate columns."""
    with gate_db._connect() as conn:
        conn.execute(
            "CREATE TABLE signal_events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT NOT NULL, session TEXT, ticker TEXT NOT NULL, "
            "strategy TEXT, signal TEXT NOT NULL, confidence REAL, rsi REAL, "
            "sma_ratio REAL, volume_ratio REAL, sentiment_score REAL, "
            "news_score REAL, social_score REAL, bull_case TEXT, bear_case TEXT, "
            "debate_outcome TEXT, price_at_signal REAL, "
            "trade_executed INTEGER NOT NULL DEFAULT 0, trade_id TEXT)"
        )
    SignalLogger(gate_db)
    SignalLogger(gate_db)
    with gate_db._connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(signal_events)")}
    assert {"cluster_gate", "cluster_votes", "cluster_direction",
            "cluster_voters"} <= cols


# ── (d) run_combined end-to-end ──────────────────────────────────────────


def _integration_coord(monkeypatch, db, votes):
    """run_combined harness: real db + real fusion + real SignalLogger,
    everything external mocked (pattern from test_signal_attribution)."""
    coord = _bare_coord(db)
    coord.signal_logger = SignalLogger(db)
    coord.macro_context = ""

    monkeypatch.setattr(
        "orchestrator.coordinator.strategy_label", lambda t: "Momentum",
    )
    monkeypatch.setattr(
        "orchestrator.coordinator.get_days_to_earnings", lambda t: None,
    )

    coord.regime_agent = MagicMock()
    coord.regime_agent.run.return_value = {}
    coord.regime_detector = MagicMock()

    coord.run = MagicMock(return_value={
        "ticker": "NVDA", "signal": "BUY", "avg_score": 0.5,
        "run_id": 1, "market": {"price": 100.0},
    })
    coord.technical_agent = MagicMock()
    coord.technical_agent.run.return_value = {
        "signal": "BUY", "signal_id": 1,
        "indicators": {"price": 100.0}, "bars": None,
        "adjusted_confidence": 0.6,
    }

    def _risk(*, signal, **kw):
        if signal in ("HOLD", "CONFLICTING"):
            return {"skipped": True, "skip_reason": "no actionable signal"}
        return {
            "skipped": False, "direction": "BUY", "shares": 10,
            "position_size_usd": 1_000.0, "stop_loss": 95.0,
            "take_profit": 108.0,
        }
    coord.risk_agent = MagicMock()
    coord.risk_agent.run.side_effect = _risk

    coord.paper_trader = MagicMock()
    coord.paper_trader.track_trade.return_value = {"trade_id": 42, "price": 100.0}
    coord._portfolio_manager = MagicMock()
    coord._portfolio_manager.can_add_position.return_value = (True, "")
    coord._has_alpaca_position = MagicMock(return_value=False)

    # strategy rows are logged by the (real) _log_strategy_result
    def _gather(ticker, bars, sentiment, *, session=None, regime=None):
        for v in votes:
            coord._log_strategy_result(ticker, v, session=session, regime=regime)
        return list(votes)
    coord._gather_strategy_votes = _gather
    return coord


def test_run_combined_solo_vote_no_trade_but_logged(monkeypatch, gate_db):
    coord = _integration_coord(monkeypatch, gate_db, SOLO)

    result = coord.run_combined(
        "NVDA", verbose=False, account_balance=10_000.0,
        execute=True, session="US_PRE",
    )

    assert result["combined_signal"] == "HOLD"
    assert result["execution"] is None
    coord.paper_trader.track_trade.assert_not_called()
    assert result["cluster_gate"] == {
        "status": "rejected_min_agreement", "votes": 1,
        "direction": "BUY", "voters": "Momentum",
    }

    combined, = _combined_rows(gate_db)
    assert combined["signal"] == "HOLD"
    assert combined["signal_path"] == "CLUSTER"
    assert combined["trade_executed"] == 0
    assert combined["cluster_gate"] == "rejected_min_agreement"
    assert combined["cluster_votes"] == 1
    assert combined["cluster_direction"] == "BUY"
    assert combined["cluster_voters"] == "Momentum"

    # (e) the three strategy rows are still there, gate columns NULL
    strat = _strategy_rows(gate_db)
    assert [r["strategy"] for r in strat] == ["Momentum", "Pullback", "NewsCatalyst"]
    assert all(r["cluster_gate"] is None for r in strat)


def test_run_combined_pair_trades_and_logs_passed(monkeypatch, gate_db):
    coord = _integration_coord(monkeypatch, gate_db, PAIR)

    result = coord.run_combined(
        "NVDA", verbose=False, account_balance=10_000.0,
        execute=True, session="US_PRE",
    )

    assert result["combined_signal"] == "BUY"
    assert result["execution"] == {"trade_id": 42, "price": 100.0}
    combined, = _combined_rows(gate_db)
    assert combined["trade_executed"] == 1
    assert combined["cluster_gate"] == "passed"
    assert combined["cluster_votes"] == 2
    assert combined["cluster_direction"] == "BUY"
    assert combined["cluster_voters"] == "Momentum,NewsCatalyst"


def test_run_combined_conflicting_logged_from_signal(monkeypatch, gate_db):
    coord = _integration_coord(monkeypatch, gate_db, SPLIT)
    result = coord.run_combined(
        "NVDA", verbose=False, account_balance=10_000.0,
        execute=True, session="US_PRE",
    )
    assert result["combined_signal"] == "CONFLICTING"
    assert result["execution"] is None
    combined, = _combined_rows(gate_db)
    assert combined["cluster_gate"] == "conflicting"
    assert combined["cluster_votes"] is None
