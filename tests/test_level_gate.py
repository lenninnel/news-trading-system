"""
Tests for the F2 Chokepoint Gate (orchestrator/level_gate.py) and its
three migrated call sites in orchestrator/coordinator.py.

Covers (R-spec v1.1):
    T1. 8 core helper cases: per leg (sl/tp) ×
        {wrong_side, None, valid-adopt, no_fill}.
    T2. Edge documentation of the LITERAL spec: 0.0 SL is adopted
        (0.0 < fill — the downstream coordinator execution guard, not
        the gate, blocks execution), 0.0 TP is wrong_side, a negative
        SL is adopted, and fill_valid=False forces reason "no_fill"
        even for otherwise-valid candidates.
    T3. Logging contract: exactly one WARNING per non-adopted leg,
        literal prefix "F2-gate:", fields ticker/session/leg/reason/
        candidate/fill/kept-fresh present; adoption never WARNs.
    T4. Integration regression, strategy-override site in
        analyse_ticker_async: wrong-side SL falls back to the fresh
        calc, valid TP is adopted; track_trade AND the stored forward
        signal both receive the gated values.
    T5. Same regression shape for the strategy-override site in
        run_combined.
    T6. Static chokepoint check: coordinator.py contains ZERO direct
        assignments to risk["stop_loss"] / risk["take_profit"].
    T7. Level-integrity invariant v2 (model_mismatch, R spec
        2026-08-26): tolerance boundary both sides, PFE-270 replay,
        ATR-kept zero deviation, missing/<=0 reference skip path,
        degenerate denominator, wrong_side precedence, and sl-only
        scope (TP leg never runs the check).

v2 note: the T1 sl valid-adopt candidate and the T2 zero/negative SL
cases were originally written with candidates far from the fresh
reference; under the invariant those now reject as model_mismatch, so
the adopt fixtures use in-tolerance candidates and the literal-spec
0.0/negative adoptions are exercised via the no-usable-reference skip
path (which is where that layering still applies).

All external collaborators are MagicMocks — no network, no real DB
(mocking conventions follow tests/test_post_session_reviewer.py).
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from orchestrator.level_gate import apply_level_override


FILL = 100.0
FRESH_SL = 95.0
FRESH_TP = 108.0


def _fresh_risk() -> dict:
    return {
        "skipped": False,
        "direction": "BUY",
        "shares": 10,
        "position_size_usd": 1_000.0,
        "stop_loss": FRESH_SL,
        "take_profit": FRESH_TP,
    }


def _ctx(fill_valid: bool = True, origin: str = "strategy") -> dict:
    return {
        "ticker": "NVDA",
        "session": "US_OPEN",
        "origin": origin,
        "fill_valid": fill_valid,
    }


# ── T1: 8 core cases ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "leg,candidate,fill_valid,expect_adopt,expect_reason",
    [
        # sl: wrong_side / None / valid-adopt / no_fill
        # (adopt candidate 94.8: deviation |95-94.8|/(100-95) = 0.04 ≤ 0.05)
        ("sl", 105.0, True,  False, "wrong_side"),
        ("sl", None,  True,  False, "null_level"),
        ("sl", 94.8,  True,  True,  None),
        ("sl", 92.0,  False, False, "no_fill"),
        # tp: wrong_side / None / valid-adopt / no_fill
        ("tp", 95.0,  True,  False, "wrong_side"),
        ("tp", None,  True,  False, "null_level"),
        ("tp", 112.0, True,  True,  None),
        ("tp", 112.0, False, False, "no_fill"),
    ],
)
def test_core_cases(caplog, leg, candidate, fill_valid, expect_adopt, expect_reason):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    key = "stop_loss" if leg == "sl" else "take_profit"
    fresh = risk[key]

    adopted = apply_level_override(risk, leg, candidate, FILL, _ctx(fill_valid))

    assert adopted is expect_adopt
    if expect_adopt:
        assert risk[key] == candidate
    else:
        # Non-adoption keeps the fresh calc for that leg
        assert risk[key] == fresh
        assert f"reason={expect_reason}" in caplog.text

    # The untouched leg always keeps its fresh value
    other_key = "take_profit" if leg == "sl" else "stop_loss"
    assert risk[other_key] == _fresh_risk()[other_key]


# ── T2: literal-spec edge documentation (amended by invariant v2) ────────
#
# The gate implements S2 LITERALLY: `is not None` presence checks, no >0
# floor, no epsilon on the side comparisons. 0.0 and negative candidates
# are PRESENT values and pass the side comparison for the SL leg. Since
# the v2 invariant they then reject as model_mismatch whenever a usable
# fresh reference exists (deviation far beyond tolerance); the literal
# adoption survives only on the no-usable-reference skip path, where the
# independent downstream execution guard (coordinator.py, "missing/zero
# SL-TP" branch) remains the layer that blocks such trades — R §7's
# layering, now scoped to that path.


def test_zero_sl_rejected_as_model_mismatch_with_reference(caplog):
    """v2: 0.0 SL passes `0.0 < fill` but deviates 19.0 from the fresh
    reference → model_mismatch, fresh calc kept."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "sl", 0.0, FILL, _ctx())
    assert adopted is False
    assert risk["stop_loss"] == FRESH_SL
    assert "reason=model_mismatch" in caplog.text


def test_zero_sl_adopted_when_no_usable_reference(caplog):
    """Literal-spec adoption of 0.0 survives on the skip path (fresh SL
    None): `0.0 < fill` adopts, downstream guard blocks execution."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    risk["stop_loss"] = None
    adopted = apply_level_override(risk, "sl", 0.0, FILL, _ctx())
    assert adopted is True
    assert risk["stop_loss"] == 0.0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_zero_tp_is_wrong_side(caplog):
    """0.0 TP fails `0.0 > fill` → wrong_side, fresh TP kept."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "tp", 0.0, FILL, _ctx())
    assert adopted is False
    assert risk["take_profit"] == FRESH_TP
    assert "reason=wrong_side" in caplog.text


def test_negative_sl_rejected_as_model_mismatch_with_reference():
    """v2: a negative SL satisfies `candidate < fill` but deviates far
    beyond tolerance from the fresh reference → model_mismatch."""
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "sl", -5.0, FILL, _ctx())
    assert adopted is False
    assert risk["stop_loss"] == FRESH_SL


def test_negative_sl_adopted_when_no_usable_reference():
    """Literal-spec adoption of a negative SL survives on the skip path."""
    risk = _fresh_risk()
    risk["stop_loss"] = None
    adopted = apply_level_override(risk, "sl", -5.0, FILL, _ctx())
    assert adopted is True
    assert risk["stop_loss"] == -5.0


@pytest.mark.parametrize("leg,candidate", [("sl", 92.0), ("tp", 112.0)])
def test_invalid_fill_forces_no_fill_reason(caplog, leg, candidate):
    """fill_valid=False → no_fill, even for otherwise-valid candidates."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(
        risk, leg, candidate, FILL, _ctx(fill_valid=False),
    )
    assert adopted is False
    assert "reason=no_fill" in caplog.text
    assert "reason=wrong_side" not in caplog.text


# ── T3: logging contract (incl. amendments A2 + A3) ──────────────────────


def _gate_records(caplog, levelno):
    return [
        r for r in caplog.records
        if r.levelno == levelno and r.name == "orchestrator.level_gate"
    ]


def _warnings(caplog):
    return _gate_records(caplog, logging.WARNING)


@pytest.mark.parametrize(
    "leg,candidate,fill_valid,reason,origin,level",
    [
        # wrong_side / no_fill → WARNING regardless of origin
        ("sl", 105.0, True,  "wrong_side", "strategy", logging.WARNING),
        ("tp", 95.0,  True,  "wrong_side", "forward",  logging.WARNING),
        ("sl", 92.0,  False, "no_fill",    "forward",  logging.WARNING),
        ("tp", 112.0, False, "no_fill",    "strategy", logging.WARNING),
        # A2 split: null_level is INFO for strategy, WARNING for forward
        ("sl", None,  True,  "null_level", "strategy", logging.INFO),
        ("tp", None,  True,  "null_level", "strategy", logging.INFO),
        ("sl", None,  True,  "null_level", "forward",  logging.WARNING),
        ("tp", None,  True,  "null_level", "forward",  logging.WARNING),
    ],
)
def test_non_adoption_emits_exactly_one_line_with_all_fields(
    caplog, leg, candidate, fill_valid, reason, origin, level,
):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    kept = risk["stop_loss" if leg == "sl" else "take_profit"]

    apply_level_override(
        risk, leg, candidate, FILL, _ctx(fill_valid, origin=origin),
    )

    records = _gate_records(caplog, level)
    assert len(records) == 1, "exactly one line per non-adopted leg"
    # Only the level differs — never both levels for one non-adoption
    other = logging.INFO if level == logging.WARNING else logging.WARNING
    assert _gate_records(caplog, other) == []

    msg = records[0].getMessage()
    assert msg.startswith("F2-gate:"), "literal grep-continuity prefix"
    assert "ticker=NVDA" in msg
    assert "session=US_OPEN" in msg
    assert f"origin={origin}" in msg, "A3: origin mandatory in every line"
    assert f"leg={leg}" in msg
    assert f"reason={reason}" in msg
    assert f"candidate={candidate}" in msg
    assert f"fill={FILL}" in msg
    assert f"kept fresh={kept}" in msg


@pytest.mark.parametrize("leg", ["sl", "tp"])
def test_null_level_strategy_is_info_not_warning(caplog, leg):
    """A2: absent strategy levels are expected — INFO, never WARNING."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(
        risk, leg, None, FILL, _ctx(origin="strategy"),
    )
    assert adopted is False
    assert _warnings(caplog) == []
    infos = _gate_records(caplog, logging.INFO)
    assert len(infos) == 1
    assert infos[0].getMessage().startswith("F2-gate:")
    assert "reason=null_level" in infos[0].getMessage()
    assert "origin=strategy" in infos[0].getMessage()


@pytest.mark.parametrize("leg", ["sl", "tp"])
def test_null_level_forward_stays_warning(caplog, leg):
    """A2: a NULL level in a forward row is still anomalous → WARNING."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(
        risk, leg, None, FILL, _ctx(origin="forward"),
    )
    assert adopted is False
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "reason=null_level" in warnings[0].getMessage()
    assert "origin=forward" in warnings[0].getMessage()
    assert _gate_records(caplog, logging.INFO) == []


@pytest.mark.parametrize("leg,candidate", [("sl", 94.8), ("tp", 112.0)])
def test_adoption_emits_no_warning(caplog, leg, candidate):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, leg, candidate, FILL, _ctx())
    assert adopted is True
    assert _warnings(caplog) == []
    # Adoption logs at most DEBUG
    for r in caplog.records:
        if r.name == "orchestrator.level_gate":
            assert r.levelno <= logging.DEBUG


# ── T7: level-integrity invariant v2 (model_mismatch) ────────────────────
#
# deviation = |reference − candidate| / (fill − reference), sl leg only,
# _MODEL_TOLERANCE = 0.05. Fixtures: fresh SL 95.0, fill 100.0 →
# denominator 5.0, so candidate 95.0 ± 5.0·d gives deviation d exactly.


def test_model_mismatch_just_below_tolerance_adopted(caplog):
    """deviation 0.049 (candidate 94.755) → adopted, no WARNING."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "sl", 94.755, FILL, _ctx())
    assert adopted is True
    assert risk["stop_loss"] == 94.755
    assert _warnings(caplog) == []


def test_model_mismatch_just_above_tolerance_rejected(caplog):
    """deviation 0.051 (candidate 94.745) → rejected, exactly one
    WARNING carrying candidate/reference/fill/deviation/tolerance/kept
    fresh and the A3 origin field."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "sl", 94.745, FILL, _ctx())
    assert adopted is False
    assert risk["stop_loss"] == FRESH_SL

    warnings = _warnings(caplog)
    assert len(warnings) == 1, "exactly one WARNING per model_mismatch"
    msg = warnings[0].getMessage()
    assert msg.startswith("F2-gate:")
    assert "leg=sl" in msg
    assert "reason=model_mismatch" in msg
    assert "origin=strategy" in msg, "A3: origin mandatory"
    assert "candidate=94.745" in msg
    assert "reference=95.0" in msg
    assert f"fill={FILL}" in msg
    assert "deviation=0.051000" in msg
    assert "tolerance=0.05" in msg
    assert f"kept fresh={FRESH_SL}" in msg


def test_model_mismatch_pfe270_replay_rejected(caplog):
    """PFE sell 270, the lowest era override deviation (0.0616): the
    tolerance 0.05 must catch the tightest real override."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    risk["stop_loss"] = 24.3015          # fresh RiskAgent proposal
    adopted = apply_level_override(risk, "sl", 24.3236, 24.66, _ctx())
    assert adopted is False
    assert risk["stop_loss"] == 24.3015
    assert len(_warnings(caplog)) == 1
    assert "reason=model_mismatch" in caplog.text


def test_model_mismatch_atr_kept_zero_deviation_adopted(caplog):
    """candidate == reference → deviation exactly 0.0 → adopted (the
    retro control group: an untouched stop can never fail v2)."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "sl", FRESH_SL, FILL, _ctx())
    assert adopted is True
    assert risk["stop_loss"] == FRESH_SL
    assert _warnings(caplog) == []


@pytest.mark.parametrize("reference", [None, 0.0, -3.0])
def test_model_mismatch_skipped_without_usable_reference(caplog, reference):
    """reference None/<=0 → check skipped (DEBUG), candidate adopted,
    no WARNING."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    risk["stop_loss"] = reference
    adopted = apply_level_override(risk, "sl", 92.0, FILL, _ctx())
    assert adopted is True
    assert risk["stop_loss"] == 92.0
    assert _warnings(caplog) == []
    debugs = [
        r.getMessage() for r in _gate_records(caplog, logging.DEBUG)
        if "model_mismatch check skipped" in r.getMessage()
    ]
    assert len(debugs) == 1
    assert debugs[0].startswith("F2-gate:")


@pytest.mark.parametrize("reference", [100.0, 105.0])
def test_model_mismatch_degenerate_denominator_rejected(caplog, reference):
    """reference >= fill → denominator <= 0, deviation undefined:
    rejected as model_mismatch with WARNING naming the denominator."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    risk["stop_loss"] = reference
    adopted = apply_level_override(risk, "sl", 92.0, FILL, _ctx())
    assert adopted is False
    assert risk["stop_loss"] == reference

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "reason=model_mismatch" in msg
    assert "deviation=undefined" in msg
    assert f"denominator={FILL - reference}" in msg


def test_wrong_side_wins_over_model_mismatch(caplog):
    """candidate > fill AND wildly deviating → wrong_side, not
    model_mismatch (order: wrong_side precedes the invariant)."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "sl", 140.0, FILL, _ctx())
    assert adopted is False
    assert "reason=wrong_side" in caplog.text
    assert "reason=model_mismatch" not in caplog.text


def test_tp_leg_never_runs_model_mismatch(caplog):
    """sl-only scope: a TP deviating 18.4× tolerance from the fresh TP
    is still adopted — the check must not run on the tp leg."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    # |108 - 200| / (100 - 108) is not even well-formed for tp; the
    # point is that no model arithmetic runs at all on this leg.
    adopted = apply_level_override(risk, "tp", 200.0, FILL, _ctx())
    assert adopted is True
    assert risk["take_profit"] == 200.0
    assert _warnings(caplog) == []
    assert "model_mismatch" not in caplog.text


# ── T4/T5 shared fixtures: a Coordinator with mocked collaborators ──────


STRATEGY_VOTE = SimpleNamespace(
    strategy_name="Momentum",
    signal="BUY",
    confidence=80.0,
    stop_loss=105.0,    # wrong side of the 100.0 fill → must be rejected
    take_profit=112.0,  # valid → must be adopted
)


def _make_coordinator(monkeypatch):
    """Bare Coordinator (no __init__) with every collaborator mocked."""
    from orchestrator.coordinator import Coordinator

    monkeypatch.setattr(
        "orchestrator.coordinator.strategy_label", lambda t: "Momentum",
    )
    monkeypatch.setattr(
        "orchestrator.coordinator.get_days_to_earnings", lambda t: None,
    )

    coord = Coordinator.__new__(Coordinator)

    coord.regime_agent = MagicMock()
    coord.regime_agent.run.return_value = {}
    coord.regime_detector = MagicMock()

    coord.technical_agent = MagicMock()
    coord.technical_agent.run.return_value = {
        "signal": "BUY",
        "signal_id": 1,
        "indicators": {"price": 100.0, "rsi": 55.0},
        "bars": None,
        "adjusted_confidence": 0.6,
    }

    coord.risk_agent = MagicMock()
    coord.risk_agent.run.return_value = _fresh_risk()

    coord.db = MagicMock()
    coord.db.log_run.return_value = 1
    coord.db.log_combined_signal.return_value = 1
    coord.db.get_portfolio_position.return_value = None

    coord.paper_trader = MagicMock()
    coord.paper_trader.track_trade.return_value = {
        "trade_id": "t-1", "price": 100.0,
    }

    coord.signal_logger = MagicMock()
    coord._portfolio_manager = MagicMock()
    coord._portfolio_manager.can_add_position.return_value = (True, "")

    # Data feeds (async path) — all empty so no sentiment calls fire
    coord.market_data = MagicMock()
    coord.market_data.fetch.return_value = {"price": 100.0}
    for feed in (
        "news_feed", "stocktwits_feed", "reddit_feed",
        "marketaux_feed", "apewisdom_feed", "adanos_feed",
    ):
        m = MagicMock()
        m.fetch.return_value = []
        setattr(coord, feed, m)
    coord.sentiment_agent = MagicMock()

    # Instance-level overrides of pipeline internals
    coord._gather_strategy_votes = MagicMock(return_value=[STRATEGY_VOTE])
    coord._fuse_signals = MagicMock(return_value=("STRONG BUY", 0.8, "CLUSTER", None))
    coord._run_pead = MagicMock(return_value=None)
    coord._has_alpaca_position = MagicMock(return_value=False)
    coord._last_combined_signal = MagicMock(return_value="HOLD")
    coord._log_signal_event = MagicMock()

    return coord


def _assert_gated_trade(coord, caplog):
    """Common assertions: fresh SL kept, TP adopted, one F2 WARNING."""
    coord.paper_trader.track_trade.assert_called_once()
    kwargs = coord.paper_trader.track_trade.call_args.kwargs
    assert kwargs["stop_loss"] == FRESH_SL, "wrong-side SL → fresh calc kept"
    assert kwargs["take_profit"] == 112.0, "valid TP adopted"

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert msg.startswith("F2-gate:")
    assert "leg=sl" in msg and "reason=wrong_side" in msg


# ── T5: run_combined strategy-override site ──────────────────────────────


def test_run_combined_gates_strategy_override(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    coord = _make_coordinator(monkeypatch)

    # run_combined sources price from the sentiment pipeline's market info
    coord.run = MagicMock(return_value={
        "ticker": "NVDA",
        "signal": "BUY",
        "avg_score": 0.5,
        "run_id": 1,
        "market": {"price": 100.0},
    })

    result = coord.run_combined(
        "NVDA", verbose=False, account_balance=10_000.0,
        execute=True, session="US_OPEN",
    )

    _assert_gated_trade(coord, caplog)
    assert result["risk"]["stop_loss"] == FRESH_SL
    assert result["risk"]["take_profit"] == 112.0
    assert result["execution"] == {"trade_id": "t-1", "price": 100.0}


# ── T4: analyse_ticker_async strategy-override site ──────────────────────


def test_analyse_ticker_async_gates_strategy_override(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    coord = _make_coordinator(monkeypatch)

    async def _drive():
        return await coord.analyse_ticker_async(
            "NVDA",
            account_balance=10_000.0,
            execute=True,
            api_semaphore=asyncio.Semaphore(4),
            data_semaphore=asyncio.Semaphore(4),
            db_lock=asyncio.Lock(),
            session="US_OPEN",
            session_type="signal",
        )

    result = asyncio.run(_drive())

    _assert_gated_trade(coord, caplog)
    assert result["risk"]["stop_loss"] == FRESH_SL
    assert result["risk"]["take_profit"] == 112.0

    # Gated values also reach the stored forward row (signal mode)
    coord.signal_logger.store_forward_signal.assert_called_once()
    fwd_row = coord.signal_logger.store_forward_signal.call_args.args[0]
    assert fwd_row["stop_loss"] == FRESH_SL
    assert fwd_row["take_profit"] == 112.0


# ── T6: static chokepoint check (R §9c) ──────────────────────────────────


def test_coordinator_has_no_direct_risk_level_assignments():
    """coordinator.py must contain ZERO `risk["stop_loss"/"take_profit"] = ...`."""
    path = (
        Path(__file__).resolve().parent.parent
        / "orchestrator" / "coordinator.py"
    )
    tree = ast.parse(path.read_text())

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "risk"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value in ("stop_loss", "take_profit")
            ):
                offenders.append(node.lineno)

    assert offenders == [], (
        f"direct risk-level assignments found at lines {offenders}; "
        "all levels must route through level_gate.apply_level_override"
    )
