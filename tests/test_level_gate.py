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
    T4. Integration regression, former strategy-override site in
        analyse_ticker_async: neither the strategy SL nor TP candidate
        reaches the gate (strategy override fully removed 2026-09-01 —
        the RiskAgent is the only level source on the analysis paths);
        track_trade AND the stored forward signal both receive the
        fresh RiskAgent values.
    T5. Same regression shape for the former strategy-override site in
        run_combined.
    T6. Static chokepoint check: coordinator.py contains ZERO direct
        assignments to risk["stop_loss"] / risk["take_profit"] and
        ZERO calls to the single-leg helper (pair function only).
    T7. Level-integrity invariant v2 (model_mismatch, R spec
        2026-08-26): tolerance boundary both sides, PFE-270 replay,
        ATR-kept zero deviation, missing/<=0 reference skip path,
        degenerate denominator, wrong_side precedence — on BOTH legs
        (tp leg symmetric since 2026-09-07).
    T8. Pair rule (2026-09-07): apply_level_pair_override adopts both
        legs or neither; pair_rejected reason code and summary line;
        replay of the seven executed BUYs 2026-09-01..04 whose mixed
        SL/TP sources produced the R:R spread (docs/
        DATA_INTEGRITY_2026-09-03.md) — every pair now lands at the
        fresh 2:1 model or wholly on the forward run.

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

from orchestrator.level_gate import (
    apply_level_override,
    apply_level_pair_override,
)


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
        # (adopt candidate 108.3: deviation |108-108.3|/(108-100) = 0.0375 ≤ 0.05)
        ("tp", 95.0,  True,  False, "wrong_side"),
        ("tp", None,  True,  False, "null_level"),
        ("tp", 108.3, True,  True,  None),
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


@pytest.mark.parametrize("leg,candidate", [("sl", 94.8), ("tp", 108.3)])
def test_adoption_emits_no_warning(caplog, leg, candidate):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, leg, candidate, FILL, _ctx())
    assert adopted is True
    assert _warnings(caplog) == []
    # Adoption logs at INFO (2026-09-03: was DEBUG — the invisible TP-leg
    # adoption hid the R:R spread of 2026-09-01/02); never WARNING.
    for r in caplog.records:
        if r.name == "orchestrator.level_gate":
            assert r.levelno <= logging.INFO
            assert "adopted" in r.getMessage()


# ── T7: level-integrity invariant v2 (model_mismatch) ────────────────────
#
# sl: deviation = |reference − candidate| / (fill − reference)
# tp: deviation = |reference − candidate| / (reference − fill)
# _MODEL_TOLERANCE = 0.05. Fixtures: fresh SL 95.0, fill 100.0 →
# denominator 5.0, so candidate 95.0 ± 5.0·d gives deviation d exactly;
# fresh TP 108.0 → denominator 8.0, candidate 108.0 ± 8.0·d.


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


def test_tp_leg_runs_model_mismatch(caplog):
    """Symmetric since 2026-09-07: a TP deviating 11.5× tolerance from
    the fresh TP is rejected as model_mismatch (was adopted on the side
    check alone — the mechanism behind the 2026-09-01/02 R:R spread)."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "tp", 200.0, FILL, _ctx())
    assert adopted is False
    assert risk["take_profit"] == FRESH_TP
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "leg=tp" in msg
    assert "reason=model_mismatch" in msg
    assert "reference=108.0" in msg
    assert "deviation=11.500000" in msg
    assert "tolerance=0.05" in msg
    assert f"kept fresh={FRESH_TP}" in msg


@pytest.mark.parametrize("candidate,adopt", [
    (108.39, True),     # deviation 0.04875 → adopted
    (107.61, True),     # deviation 0.04875 the other way (TP below fresh, still > fill)
    (108.41, False),    # deviation 0.05125 → rejected
    (107.59, False),    # deviation 0.05125 → rejected
])
def test_tp_model_mismatch_boundary(caplog, candidate, adopt):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_override(risk, "tp", candidate, FILL, _ctx())
    assert adopted is adopt
    assert risk["take_profit"] == (candidate if adopt else FRESH_TP)
    assert (len(_warnings(caplog)) == 0) is adopt


@pytest.mark.parametrize("reference", [100.0, 95.0])
def test_tp_model_mismatch_degenerate_denominator_rejected(caplog, reference):
    """fresh TP at/below the fill → denominator <= 0 → rejected."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    risk["take_profit"] = reference
    adopted = apply_level_override(risk, "tp", 112.0, FILL, _ctx())
    assert adopted is False
    assert risk["take_profit"] == reference
    msg = _warnings(caplog)[0].getMessage()
    assert "reason=model_mismatch" in msg
    assert "deviation=undefined" in msg
    assert f"denominator={reference - FILL}" in msg


@pytest.mark.parametrize("reference", [None, 0.0])
def test_tp_model_mismatch_skipped_without_usable_reference(caplog, reference):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    risk["take_profit"] = reference
    adopted = apply_level_override(risk, "tp", 112.0, FILL, _ctx())
    assert adopted is True
    assert risk["take_profit"] == 112.0
    assert _warnings(caplog) == []
    assert "model_mismatch check skipped" in caplog.text


# ── T8: pair rule — both legs from one source, never mixed ───────────────


def _pair_lines(caplog):
    return [
        r.getMessage() for r in caplog.records
        if r.name == "orchestrator.level_gate" and r.getMessage().startswith("F2-gate: pair")
    ]


def test_pair_both_pass_adopts_both(caplog):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_pair_override(risk, 94.8, 108.3, FILL, _ctx(origin="forward"))
    assert adopted is True
    assert risk["stop_loss"] == 94.8
    assert risk["take_profit"] == 108.3
    assert _warnings(caplog) == []
    infos = [r.getMessage() for r in _gate_records(caplog, logging.INFO)]
    assert sum("adopted" in m and "leg=sl" in m for m in infos) == 1
    assert sum("adopted" in m and "leg=tp" in m for m in infos) == 1
    (pair,) = _pair_lines(caplog)
    assert "outcome=forward sl=adopted tp=adopted" in pair
    assert "origin=forward" in pair


@pytest.mark.parametrize(
    "sl_cand,tp_cand,sl_reason,tp_reason",
    [
        # sl fails invariant, tp would pass on its own → both fresh
        (94.5, 108.3, "model_mismatch", "pair_rejected"),
        # tp fails invariant, sl would pass → both fresh (the case a
        # per-leg symmetric gate would still mix; CVX 2026-09-02 shape)
        (94.8, 109.0, "pair_rejected", "model_mismatch"),
        # sl wrong side, tp fine
        (101.0, 108.3, "wrong_side", "pair_rejected"),
        # tp wrong side, sl fine
        (94.8, 99.0, "pair_rejected", "wrong_side"),
        # sl missing, tp fine
        (None, 108.3, "null_level", "pair_rejected"),
        # both fail on their own
        (94.5, 109.0, "model_mismatch", "model_mismatch"),
    ],
)
def test_pair_any_rejection_keeps_both_fresh(caplog, sl_cand, tp_cand, sl_reason, tp_reason):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_pair_override(risk, sl_cand, tp_cand, FILL, _ctx(origin="forward"))
    assert adopted is False
    assert risk["stop_loss"] == FRESH_SL, "no mixed pair: fresh SL kept"
    assert risk["take_profit"] == FRESH_TP, "no mixed pair: fresh TP kept"

    warnings = [r.getMessage() for r in _warnings(caplog)]
    assert len(warnings) == 2, "exactly one line per leg"
    sl_line = next(m for m in warnings if " leg=sl " in m)
    tp_line = next(m for m in warnings if " leg=tp " in m)
    assert f"reason={sl_reason}" in sl_line
    assert f"reason={tp_reason}" in tp_line
    for m in (sl_line, tp_line):
        assert m.startswith("F2-gate: rejected")
        assert "origin=forward" in m
    if sl_reason == "pair_rejected":
        assert f"other_leg=tp:{tp_reason}" in sl_line
    if tp_reason == "pair_rejected":
        assert f"other_leg=sl:{sl_reason}" in tp_line
    # No adoption line may appear for a leg that was held back
    assert not any("adopted" in r.getMessage() for r in _gate_records(caplog, logging.INFO))
    (pair,) = _pair_lines(caplog)
    assert f"outcome=fresh sl={sl_reason} tp={tp_reason}" in pair


def test_pair_no_fill_rejects_both(caplog):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    adopted = apply_level_pair_override(
        risk, 94.8, 108.3, FILL, _ctx(fill_valid=False, origin="forward"),
    )
    assert adopted is False
    assert risk["stop_loss"] == FRESH_SL and risk["take_profit"] == FRESH_TP
    assert all("reason=no_fill" in r.getMessage() for r in _warnings(caplog))
    assert len(_warnings(caplog)) == 2


def test_pair_model_mismatch_lines_count_per_leg(caplog):
    """The pre-registered rejection-rate watch greps model_mismatch
    lines; a pair with both legs beyond tolerance yields exactly one
    such line per leg, distinguishable by leg=sl / leg=tp."""
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    apply_level_pair_override(risk, 90.0, 120.0, FILL, _ctx(origin="forward"))
    mm = [r.getMessage() for r in _warnings(caplog) if "reason=model_mismatch" in r.getMessage()]
    assert len(mm) == 2
    assert sum(" leg=sl " in m for m in mm) == 1
    assert sum(" leg=tp " in m for m in mm) == 1


# Replay of the executed BUYs 2026-09-01..04 (prod DB, read-only, 2026-09-07):
# forward SL/TP from the US_PRE risk_calculations row (stored on the
# forward_signals row), P_open = the 14:30 US_OPEN quote the gate is run
# against in production (trade_history.intended_price) with the fresh
# SL/TP of that run; fill = trade_history.executed_price. Under the old
# per-leg gate the first six executed with a fresh SL + forward TP (R:R
# 1.30–2.24 vs fill); TSLA adopted both forward legs. Columns: expected
# pair outcome under the pair rule and the executed R:R that actually
# happened, (TP − fill) / (fill − SL).
_REPLAY = [
    # ticker, fwd_sl,   fwd_tp,   fresh_sl, fresh_tp, p_open,  fill,   outcome,   rr_old
    ("MSFT",  483.5215, 537.7570, 486.1315, 540.3670, 504.210, 504.21, "fresh",   1.856),
    ("XOM",   159.2886, 174.9828, 157.8686, 173.5628, 163.100, 163.15, "fresh",   2.240),
    ("CVX",   204.9480, 223.1641, 205.5280, 223.7441, 211.600, 211.59, "fresh",   1.909),
    ("BAC",    60.4357,  65.1585,  61.5207,  66.2435,  63.095,  63.10, "fresh",   1.303),
    ("TSLA",  335.4449, 397.3502, 334.5049, 396.4102, 355.140, 355.28, "forward", 2.121),
    ("VRT",   247.4816, 311.5567, 253.1316, 317.2067, 274.490, 274.88, "fresh",   1.687),
    ("AMZN",  249.5414, 280.0172, 247.1614, 277.6372, 257.320, 257.41, "fresh",   2.206),
]


def _rr(sl, tp, price):
    return (tp - price) / (price - sl)


@pytest.mark.parametrize(
    "ticker,fwd_sl,fwd_tp,fresh_sl,fresh_tp,p_open,fill,outcome,rr_old", _REPLAY,
    ids=[r[0] for r in _REPLAY],
)
def test_pair_replay_2026_09(caplog, ticker, fwd_sl, fwd_tp, fresh_sl, fresh_tp,
                             p_open, fill, outcome, rr_old):
    caplog.set_level(logging.DEBUG, logger="orchestrator.level_gate")
    risk = _fresh_risk()
    risk["stop_loss"], risk["take_profit"] = fresh_sl, fresh_tp
    ctx = {"ticker": ticker, "session": "US_OPEN", "origin": "forward", "fill_valid": True}

    # fixture sanity: the executed pair really had the R:R from the report
    executed = (fresh_sl, fwd_tp) if outcome == "fresh" else (fwd_sl, fwd_tp)
    assert _rr(*executed, fill) == pytest.approx(rr_old, abs=0.001)
    # and the fresh 14:30 pair is the 2:1 model on its own quote
    assert _rr(fresh_sl, fresh_tp, p_open) == pytest.approx(2.0, abs=0.001)

    adopted = apply_level_pair_override(risk, fwd_sl, fwd_tp, p_open, ctx)

    assert adopted is (outcome == "forward")
    expected = (fwd_sl, fwd_tp) if outcome == "forward" else (fresh_sl, fresh_tp)
    assert (risk["stop_loss"], risk["take_profit"]) == expected
    # One source only — the mixed pair that actually executed is impossible now
    assert (risk["stop_loss"], risk["take_profit"]) != (fresh_sl, fwd_tp)

    rr_new = _rr(risk["stop_loss"], risk["take_profit"], fill)
    if outcome == "fresh":
        # 2:1 on the quote; what remains vs the fill is pure fill slippage
        assert abs(rr_new - 2.0) <= abs(_rr(fresh_sl, fresh_tp, fill) - 2.0) + 1e-12
        assert abs(rr_new - 2.0) < 0.06
    else:
        assert rr_new == pytest.approx(rr_old, abs=0.001)


# ── T4/T5 shared fixtures: a Coordinator with mocked collaborators ──────


STRATEGY_VOTE = SimpleNamespace(
    strategy_name="Momentum",
    signal="BUY",
    confidence=80.0,
    stop_loss=105.0,    # must never reach the gate — strategy override removed 2026-09-01
    take_profit=112.0,  # must never reach the gate either
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
    """Common assertions: fresh SL and TP untouched (the strategy override
    was fully removed 2026-09-01 — neither candidate reaches the gate, so
    no F2 line at all on the analysis paths)."""
    coord.paper_trader.track_trade.assert_called_once()
    kwargs = coord.paper_trader.track_trade.call_args.kwargs
    assert kwargs["stop_loss"] == FRESH_SL, "RiskAgent SL is the only stop source"
    assert kwargs["take_profit"] == FRESH_TP, "RiskAgent TP is the only TP source"

    assert _warnings(caplog) == [], "no candidates offered → no F2 WARNING"
    assert "F2-gate" not in caplog.text, "strategy levels must never reach the gate"


# ── T5: run_combined strategy-override site ──────────────────────────────


def test_run_combined_strategy_override_removed(monkeypatch, caplog):
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
    assert result["risk"]["take_profit"] == FRESH_TP
    assert result["execution"] == {"trade_id": "t-1", "price": 100.0}


# ── T4: analyse_ticker_async strategy-override site ──────────────────────


def test_analyse_ticker_async_strategy_override_removed(monkeypatch, caplog):
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
    assert result["risk"]["take_profit"] == FRESH_TP

    # Gated values also reach the stored forward row (signal mode)
    coord.signal_logger.store_forward_signal.assert_called_once()
    fwd_row = coord.signal_logger.store_forward_signal.call_args.args[0]
    assert fwd_row["stop_loss"] == FRESH_SL
    assert fwd_row["take_profit"] == FRESH_TP


# ── T6: static chokepoint check (R §9c) ──────────────────────────────────


def _coordinator_tree():
    path = (
        Path(__file__).resolve().parent.parent
        / "orchestrator" / "coordinator.py"
    )
    return ast.parse(path.read_text())


def test_coordinator_has_no_direct_risk_level_assignments():
    """coordinator.py must contain ZERO `risk["stop_loss"/"take_profit"] = ...`."""
    offenders = []
    for node in ast.walk(_coordinator_tree()):
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
        "all levels must route through level_gate.apply_level_pair_override"
    )


def test_coordinator_uses_only_the_pair_gate():
    """No single-leg gate call in coordinator.py: gating legs independently
    is what mixed a fresh SL with a forward TP (2026-09-01/02). Exactly
    the pair helper, called at least once."""
    single, pair = [], []
    for node in ast.walk(_coordinator_tree()):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == "apply_level_override":
                single.append(node.lineno)
            elif name == "apply_level_pair_override":
                pair.append(node.lineno)
    assert single == [], f"single-leg gate calls at lines {single}"
    assert pair, "the forward site must route through apply_level_pair_override"
