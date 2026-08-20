"""
F2 Chokepoint Gate — the single route for candidate SL/TP level adoption.

``apply_level_override`` is the ONLY path by which any candidate level
(forward-signal or strategy override) may reach ``risk["stop_loss"]`` /
``risk["take_profit"]``.  Direct assignments to those keys are banned in
orchestrator/coordinator.py (enforced by an AST test).

Adoption rule (long-only, against the executed fill):

    leg "sl": adopt iff candidate is not None AND fill_valid AND candidate < fill
    leg "tp": adopt iff candidate is not None AND fill_valid AND candidate > fill

Non-adoption keeps the fresh risk-agent calc for that leg and logs
exactly one WARNING with the literal prefix "F2-gate:" (grep continuity
with pre-gate monitoring).

Deliberate design notes (R-spec v1.1):

* ``is not None`` checks only — 0.0 is a PRESENT value.  A 0.0 SL with a
  positive fill passes ``0.0 < fill`` and IS adopted here; the
  independent downstream execution guard (coordinator.py:1788-1789)
  then blocks the trade.  That layering is intentional (R §7) — no >0
  floor belongs in this helper.
* No epsilon/tolerance anywhere.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# leg → risk dict key holding the fresh calc that non-adoption preserves
_LEG_KEYS = {"sl": "stop_loss", "tp": "take_profit"}


def apply_level_override(risk, leg, candidate, fill, ctx) -> bool:
    """
    Gate one candidate level onto the risk dict.

    Args:
        risk:      RiskAgent result dict (mutated in place on adoption).
        leg:       "sl" or "tp" — legs are gated independently.
        candidate: Proposed level (may be None).
        fill:      Executed fill price the level is validated against.
        ctx:       Dict with at minimum: ticker, session,
                   origin ("forward" | "strategy"), fill_valid (bool).

    Returns:
        True if the candidate was adopted into ``risk``; False if the
        fresh calc was kept (one WARNING logged with the reason).
    """
    key = _LEG_KEYS[leg]

    if candidate is None:
        reason = "null_level"
    elif not ctx.get("fill_valid"):
        reason = "no_fill"
    elif (candidate < fill) if leg == "sl" else (candidate > fill):
        risk[key] = candidate
        log.debug(
            "F2-gate: adopted %s ticker=%s session=%s origin=%s "
            "candidate=%s fill=%s",
            leg, ctx.get("ticker"), ctx.get("session"), ctx.get("origin"),
            candidate, fill,
        )
        return True
    else:
        reason = "wrong_side"

    log.warning(
        "F2-gate: rejected ticker=%s session=%s leg=%s reason=%s "
        "candidate=%s fill=%s kept fresh=%s",
        ctx.get("ticker"), ctx.get("session"), leg, reason,
        candidate, fill, risk.get(key),
    )
    return False
