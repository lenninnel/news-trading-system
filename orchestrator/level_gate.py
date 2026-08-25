"""
F2 Chokepoint Gate — the single route for candidate SL/TP level adoption.

``apply_level_override`` is the ONLY path by which any candidate level
(forward-signal or strategy override) may reach ``risk["stop_loss"]`` /
``risk["take_profit"]``.  Direct assignments to those keys are banned in
orchestrator/coordinator.py (enforced by an AST test).

Adoption rule (long-only, against the executed fill):

    leg "sl": adopt iff candidate is not None AND fill_valid AND candidate < fill
              AND the level-integrity invariant v2 holds (see below)
    leg "tp": adopt iff candidate is not None AND fill_valid AND candidate > fill

Check order on the sl leg (R spec 2026-08-26):

    null_level → no_fill → wrong_side → model_mismatch → adopt

``model_mismatch`` (sl leg ONLY — TP is out of scope this window)
rejects a candidate whose deviation from the fresh RiskAgent level
exceeds ``_MODEL_TOLERANCE`` under the v2 drift-free formula. When no
usable reference exists (fresh level None or <= 0) the check is
skipped and the candidate falls through to adopt, preserving
pre-change behaviour; the downstream execution guard
(coordinator.py:1788-1789) still blocks execution on a missing/<=0
level. A degenerate denominator (fresh level at or above the fill)
rejects — adopting against an unusable reference is the unsafe
direction.

Non-adoption keeps the fresh risk-agent calc for that leg and logs
exactly one line with the literal prefix "F2-gate:" (grep continuity
with pre-gate monitoring). Level split (amendment A2, R 2026-08-20):
``null_level`` with ctx origin "strategy" logs at INFO (expected —
strategies routinely emit no levels; R tracks the weekly INFO count as
a research signal); every other non-adoption (wrong_side, no_fill,
null_level with origin "forward", and model_mismatch — never a
by-design absence) logs at WARNING. ``origin`` is a mandatory field
in every non-adoption line (amendment A3); model_mismatch lines
additionally carry reference, deviation, and tolerance.

Deliberate design notes (R-spec v1.1, amended by the v2 invariant):

* ``is not None`` checks only — 0.0 is a PRESENT value.  A 0.0 SL with a
  positive fill passes ``0.0 < fill``; since the v2 invariant it is
  then rejected as ``model_mismatch`` whenever a usable fresh
  reference exists (its deviation is far beyond tolerance). Only when
  the reference itself is missing/<=0 does the literal adoption
  survive, and the independent downstream execution guard
  (coordinator.py:1788-1789) then blocks the trade.  That layering is
  intentional (R §7) — no >0 floor belongs in this helper.
* No epsilon/tolerance on the side comparisons; the single tolerance
  in this module is ``_MODEL_TOLERANCE`` for the v2 invariant.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# leg → risk dict key holding the fresh calc that non-adoption preserves
_LEG_KEYS = {"sl": "stop_loss", "tp": "take_profit"}

# ── Level-integrity invariant v2 (R, tolerance locked 2026-08-26) ────────
#
# v2 formula: |rc.stop_loss - candidate| / (fill - rc.stop_loss);
# the fill term cancels, so drift cannot produce a false reject
# (the v1 formulation misfired on 19 of 47 era rows).
#
# Derivation of 0.05 (R, after the forward-path retro):
# In the era, ZERO of 181 candidates (44 override stops + 134 forward
# levels + 3 ATR-kept) originated from the risk model on the override
# path; all non-ATR candidates reconstruct as (2dp price) x 0.98,
# frozen and carried up to 34 days. The invariant therefore rejects
# ~95% on the forward path and 44/44 on the override path - this is
# the intended outcome, not collateral damage. The "perfect
# separation at 0.0616" from the first retro was a 3-row sampling
# artifact; 6 override rows fall below it, down to 0.0191. 0.05 is
# chosen at the tight end on error asymmetry (false reject = fresh
# RiskAgent level, the intended fallback; false accept = VRT class),
# NOT because a separation point was measured. Its discriminating
# power is untested and only becomes testable once the override is
# removed and genuine ATR levels appear on the forward path.
#
# PRE-REGISTERED POST-DEPLOY WATCH (R, 2026-08-26): monitor the
# model_mismatch rejection rate per origin. Expectation ~95% forward,
# ~100% strategy override. A material drop in forward rejection means
# genuine ATR candidates have started appearing - that is the trigger
# to re-derive the tolerance on a real control group.
#
# The invariant is a CONSISTENCY check against the risk model, NOT
# budget protection: the 10% portfolio cap bound 100% of era
# positions, so stop_pct never drove sizing. Hence model_mismatch,
# not budget_mismatch.
_MODEL_TOLERANCE = 0.05


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
        fresh calc was kept (one INFO/WARNING logged with the reason —
        see the module docstring for the A2 level split).
    """
    key = _LEG_KEYS[leg]

    if candidate is None:
        reason = "null_level"
    elif not ctx.get("fill_valid"):
        reason = "no_fill"
    elif (candidate < fill) if leg == "sl" else (candidate > fill):
        if leg == "sl":
            reference = risk.get(key)
            if reference is None or reference <= 0:
                # No usable reference — deviation undefined; fall through
                # to adopt (pre-invariant behaviour; downstream guard
                # blocks execution on a missing/<=0 resulting level).
                log.debug(
                    "F2-gate: model_mismatch check skipped ticker=%s "
                    "session=%s origin=%s leg=%s reference=%s "
                    "(no usable reference)",
                    ctx.get("ticker"), ctx.get("session"),
                    ctx.get("origin"), leg, reference,
                )
            else:
                denominator = fill - reference
                if denominator <= 0:
                    # Fresh level at/above the fill — broken reference;
                    # adopting against it is the unsafe direction.
                    log.warning(
                        "F2-gate: rejected ticker=%s session=%s origin=%s "
                        "leg=%s reason=model_mismatch candidate=%s "
                        "reference=%s fill=%s deviation=undefined "
                        "denominator=%s tolerance=%s kept fresh=%s",
                        ctx.get("ticker"), ctx.get("session"),
                        ctx.get("origin"), leg, candidate, reference, fill,
                        denominator, _MODEL_TOLERANCE, risk.get(key),
                    )
                    return False
                deviation = abs(reference - candidate) / denominator
                if deviation > _MODEL_TOLERANCE:
                    log.warning(
                        "F2-gate: rejected ticker=%s session=%s origin=%s "
                        "leg=%s reason=model_mismatch candidate=%s "
                        "reference=%s fill=%s deviation=%.6f tolerance=%s "
                        "kept fresh=%s",
                        ctx.get("ticker"), ctx.get("session"),
                        ctx.get("origin"), leg, candidate, reference, fill,
                        deviation, _MODEL_TOLERANCE, risk.get(key),
                    )
                    return False
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

    level = (
        logging.INFO
        if reason == "null_level" and ctx.get("origin") == "strategy"
        else logging.WARNING
    )
    log.log(
        level,
        "F2-gate: rejected ticker=%s session=%s origin=%s leg=%s reason=%s "
        "candidate=%s fill=%s kept fresh=%s",
        ctx.get("ticker"), ctx.get("session"), ctx.get("origin"), leg, reason,
        candidate, fill, risk.get(key),
    )
    return False
