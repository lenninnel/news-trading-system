"""
F2 Chokepoint Gate — the single route for candidate SL/TP level adoption.

``apply_level_pair_override`` is the ONLY path by which a candidate level
pair (forward-signal SL + TP) may reach ``risk["stop_loss"]`` /
``risk["take_profit"]`` from orchestrator/coordinator.py. Direct
assignments to those keys are banned there, and so is the single-leg
helper ``apply_level_override`` (both enforced by AST tests) — a
single-leg call is exactly how the two legs came to be sourced from
different runs (see "Pair rule" below).

Per-leg adoption rule (long-only, against the executed fill):

    leg "sl": adopt iff candidate is not None AND fill_valid AND candidate < fill
              AND the level-integrity invariant v2 holds
    leg "tp": adopt iff candidate is not None AND fill_valid AND candidate > fill
              AND the level-integrity invariant v2 holds

Check order per leg (R spec 2026-08-26, tp leg symmetric since 2026-09-07):

    null_level → no_fill → wrong_side → model_mismatch → adopt

``model_mismatch`` rejects a candidate whose deviation from the fresh
RiskAgent level exceeds ``_MODEL_TOLERANCE`` under the v2 drift-free
formula, on BOTH legs:

    sl: |fresh_SL − candidate| / (fill − fresh_SL)
    tp: |fresh_TP − candidate| / (fresh_TP − fill)

When no usable reference exists (fresh level None or <= 0) the check is
skipped and the candidate falls through to adopt, preserving
pre-change behaviour; the downstream execution guard
(coordinator.py, "missing/zero SL-TP" branch) still blocks execution
on a missing/<=0 level. A degenerate denominator (fresh SL at/above
the fill, fresh TP at/below the fill) rejects — adopting against an
unusable reference is the unsafe direction.

Pair rule (2026-09-07)
----------------------
Until 2026-09-07 the invariant ran on the sl leg only; the tp leg was
adopted on the side check alone. Any pre-market → open drift beyond
5 % of the stop distance therefore rejected the forward SL (fresh SL
on the open price kept) while the forward TP (set on the pre-market
price) was adopted: executed R:R 1.30–2.24 against a 2.00 model on
2026-09-01/02 (docs/DATA_INTEGRITY_2026-09-03.md). Making the tp leg
symmetric alone is not enough: with SL = P − 1.5·ATR and
TP = P + 3·ATR the tp-leg deviation is half the sl-leg deviation for
the same drift, so drifts between 7.5 % and 15 % of ATR would still
mix a fresh SL with a forward TP (CVX 2026-09-02 is such a row).

``apply_level_pair_override`` therefore evaluates both legs first and
mutates ``risk`` only when BOTH pass: either both levels come from the
forward run or both stay fresh. Never a mixed pair. The RiskAgent's
2:1 property survives to the executed trade on either path.

Logging
-------
Non-adoption logs exactly one line per leg with the literal prefix
"F2-gate:" (grep continuity with pre-gate monitoring). Level split
(amendment A2, R 2026-08-20): ``null_level`` with ctx origin
"strategy" logs at INFO; every other non-adoption (wrong_side,
no_fill, null_level with origin "forward", model_mismatch, and
pair_rejected) logs at WARNING. ``origin`` is a mandatory field in
every line (amendment A3); model_mismatch lines additionally carry
reference, deviation, and tolerance.

Reason codes:

    null_level      candidate is None
    no_fill         ctx.fill_valid is False
    wrong_side      candidate on the wrong side of the fill
    model_mismatch  invariant v2 violated (per leg — the pre-registered
                    rejection-rate watch counts these lines per origin)
    pair_rejected   this leg passed on its own but the OTHER leg was
                    rejected, so the pair stays fresh (pair function
                    only; the line names the other leg's reason)

The pair function additionally logs one INFO summary line per pair:
``F2-gate: pair … outcome=forward|fresh sl=<adopted|reason> tp=<…>``.

Deliberate design notes (R-spec v1.1, amended by the v2 invariant):

* ``is not None`` checks only — 0.0 is a PRESENT value.  A 0.0 SL with a
  positive fill passes ``0.0 < fill``; since the v2 invariant it is
  then rejected as ``model_mismatch`` whenever a usable fresh
  reference exists (its deviation is far beyond tolerance). Only when
  the reference itself is missing/<=0 does the literal adoption
  survive, and the independent downstream execution guard then blocks
  the trade.  That layering is intentional (R §7) — no >0 floor
  belongs in this helper.
* No epsilon/tolerance on the side comparisons; the single tolerance
  in this module is ``_MODEL_TOLERANCE`` for the v2 invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# leg → risk dict key holding the fresh calc that non-adoption preserves
_LEG_KEYS = {"sl": "stop_loss", "tp": "take_profit"}

# ── Level-integrity invariant v2 (R, tolerance locked 2026-08-26) ────────
#
# v2 formula: |rc.stop_loss - candidate| / (fill - rc.stop_loss);
# the fill term cancels, so drift cannot produce a false reject
# (the v1 formulation misfired on 19 of 47 era rows). The tp leg uses
# the mirror image |rc.take_profit - candidate| / (rc.take_profit - fill).
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
# to re-derive the tolerance on a real control group. Since 2026-09-07
# the tp leg produces model_mismatch lines too; count per leg
# (``leg=sl`` / ``leg=tp``) when comparing against the sl-only baseline.
#
# The invariant is a CONSISTENCY check against the risk model, NOT
# budget protection: the 10% portfolio cap bound 100% of era
# positions, so stop_pct never drove sizing. Hence model_mismatch,
# not budget_mismatch.
_MODEL_TOLERANCE = 0.05


@dataclass
class _Verdict:
    """Outcome of evaluating one leg (no mutation, no logging yet)."""
    leg: str
    candidate: object
    reason: str | None            # None → the leg passes
    level: int = logging.WARNING  # log level for a non-adoption line
    extra: str = ""               # appended to model_mismatch lines


def _evaluate_leg(risk, leg, candidate, fill, ctx) -> _Verdict:
    """Run null_level → no_fill → wrong_side → model_mismatch for one leg."""
    key = _LEG_KEYS[leg]

    if candidate is None:
        level = (
            logging.INFO if ctx.get("origin") == "strategy" else logging.WARNING
        )
        return _Verdict(leg, candidate, "null_level", level)
    if not ctx.get("fill_valid"):
        return _Verdict(leg, candidate, "no_fill")
    on_side = (candidate < fill) if leg == "sl" else (candidate > fill)
    if not on_side:
        return _Verdict(leg, candidate, "wrong_side")

    reference = risk.get(key)
    if reference is None or reference <= 0:
        # No usable reference — deviation undefined; fall through to
        # adopt (pre-invariant behaviour; downstream guard blocks
        # execution on a missing/<=0 resulting level).
        log.debug(
            "F2-gate: model_mismatch check skipped ticker=%s session=%s "
            "origin=%s leg=%s reference=%s (no usable reference)",
            ctx.get("ticker"), ctx.get("session"), ctx.get("origin"), leg,
            reference,
        )
        return _Verdict(leg, candidate, None)

    denominator = (fill - reference) if leg == "sl" else (reference - fill)
    if denominator <= 0:
        # Fresh level on the wrong side of the fill — broken reference;
        # adopting against it is the unsafe direction.
        return _Verdict(
            leg, candidate, "model_mismatch", logging.WARNING,
            f"reference={reference} deviation=undefined "
            f"denominator={denominator} tolerance={_MODEL_TOLERANCE}",
        )
    deviation = abs(reference - candidate) / denominator
    if deviation > _MODEL_TOLERANCE:
        return _Verdict(
            leg, candidate, "model_mismatch", logging.WARNING,
            f"reference={reference} deviation={deviation:.6f} "
            f"tolerance={_MODEL_TOLERANCE}",
        )
    return _Verdict(leg, candidate, None)


def _log_rejection(risk, verdict: _Verdict, fill, ctx, *, reason=None,
                   level=None, extra="") -> None:
    """One line per non-adopted leg; fields per amendment A3."""
    key = _LEG_KEYS[verdict.leg]
    reason = reason or verdict.reason
    fields = f" {extra}" if extra else (
        f" {verdict.extra}" if verdict.extra else ""
    )
    log.log(
        level if level is not None else verdict.level,
        "F2-gate: rejected ticker=%s session=%s origin=%s leg=%s reason=%s "
        "candidate=%s%s fill=%s kept fresh=%s",
        ctx.get("ticker"), ctx.get("session"), ctx.get("origin"),
        verdict.leg, reason, verdict.candidate, fields, fill, risk.get(key),
    )


def _adopt(risk, verdict: _Verdict, fill, ctx) -> None:
    """Mutate one leg and log the adoption at INFO (was DEBUG until
    2026-09-03 — the invisible tp-leg adoption hid the R:R spread of
    2026-09-01/02; the journal must show which leg came from where)."""
    key = _LEG_KEYS[verdict.leg]
    fresh = risk.get(key)
    risk[key] = verdict.candidate
    log.info(
        "F2-gate: adopted ticker=%s session=%s origin=%s leg=%s "
        "candidate=%s fill=%s replaced fresh=%s",
        ctx.get("ticker"), ctx.get("session"), ctx.get("origin"), verdict.leg,
        verdict.candidate, fill, fresh,
    )


def apply_level_override(risk, leg, candidate, fill, ctx) -> bool:
    """
    Gate ONE candidate level onto the risk dict (per-leg primitive).

    Not for use from the coordinator: gating legs independently is what
    produced mixed SL/TP sources. Use :func:`apply_level_pair_override`
    there. Kept as the tested primitive the pair function is built on.

    Args:
        risk:      RiskAgent result dict (mutated in place on adoption).
        leg:       "sl" or "tp".
        candidate: Proposed level (may be None).
        fill:      Executed fill price the level is validated against.
        ctx:       Dict with at minimum: ticker, session,
                   origin ("forward" | "strategy"), fill_valid (bool).

    Returns:
        True if the candidate was adopted into ``risk``; False if the
        fresh calc was kept (one INFO/WARNING logged with the reason).
    """
    verdict = _evaluate_leg(risk, leg, candidate, fill, ctx)
    if verdict.reason is not None:
        _log_rejection(risk, verdict, fill, ctx)
        return False
    _adopt(risk, verdict, fill, ctx)
    return True


def apply_level_pair_override(risk, sl_candidate, tp_candidate, fill, ctx) -> bool:
    """
    Gate a candidate SL/TP PAIR onto the risk dict — all or nothing.

    Both legs are evaluated with the per-leg rule (incl. invariant v2 on
    each). ``risk`` is mutated only if both pass; otherwise both fresh
    RiskAgent levels are kept, so SL and TP always share one source.

    Logging: one line per leg as in the single-leg helper (a leg that
    passed on its own but is held back by the other leg logs
    ``reason=pair_rejected other_leg=<reason>`` at WARNING), plus one
    INFO summary line ``F2-gate: pair … outcome=forward|fresh``.

    Returns:
        True if both candidates were adopted; False if both fresh levels
        were kept.
    """
    sl = _evaluate_leg(risk, "sl", sl_candidate, fill, ctx)
    tp = _evaluate_leg(risk, "tp", tp_candidate, fill, ctx)

    if sl.reason is None and tp.reason is None:
        _adopt(risk, sl, fill, ctx)
        _adopt(risk, tp, fill, ctx)
        outcome, sl_word, tp_word = "forward", "adopted", "adopted"
    else:
        for own, other in ((sl, tp), (tp, sl)):
            if own.reason is not None:
                _log_rejection(risk, own, fill, ctx)
            else:
                _log_rejection(
                    risk, own, fill, ctx, reason="pair_rejected",
                    level=logging.WARNING,
                    extra=f"other_leg={other.leg}:{other.reason}",
                )
        outcome = "fresh"
        sl_word = sl.reason or "pair_rejected"
        tp_word = tp.reason or "pair_rejected"

    log.info(
        "F2-gate: pair ticker=%s session=%s origin=%s outcome=%s sl=%s tp=%s "
        "fill=%s stop_loss=%s take_profit=%s",
        ctx.get("ticker"), ctx.get("session"), ctx.get("origin"), outcome,
        sl_word, tp_word, fill, risk.get("stop_loss"), risk.get("take_profit"),
    )
    return outcome == "forward"
