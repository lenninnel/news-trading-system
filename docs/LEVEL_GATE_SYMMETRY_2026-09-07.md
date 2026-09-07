# F2 level gate — symmetric pair rule, 2026-09-07

Follow-up to the R:R finding in `docs/DATA_INTEGRITY_2026-09-03.md` §3.
No change to signal selection, sizing, or the RiskAgent; only the
adoption of forward-signal levels in the cached US_OPEN executor
(`coordinator._execute_forward_signals_async`, step 5) changes.

## What was wrong

`level_gate.apply_level_override` gated each leg on its own. The sl leg
ran the level-integrity invariant v2 (`|fresh − candidate| / (fill −
fresh) ≤ 0.05`); the tp leg only checked `candidate > fill`. Any
pre-market → open drift above 5 % of the stop distance therefore kept
the **fresh SL** (set on the 14:30 quote) while adopting the **forward
TP** (set on the 13:1x pre-market quote). The drift landed in the R:R:

    R:R = (P_pre + 3·ATR − P_open) / (1.5·ATR) = 2 − drift / (1.5·ATR)

Making the tp leg symmetric on its own is not enough. With SL = P −
1.5·ATR and TP = P + 3·ATR the tp deviation is exactly half the sl
deviation for the same drift, so drifts between 7.5 % and 15 % of ATR
would still pass the TP and reject the SL — a mixed pair again (CVX
2026-09-02: sl 0.0955, tp 0.0478).

## What changed

`orchestrator/level_gate.py`

* `_evaluate_leg` — pure per-leg evaluation, same check order
  `null_level → no_fill → wrong_side → model_mismatch`, now on **both
  legs**. tp formula: `|fresh_TP − candidate| / (fresh_TP − fill)`,
  same tolerance 0.05, same skip path (no usable reference) and the
  same degenerate-denominator rejection (fresh TP at/below the fill).
* `apply_level_pair_override(risk, sl, tp, fill, ctx)` — evaluates both
  legs first, mutates `risk` only if **both** pass. Either both levels
  come from the forward run or both stay fresh. Never mixed.
* `apply_level_override` (single leg) stays as the tested primitive
  but is banned from `coordinator.py` (AST test
  `test_coordinator_uses_only_the_pair_gate`), next to the existing ban
  on direct `risk["stop_loss"/"take_profit"]` assignments.

`orchestrator/coordinator.py` — the forward site calls the pair helper
once per pending forward row (first match wins, as before).

### Logging (journal contract)

* Per leg, unchanged prefix `F2-gate: rejected … leg=sl|tp reason=…`.
  `model_mismatch` lines keep `reference= deviation= tolerance=`; the
  pre-registered rejection-rate watch (R, 2026-08-26) keeps working —
  count per leg now, since the tp leg produces these lines too.
* New reason code `pair_rejected` with `other_leg=<leg>:<reason>` for a
  leg that passed on its own but was held back by the other leg.
* New INFO summary per pair:
  `F2-gate: pair ticker=… origin=forward outcome=forward|fresh sl=<adopted|reason> tp=<…> fill=… stop_loss=… take_profit=…`
* Adoption lines stay at INFO (since 2026-09-03).

Post-deploy grep:

```bash
journalctl --user -u nts-trading --since today | grep "F2-gate: pair"
journalctl --user -u nts-trading --since today | grep "reason=model_mismatch" | grep -c " leg=sl "
journalctl --user -u nts-trading --since today | grep "reason=model_mismatch" | grep -c " leg=tp "
```

## Replay: the executed BUYs 2026-09-01..04 under the pair rule

Prod DB read-only, 2026-09-07. Forward SL/TP = the US_PRE
`risk_calculations` row stored on `forward_signals`; fresh SL/TP and
P_open = the 14:30 US_OPEN row (`trade_history.intended_price`, the
price the gate runs against); fill = `trade_history.executed_price`.
dev = invariant v2 deviation per leg (tolerance 0.05).

| trade | dev sl | dev tp | pair outcome | executed SL / TP (old gate) | R:R executed | SL / TP under pair rule | R:R vs fill | R:R vs P_open |
|---|---|---|---|---|---|---|---|---|
| MSFT 09-01 | 0.144 | 0.072 | both fresh | 486.1315 / **537.7570** | **1.856** | 486.1315 / 540.3670 | 2.000 | 2.000 |
| XOM 09-02 | 0.271 | 0.136 | both fresh | 157.8686 / **174.9828** | **2.240** | 157.8686 / 173.5628 | 1.972 | 2.000 |
| CVX 09-02 | 0.096 | 0.048 | both fresh | 205.5280 / **223.1641** | **1.909** | 205.5280 / 223.7441 | 2.005 | 2.000 |
| BAC 09-02 | 0.689 | 0.345 | both fresh | 61.5207 / **65.1585** | **1.303** | 61.5207 / 66.2435 | 1.990 | 2.000 |
| TSLA 09-02 | 0.046 | 0.023 | both forward | 335.4449 / 397.3502 | 2.121 | 335.4449 / 397.3502 (unchanged) | 2.121 | 2.143 |
| VRT 09-04 | 0.265 | 0.132 | both fresh | 253.1316 / **311.5567** | **1.686** | 253.1316 / 317.2067 | 1.946 | 2.000 |
| AMZN 09-04 | 0.234 | 0.117 | both fresh | 247.1614 / **280.0172** | **2.206** | 247.1614 / 277.6372 | 1.974 | 2.000 |

Bold = the forward TP that was adopted next to a fresh SL. CVX is the
row a per-leg symmetric gate would still have mixed (tp 0.048 ≤ 0.05,
sl 0.096 > 0.05); the pair rule keeps both fresh.

What remains between "R:R vs P_open" (exactly 2.000 on every fresh pair)
and "R:R vs fill" is fill slippage against the 14:30 quote — up to
$0.39 (VRT), i.e. ≤ 0.054 R:R. TSLA is the only pair that adopts the
forward run entirely; its R:R against the fill is the forward model's
2:1 minus the same slippage effect. Re-anchoring levels to the actual
fill remains out of scope (as stated on 2026-09-03).

Note: the RiskAgent's 1 % stop-distance floor (`_ATR_STOP_FLOOR_PCT`)
can make its own R:R < 2 on flat tape; none of the seven rows hit it.

## Other sites checked for the same asymmetry

Every other SL/TP write in the repo takes both legs from one source:
RiskAgent result dict (analysis paths, strategy override removed
2026-09-01), one `trade_history` row (PositionManager, API, price
monitor), one forward row (US_PRE storage). The PositionManager's
trailing stop moves the SL alone by design. The two calls at the
forward site were the only place two runs could meet.

Tests: `tests/test_level_gate.py` (T7 on both legs, T8 pair rule and
the seven-row replay), `tests/test_coordinator.py::TestF2ForwardOverrideSanityGate`
(2026-07-02 replay now ends in both-fresh).
