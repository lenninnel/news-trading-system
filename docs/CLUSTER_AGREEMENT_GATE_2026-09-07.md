# Cluster agreement gate, 2026-09-07

Last open change in the 2026-09 series (Wilder-ATR 4a4bbe7 → OHLC
freshness 8f9b312 → alerting fccdaa6 → fill reconciliation 677c120 → F2
pair gate e096481) and the only one that changes *which* trades come
into existence. No change to MIN_CONFIDENCE (0.35), the +0.10 boost,
the RiskAgent, the level gate or the PEAD path.

## 1. What was wrong

`orchestrator/cluster_detector.py` filtered each vote on
`MIN_CONFIDENCE = 0.35` and then emitted the strongest vote on the
winning side as the "cluster" verdict — it never checked that more than
one strategy was on that side. A single Momentum BUY at 52 % went out
as `signal_path=CLUSTER`, `Combined BUY`, and traded.

Read-only replay of the production `signal_events` (2026-04-22 →
2026-09-04, 922 directional Combined rows on the CLUSTER path, each
re-grouped with its three strategy rows and re-scored with the detector's
own rules):

| agreeing votes ≥ 0.35 | Combined signals | share | executed BUYs (168 attributable) |
|---|---|---|---|
| 1 | 746 | 80.9 % | 138 |
| 2 | 169 | 18.3 % | 29 |
| 3 | 7 | 0.8 % | 1 |

The "~78 % of trades were a single vote" finding from the research era
holds: 138 / 168 = 82 % of the executed BUYs (171 in `trade_history`,
3 from the first paper day 2026-05-04 have no cluster row) sat on one
directional vote. Seven three-strategy clusters in ~19 weeks.

## 2. Are the strategies independent?

Source check (`strategies/momentum.py`, `pullback.py`,
`news_catalyst.py`, `coordinator._gather_strategy_votes`,
`_build_news_data`):

| input | Momentum | Pullback | NewsCatalyst |
|---|---|---|---|
| sentiment feed (`sentiment_signal` / `news_score = (avg+1)/2`) | condition 4 | – | conditions 1 + 3 |
| volume vs 20-day average | condition 3 (`vol_ratio > 1.3`, rolling(20) incl. today) | – | condition 2 (`rvol > 1.3`, prior 20 bars) |
| price structure | SMA20/50 trend, RSI 50-65 | SMA50 distance, RSI dip, stochastic | `|pct_change| > 0.5` |

* **Momentum ↔ NewsCatalyst are coupled.** NewsCatalyst's condition 1
  (`news_score ≥ 0.70` ⇔ `avg ≥ 0.40`) implies the sentiment signal is
  BUY (`BUY_THRESHOLD = 0.3`), i.e. Momentum condition 4. NewsCatalyst's
  condition 2 is Momentum's condition 3 on an almost identical series
  (median |Δ| of the two logged volume ratios: 0.008; both > 1.3 in
  288 of the 291 / 297 cases). A full NewsCatalyst BUY therefore hands
  Momentum two of the three conditions it needs.
* Empirically (3 457 runs with all three votes): Momentum votes
  directional in **26.2 %** of the runs where NewsCatalyst does, vs
  **4.3 %** where it does not. Joint firing 143× vs 42 expected under
  independence — lift **3.4**. Pullback shows no lift against either
  (0.84 / 0.61).
* Consequence for the gate: of the 176 signals that clear "≥ 2
  strategies", **136 are exactly the Momentum + NewsCatalyst pair**
  (77 %). Pullback + NewsCatalyst 21, Momentum + Pullback 12, all three
  7. Folding Momentum and NewsCatalyst into one vote source leaves
  40 signals (4.3 %) and **6** of the 168 executed BUYs.

The pair is **not folded** in this change. It is counted, but never
silently: every Combined row records its voters (`cluster_voters`), the
knob exists in `config/settings.py` (`CLUSTER_VOTE_SOURCES`), and the
decision is called out in the delivery note. Folding is a rule change
on top of this one and is Lenni's call.

## 3. What changed

`config/settings.py` — single place for the rule:

```python
CLUSTER_MIN_AGREEING_STRATEGIES: int = 2        # 1 = pre-gate behaviour
CLUSTER_VOTE_SOURCES = {"Momentum": "Momentum", "Pullback": "Pullback",
                        "NewsCatalyst": "NewsCatalyst"}   # same value = one source
```

`orchestrator/cluster_detector.py`

* After the existing filter / bucketing / CONFLICTING checks, the
  agreeing side is reduced to *distinct sources* via
  `CLUSTER_VOTE_SOURCES`. Fewer than `MIN_AGREEING_STRATEGIES` →
  `cluster_signal = "HOLD"`, confidence by the existing HOLD convention
  (`max(HOLD votes) × 0.8`, else 0.25), `gate_status =
  "rejected_min_agreement"`, `agreeing_strategies` / `cluster_strength`
  / `vote_direction` keep the rejected vote visible,
  `strongest_supplier` / `boost_applied` stay None (no verdict, no
  supplier).
* Every exit now carries `gate_status` ∈ {`no_votes`, `no_directional`,
  `conflicting`, `rejected_min_agreement`, `passed`}.
* Order is unchanged: CONFLICTING is decided before the gate, so
  opposing confident votes never show up as a rejected solo.
* With the threshold at 1 every decision is byte-identical to before
  (pinned in `tests/test_signal_attribution.py::test_fusion_decisions_pinned`).

`orchestrator/coordinator.py`

* `_fuse_signals` puts `gate_status / gate_votes / gate_direction /
  gate_voters` into `vote_ctx` (CONFLICTING keeps `vote_ctx = None`,
  A4 unchanged) and logs the journal line
  `Cluster-gate: rejected ticker=… direction=… votes=… sources=… min=… voters=…`.
* `_cluster_gate_fields` maps the fusion exit to the `cluster_*`
  payload; `run_combined` and `analyse_ticker_async` put it on
  `final_result["cluster_gate"]`, `_log_signal_event` writes it.
* Downstream needs nothing: a HOLD verdict is skipped by the RiskAgent,
  not stored as a forward signal (`combined_signal not in ("HOLD",
  "CONFLICTING")`), and the cached US_OPEN executor reads the HOLD back
  from `signal_events` and does not trade.

`analytics/signal_logger.py` — four idempotent-ALTER columns on
`signal_events`, written on every Combined row that went through the
detector, NULL on strategy rows and on the paths that never reach it
(PEAD override, FUSION_FALLBACK):

| column | content |
|---|---|
| `cluster_gate` | `passed` / `rejected_min_agreement` / `no_directional` / `conflicting` / `no_votes` |
| `cluster_votes` | agreeing directional votes ≥ MIN_CONFIDENCE (raw, before source folding) |
| `cluster_direction` | `BUY` / `SELL` of those votes |
| `cluster_voters` | strategies behind them, comma-joined |

The three strategy rows per run are logged exactly as before, so the
data stream does not get thinner — a rejected run is one HOLD row that
says *why* plus its three votes.

## 4. Expected effect

Replay (same population as §1, `scripts/cluster_gate_backtest.py`):

|  | before | ≥ 2 strategies | ≥ 2 sources, Momentum+NC folded |
|---|---|---|---|
| directional Combined signals (CLUSTER path) | 922 | 176 (−81 %) | 40 (−96 %) |
| executed BUYs, 18 paper weeks | 168 | 30 (−82 %) | 6 (−96 %) |
| executed BUYs per week (171 / 18) | 9.5 | **1.7** | 0.3 |
| weeks with zero BUYs | 0 / 18 | 5 / 18 | 14 / 18 |

"Roughly three quarters" is slightly optimistic: the gate removes about
four fifths. Realistic residual: **1–2 BUYs per week**, in bursts (0 to
4 per week), almost all of them Momentum + NewsCatalyst pairs.

The gate is symmetric — SELL-side solo votes are rejected the same way
(15 WEAK SELL Combined rows in the population) — but exits through the
PositionManager (stops, trailing, TP) are not signals and are untouched.

## 5. Post-deploy checks

```bash
# rejected runs, with their votes
journalctl --user -u nts-trading --since today | grep "Cluster-gate: rejected"

# the same from the DB, read-only
python3 - <<'EOF'
import sqlite3
c = sqlite3.connect("file:/home/trading/trading-data/news_trading.db?mode=ro", uri=True)
for r in c.execute("""SELECT cluster_gate, cluster_votes, cluster_direction, count(*)
                      FROM signal_events WHERE strategy='Combined'
                       AND timestamp >= date('now','-7 day')
                      GROUP BY 1,2,3 ORDER BY 1,2"""):
    print(r)
EOF

# replay on the live DB (read-only URI, no writes)
python3 scripts/cluster_gate_backtest.py
```

Expected on the first full day: `Cluster-gate: rejected` lines for most
directional tickers, `cluster_gate='passed'` only where two strategies
agree, no BUY on a `cluster_votes = 1` row.

## 6. Tests

`tests/test_cluster_detector.py` (gate semantics, threshold, source
folding, CONFLICTING precedence), `tests/test_cluster_gate.py`
(coordinator wiring, journal line, `signal_events` columns and
migration, `run_combined` end-to-end: solo → no trade + reasoned row,
pair → trade + `passed`), `tests/test_signal_attribution.py` (pinned
decisions updated, solo-vote attribution kept via threshold 1).
