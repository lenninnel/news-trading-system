# Data integrity in the execution path — 2026-09-03

Three logging / integrity changes on the execution path. No change to
signal selection (ClusterDetector) or to level logic; the F2 gate keeps
its rules so the Wilder-ATR effect (4a4bbe7, live 2026-09-01) stays
measurable.

1. Fill reconciliation — no execution at the broker without a DB row.
2. News age — the publication time of every processed headline is kept
   and persisted through to the signal.
3. Executed R:R spread — diagnosed, not fixed (finding below).

---

## 1. Fill reconciliation (`execution/ibkr_trader.py`)

### What was wrong

`track_trade` polled `orderStatus.status` and only on `"Filled"` wrote
`trade_history` + `portfolio_positions`. Any other outcome returned a
"skipped" dict and wrote nothing — including when part of the order had
already executed. Real sequence (TXRH, 2026-08-10, journal):

```
13:45:32  placeOrder  BUY 132                       PendingSubmit
13:45:34  orderStatus Submitted  filled=100 rem=32  avg=207.35
13:46:03  cancelOrder (ORDER_FILL_TIMEOUT=30s)      PendingCancel filled=100
13:46:03  orderStatus Cancelled  filled=100 rem=32
→ 100 shares at IBKR, 0 in the DB, no SL/TP, orphan SELL later
```

Eight such cases (UFPI ×5, CACI ×2, TXRH ×1) between June and August.
The same exposure existed on every non-filled outcome (`cancelled`,
`stuck`) and on `close_position` (which recorded nothing even on a
complete fill) and `place_order`.

### Behaviour now

All order entry points (`track_trade`, `close_position`, `place_order`)
run through one `_place_and_settle` → `_record_fill` path.

| outcome after waiting | broker action | what lands in the DB | return dict |
|---|---|---|---|
| `filled` | — | one row `fill_status='filled'` | as before |
| `timeout` (BUY, 30 s), fill beat the cancel | cancel requested, order reports `Filled` inside `CANCEL_SETTLE_WAIT` (5 s) | one row `filled` | as before |
| `timeout`, partial fill | cancel remainder, wait ≤5 s for `Cancelled` | row for the filled shares, `fill_status='partial'`, `requested_shares`=sent, `broker_order_id`; position synced with the filled shares | `partial=True`, `outcome='timeout'`, `shares`=filled, `unfilled_shares`, `skipped=False`, `trade_id` |
| `timeout`, nothing filled | cancel | nothing | `skipped=True` as before (+ `filled_shares=0`, `broker_order_id`) |
| `timeout`, cancel still pending after 5 s | — | `partial` row now; late-fill watcher armed | `partial=True` |
| `cancelled` by the broker, partial | — | `partial` row | `partial=True`, `outcome='cancelled'` |
| `stuck` (SELL > 300 s, never cancelled) | — | `partial` row if anything filled; late-fill watcher armed | `skipped=True` (0 filled) or `partial=True` |
| late fill arrives on a watched order | — | row `fill_status='late'` for the newly filled shares, price = marginal price from the cumulative averages, SL/TP/strategy carried over; position synced | logged `IBKR LATE-FILL WATCH closed` |

The late-fill watcher hooks `trade.statusEvent` (ib_insync emits it on
the IB loop thread, i.e. the same thread that writes synchronous fills).
If a trade object has no event to hook, that is logged at ERROR
(`LATE-FILL WATCH unavailable`) — the gap is visible, never silent.

Consumers:

* **Coordinator** — checks `execution.get("trade_id")`; a partial BUY
  now registers position metadata like any other fill (it *is* a
  position). Forward signals are marked executed.
* **PositionManager** — `_handle_close_result` sees `partial=True` before
  the filled check: Telegram `⚠️ PARTIAL EXIT … filled n/m`, event
  logged, trailing stop **kept** (position still open). `outcome='stuck'`
  additionally sets the stuck cooldown so the next cycle does not put a
  second SELL on top of the in-flight remainder; `outcome='cancelled'` /
  `'timeout'` lets the next cycle retry the remainder (PM reads live
  IBKR positions, so it sees the remaining shares).
* **Dispatch timeout** (`_run_in_ib_loop` raising `TimeoutError`) is
  unchanged: the sync `_impl` cannot be interrupted once running, so its
  DB writes still complete on the loop thread; the caller merely does not
  learn the result.

### Remaining gaps (not in scope, stated)

* A daemon crash / SIGTERM *during* the 30–300 s wait loses the in-memory
  watcher. There is still no periodic broker-vs-DB position
  reconciliation; `broker_order_id` is now stored so one can be built
  against `ib.fills()` / `ib.executions()`.
* `PaperTrader` is unaffected (simulated fills are always complete).

### Post-deploy check

```sql
-- partial / late legs since deploy
SELECT id, created_at, ticker, action, shares, requested_shares, price,
       fill_status, broker_order_id
FROM trade_history
WHERE fill_status IN ('partial', 'late')
ORDER BY id;

-- net shares per ticker must equal the IBKR position
SELECT ticker, SUM(CASE action WHEN 'BUY' THEN shares ELSE -shares END) AS net
FROM trade_history GROUP BY ticker HAVING net <> 0;
```

Journal markers: `IBKR ORDER PARTIAL`, `IBKR ORDER LATE`,
`IBKR LATE-FILL WATCH armed|closed|unavailable`.

---

## 2. News age

### What is kept

| stage | field | source |
|---|---|---|
| `NewsFeed.fetch_articles()` | `published_at` (ISO-8601 UTC) + `fetched_at` | NewsAPI `publishedAt` |
| `MarketauxFeed` items | `published_at` | Marketaux `published_at`; EODHD `date` on enrichment rows |
| `StockTwitsFeed` / `RedditFeed` items | `published_at` | `created_at` / `created_utc` |
| ApeWisdom / Adanos items | `published_at = None` | aggregates, no per-item time |
| coordinator `items` → `scored` | `published_at` per headline | carried through the SentimentAgent result |
| `headline_scores.published_at` | per scored headline | `log_headline_score(..., published_at=)` |
| `signal_events.news_newest_published_at` / `news_age_minutes` / `news_ts_missing` | per signal row (NewsCatalyst strategy row **and** the Combined row) | `Coordinator._news_timing(scored)` at decision time (`signal_events.timestamp`) |

`NULL` means the provider supplied no timestamp or it was unparseable
(`utils/timeparse.parse_utc` never invents a value; naive inputs are
taken as UTC because every provider here documents UTC). NewsAPI logs
one WARNING per response listing how many articles had no usable
`publishedAt`. Legacy cache entries (title lists written before the
deploy) surface as `published_at = NULL` until the first live fetch
after the deploy replaces them (24 h cache).

`NewsFeed.fetch()` still returns titles for older callers; only
`fetch_articles()` carries the time. `NewsCatalystStrategy` receives the
three timing fields in `news_data` and copies them into `indicators` —
no condition reads them.

### From when it can be evaluated

From the first session after the VPS pull + restart. Rows before that
have `NULL` in all new columns. Evaluation:

```sql
-- age of the newest headline behind each NewsCatalyst / Combined signal
SELECT timestamp, session, ticker, strategy, signal, confidence,
       news_age_minutes, news_newest_published_at, news_ts_missing
FROM signal_events
WHERE timestamp >= '2026-09-04' AND strategy IN ('NewsCatalyst', 'Combined')
ORDER BY timestamp;

-- per-headline age at run time
SELECT r.created_at AS decided_at, r.ticker, h.source, h.published_at,
       ROUND((julianday(r.created_at) - julianday(h.published_at)) * 1440, 1) AS age_min,
       h.sentiment, h.headline
FROM headline_scores h JOIN runs r ON r.id = h.run_id
WHERE r.created_at >= '2026-09-04'
ORDER BY r.created_at, age_min;
```

---

## 3. Executed R:R spread — diagnosis (no fix in this deploy)

### Claim under test

After the strategy-override removal every RiskAgent calc is exactly
2.00 R:R (`risk_calculations`, 25/25 on 2026-09-01/02). The five
executed BUYs show 1.30 / 1.86 / 1.91 / 2.12 / 2.24. Suspected cause:
levels set against a reference price, R:R measured against the fill.

### Data (prod DB, read-only; `risk_calculations`, `forward_signals`, `trade_history`)

Each ticker has two RiskAgent runs: US_PRE (13:1x UTC, price `P_pre`,
stored on the `forward_signals` row) and US_OPEN (14:30 UTC, price
`P_open` = `intended_price`, the run whose levels are the fresh
reference).

| | MSFT 09-01 | XOM 09-02 | CVX 09-02 | BAC 09-02 | TSLA 09-02 |
|---|---|---|---|---|---|
| P_pre (forward row) | 501.60 | 164.52 | 211.02 | 62.01 | 356.08 |
| forward SL / TP | 483.5215 / 537.757 | 159.2886 / 174.9828 | 204.948 / 223.1641 | 60.4357 / 65.1585 | 335.4449 / 397.3502 |
| P_open = intended | 504.21 | 163.10 | 211.60 | 63.095 | 355.14 |
| fresh SL / TP (14:30) | 486.1315 / 540.367 | 157.8686 / 173.5628 | 205.528 / 223.7441 | 61.5207 / 66.2435 | 334.5049 / 396.4102 |
| fresh R:R | 2.000 | 2.000 | 2.000 | 2.000 | 2.000 |
| fill (`executed_price`) | 504.21 | 163.15 | 211.59 | 63.10 | 355.28 |
| slippage fill − P_open | 0.000 | +0.050 | −0.010 | +0.005 | +0.140 |
| **SL on the trade** | 486.1315 (fresh) | 157.8686 (fresh) | 205.528 (fresh) | 61.5207 (fresh) | 335.4449 (**forward**) |
| **TP on the trade** | 537.757 (**forward**) | 174.9828 (**forward**) | 223.1641 (**forward**) | 65.1585 (**forward**) | 397.3502 (**forward**) |
| R:R vs fill | **1.856** | **2.240** | **1.909** | **1.303** | **2.121** |
| R:R vs P_open (no slippage) | 1.856 | 2.271 | 1.904 | 1.311 | 2.143 |
| F2 sl-leg deviation (tol 0.05) | 0.144 → reject | 0.271 → reject | 0.096 → reject | 0.689 → reject | 0.046 → **adopt** |
| P_open − P_pre (drift) | +2.61 | −1.42 | +0.58 | +1.085 | −0.94 |

Journal 2026-09-02 14:30 confirms the sl-leg rejections verbatim
(`F2-gate: rejected … leg=sl reason=model_mismatch … XOM deviation=0.271438`,
JPM 0.613959); tp-leg adoptions were logged at DEBUG only and are
therefore absent from the journal.

### Mechanism

In the cached US_OPEN executor (`coordinator.py` step 5) the fresh 14:30
calc is followed by `apply_level_override` for both forward legs. The
gate is asymmetric by spec (`level_gate.py`: "model_mismatch (sl leg
ONLY — TP is out of scope this window)"):

* **sl leg**: adopt only if `candidate < fill` **and**
  `|fresh_SL − forward_SL| / (fill − fresh_SL) ≤ 0.05`. Any pre→open
  drift larger than 5 % of the stop distance rejects → fresh SL kept
  (4 of 5). TSLA's drift was 4.6 % → forward SL adopted.
* **tp leg**: adopt if `candidate > fill`. Always true for a long whose
  price did not run past the pre-market TP → forward TP adopted (5 of 5).

So the executed pair is *fresh SL on P_open* + *forward TP on P_pre*.
With SL = P_open − 1.5·ATR and TP = P_pre + 3·ATR:

```
R:R = (P_pre + 3·ATR − P_open) / (1.5·ATR) = 2 − drift / (1.5·ATR)
```

MSFT: 2 − 2.61/18.08 = 1.856 ✓ · XOM: 2 + 1.42/5.23 = 2.271 ✓ ·
CVX: 2 − 0.58/6.07 = 1.904 ✓ · BAC: 2 − 1.085/1.574 = **1.311** ✓
(BAC's stop distance is only $1.57, so a $1.09 gap-up eats 69 % of it).
TSLA (both legs forward, set on P_pre): (397.35 − 355.28)/(355.28 − 335.44)
= 2.121 ✓. The residual between "vs P_open" and "vs fill" columns is the
true fill slippage: at most 0.03 R:R (XOM, TSLA).

### Verdict

**Not slippage.** Fill slippage explains ≤ 0.03 of the spread. The spread
is the pre-market → open price drift entering the TP leg only, because
the F2 gate validates forward SLs against the fresh model and forward
TPs only against the side of the fill. The R:R on the trade is therefore
neither the 2:1 of the fresh model nor the 2:1 of the forward model.

Per the brief ("if it is something other than slippage: report, do not
guess") no level logic is changed in this deploy. The only change is the
adoption log level (DEBUG → INFO) so both legs' origin is visible in the
journal from now on.

Recommendation for the follow-up decision (Lenni / R):

* **Symmetric gate** — apply the same 0.05 model-mismatch invariant on
  the tp leg (reference = fresh TP, denominator = fresh TP − fill). Then
  either both legs come forward (small drift, TSLA case, R:R ≈ 2 ± drift
  effect on both legs) or both stay fresh (R:R = 2.00). This is the
  minimal change consistent with the existing spec.
* Alternative: stop feeding forward levels into the cached executor at
  all — since the override removal both forward and fresh levels are the
  same RiskAgent formula, the forward one merely older, so adoption can
  only add staleness. This would, however, remove the pre-registered
  post-deploy watch on the forward model_mismatch rate (R, 2026-08-26).

Re-anchoring levels to the actual fill is a separate question and not
needed for the 2:1 property: the fresh levels are set on the 14:30 quote
and the fills deviate from it by ≤ $0.14.
