# Portfolio view shows the truth — 2026-09-23

## Symptom

`get_portfolio` (MCP) and `/api/portfolio` returned identical numbers on
2026-09-19 and 2026-09-23: "Total value" = position value only (no cash),
`now` == `entry` for days, "Daily P&L" 0.00.

## Root cause (three independent defects on one read path)

1. **The mark was clobbered.** `IBKRTrader.get_portfolio()` runs at every
   session start (pre-session portfolio sync in `_execute_run`) and in every
   PositionManager cycle. IBKR's position list carries no market price, so
   it wrote `current_value = shares × avg_price` — the entry value — on top
   of the PositionManager's live mark. The PositionManager only re-marks
   during RTH (yfinance 1-minute bars, the same price it trails stops on),
   so from the EOD session (22:45 UTC) until the next open the table held
   entry values; the MCP was queried in that window. Known since 2026-05-05
   as the "get_portfolio vs mark_to_market race" backlog item.
2. **"Total value" was positions only** — `Σ current_value`, no cash.
   "Cash" was `risk_calculations.account_balance` (= IBKR NetLiquidation at
   the last session) minus cost basis: that is cash + unrealised P&L, not
   cash, and the two were never added up.
3. **"Daily P&L" was today's realised SELL P&L** (`Σ trade_history.pnl`
   today), so any day without an exit read 0.00 regardless of what the
   positions did.

The IBKR paper account does deliver market prices (`updatePortfolio`
events in the journal: `marketPrice=342.98, marketValue=27781.38` for
AAPL at 13:15 UTC); the old code comment saying it does not is outdated.

## Definitions now (one function: `analytics/portfolio_view.py`)

| Field | Source |
|-------|--------|
| `positions_value` | Σ shares × `mark_price` — the PositionManager's last mark (yfinance 1-min, the price exits are judged on) or the last fill. Never-marked rows fall back to cost and are listed in `notes`. |
| `cash` | `TotalCashValue` from the newest `account_snapshots` row (broker). Before the first snapshot: the old estimate, labelled `estimate:` in `cash_source`. |
| `value` / `nav` | `cash + positions_value` |
| `daily_pnl` | NAV − previous close. Previous close = IBKR `PreviousDayEquityWithLoanValue` from the newest snapshot, else the NetLiquidation of the last `kind='eod'` snapshot before today. Neither → `null` with `daily_pnl_basis` saying why. |
| `realized_today` | Σ `trade_history.pnl` today (kept, separate) |
| `broker_nav`, `nav_minus_broker` | NetLiquidation of the newest snapshot and the difference to the view's NAV, for reconciliation |
| `cash_as_of`, `marks_as_of`, per-position `mark_source` / `marked_at` | provenance |

Legacy keys (`value`, `cash`, `daily_pnl`, `daily_pnl_pct`, `positions[]`)
are kept for the dashboard (`app/portfolio/page.tsx` reads exactly those);
`value` is now NAV.

## Writers

* `portfolio_positions` gains `mark_price`, `mark_source`, `marked_at`
  (additive migration in `Database._init_schema`).
  `Database.sync_portfolio_position` (broker sync) updates shares/avg_price
  and recomputes `current_value = shares × mark_price`, never touching the
  mark; `mark_portfolio_position` (PositionManager, every RTH cycle) sets
  it; fills set it to the fill price (`mark_source='fill'`); Alpaca sync
  sets it from `market_value`.
* New table `account_snapshots` (ts, net_liquidation, total_cash,
  gross_position_value, prev_day_equity, buying_power, source, kind).
  Written by `DailyScheduler._fetch_session_account_balance` at every
  session (`kind='session'`, `'eod'` for EOD) and by
  `PositionManager._snapshot_account` during RTH at most every 5 min
  (`kind='pm'`), after the stop loop so a slow `get_account()` never delays
  an exit, also with zero positions. `IBKRTrader.get_account` now
  reads `PreviousDayEquityWithLoanValue` and `GrossPositionValue` too.
  The session snapshot is taken after the bar-freshness gate, so a
  session the gate aborts writes none; the PositionManager covers RTH
  regardless. Snapshot failures are logged and throttled, never fatal.
* Readers: `/api/portfolio` and MCP `_sql_portfolio` both call
  `build_portfolio_view` (read-only URI connection); MCP HTTP mode gets the
  API's dict. `/api/state` is unchanged and now sees marks in
  `current_value` automatically. The Telegram EOD summary is unchanged
  (uses the configured baseline by design).

Trading logic, thresholds and sizing are untouched: the deployment cap
still reads `Σ current_value`, which is now the live mark instead of
sometimes-entry / sometimes-live.

## Before / after against IBKR NetLiquidation

Prod DB (read-only copy, 2026-09-23) and the daemon journal. At 13:15 UTC
the account held 81 AAPL (avg 333.45); IBKR: NetLiquidation 271,095.71,
AAPL marketPrice 342.98 / marketValue 27,781.38.

| | old view | new view | IBKR |
|---|---|---|---|
| Total value | 27,009.64 (positions at entry) | 271,095.71 (NAV) | 271,095.71 |
| Cash | 244,086.07 (NetLiq − cost) | 243,314.33 (TotalCashValue) | 243,314.33 (implied) |
| Positions | 27,009.64 | 27,781.38 | 27,781.38 |
| AAPL now | 333.45 (= entry) | 342.98 | 342.98 |
| Daily P&L | 0.00 | NAV − prev close (n/a in the reconstruction: no PreviousDayEquity in the journal) | — |

The new numbers are a reconstruction on the copy: the AAPL mark and the
snapshot were inserted by hand from the journal values, because the
daemon that writes them is not deployed yet. `nav_minus_broker` = 0.00
by construction there. After deploy the check is live in the view itself
(`Broker NetLiq … (view − broker: …)` in `get_portfolio`); a non-zero
difference is the mark-source gap (yfinance 1-min vs IBKR marketPrice)
plus anything held outside `portfolio_positions` (FX balances are skipped
by `get_positions`).

Today's prod state after the AAPL exit (no snapshot yet): old view
value 0 / cash 270,554.40 / daily P&L +317.33; new view NAV 270,554.40
labelled `estimate:` with `daily_pnl: null` and the notes saying why —
until the first session after deploy writes a snapshot.

## Deploy

`git pull` on the VPS, then restart `nts-trading` (writers: scheduler +
PositionManager; its `Database()` applies the additive migration on start),
`nts-api` (endpoint), `nts-mcp` (tool). The dashboard needs no change.
First snapshot = first session start after the restart; daily P&L appears
from then on (IBKR's previous-day equity tag), the `eod` reference from
the first EOD.
