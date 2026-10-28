# KW44 corrections, 2026-10-28

Follow-up to the read-only KW44 review (server report 2026-10-28 06:48 UTC,
`~/Projekte/NTS/reviews/2026-10-kw44/kw44_report.txt` on the Mac). Option B of
the review: repair first, then measure. No strategy thresholds, no risk
parameters, no gate logic changed.

## 1. News input hygiene (changes NewsCatalyst's input)

Finding: NewsCatalyst co-voted in 29 of 35 gate-passed Combined signals
(2026-09-23 → 10-27), but its headlines were on average 29–77 h old at
decision time (max 163 h). The same CVX headline was scored 23 times between
10-12 and 10-27; "cvx-linalg 1.1.2" (a Python package) counted 12 times as
Chevron news; JPM drew stablecoin/GenScript/Indian-bank stories. Cause:
`q=<ticker>` over all fields, no date window, and the 24 h cache re-served
the same batch.

Change (`data/news_feed.py`, `config/settings.py`):

- `NEWS_MAX_AGE_HOURS` (default 72): sent as NewsAPI `from`, and every
  article older than that is dropped on every read (live and cache), so a
  cached batch ages out instead of being re-served.
- `NEWS_RELEVANCE_FILTER` (default true): query `"Chevron" OR CVX` with
  `searchIn=title`, `pageSize` 3× `MAX_HEADLINES`; headline kept only if it
  names the company (`NEWS_COMPANY_ALIASES`) or carries the ticker as a
  case-sensitive standalone token. In-batch duplicates collapsed.
- Articles without a timestamp are kept (age unknown stays visible via
  `news_ts_missing`).

Rollback without code: `NEWS_RELEVANCE_FILTER=false` and
`NEWS_MAX_AGE_HOURS=0` in the service environment, restart `nts-trading`.

Expected effect: fewer NewsCatalyst WEAK BUYs on stale news → fewer
two-vote clusters → probably fewer trades at first. That is the point: the
next 8 weeks measure the system as designed.

## 2. Executed-signal stamp

Finding: 0 of 35 gate-passed Combined rows had `trade_executed=1`, while 7
buys happened. The US_OPEN fast path trades on the cached US_PRE row and
never wrote back.

Change: `Database.mark_signal_executed(id, trade_id)`; called right after a
successful `track_trade` in `_execute_forward_signals_async` (inside the
existing `db_lock`). Non-fatal on failure.

## 3. NTS access (API + MCP)

- `/api/signals?strategy=` server-side filter (before LIMIT). The MCP sends
  it; the client-side filter stays as a fallback for an older API.
- `/api/trades?days=` optional window.
- MCP tool `get_trades(days, limit)`: rows, realized P&L, closed-in-profit
  count.
- `requirements.txt`: `mcp<2` — mcp 2.x renamed FastMCP; an unpinned fresh
  install breaks `nts_mcp.py`.

## 4. Portfolio view: broker accruals

Finding: 2026-10-27, zero positions, view showed −$382.08 daily P&L =
NetLiquidation − TotalCashValue (accrued interest), not a loss.

Change: NAV = cash + marked positions + accruals, accruals =
NetLiquidation − TotalCashValue − GrossPositionValue of the same snapshot
(0 if a tag is missing); exposed as `accruals`.

## Not changed (decisions for 2026-11-02)

- `PEAD_ENABLED=false` on the VPS (env, since ≥ 2026-09-01). Re-enabling is a
  decision, not a fix. Earnings source: Finnhub delivered all 8 realized
  Q3 reports same day; yfinance had wrong actuals (META, BAC).
- PositionManager prices exits from yfinance 1-minute bars only; empty bars
  at 13:30 UTC and a yfinance hiccup on 10-12 skip exit evaluation for a
  poll. Known (stops are software-only); fallback to Alpaca would be the fix.

## Tests

New: `test_news_hygiene.py`, `test_signal_executed_stamp.py`,
`test_kw44_access.py`, one case in `test_portfolio_view.py`. Adjusted:
`test_news_age.py` (age window off for a fixed-date plumbing test),
`test_recovery.py` (cached fixture headlines name the company).
Full suite: 1915 passed; 2 failures pre-exist on `main` (NewsAggregator
fallback tests, unrelated path).

## Deploy (Lenni)

1. Merge `fix/kw44-corrections` → `dev` → `main`.
2. VPS: `git pull`, then `systemctl --user restart nts-trading nts-api nts-mcp`.
3. Mac (local MCP in HTTP mode): `git pull` in `~/Projekte/NTS/news-trading-system`,
   restart the Claude desktop app → `get_trades` appears.
4. First check after the next US_PRE: journal lines
   `NewsAPI filter <TICKER>: kept …` for each ticker.
