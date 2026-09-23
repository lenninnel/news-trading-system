# No US sessions on NYSE holidays — 2026-09-23

## Incident

Labor Day, Monday 2026-09-07: the daemon fired all eight sessions on a
closed market (`session_runs` rows for XETRA_PRE … EOD, 120 `signal_events`
rows from PREMARKET_SCAN / US_PRE / EOD). The scheduler knew weekdays only
(`dt.weekday() < 5`); `data/market_calendar.py` existed but was used by the
ingest gate, the PositionManager and the re-entry lock — not by the schedule.
The new bar-freshness gate would now refuse to trade on that day's stale bar,
but the sessions still run, the rows still land, and the watchdog would have
alerted on nothing.

## Rule (one place: `config/sessions.py`)

Every schedule entry carries a `market`:

| market  | session exists on day D iff |
|---------|-----------------------------|
| `US`    | D is a US trading day (`is_us_trading_day`) **and** the session's UTC time converted to New York is before D's close (`us_rth_close`: 16:00, or 13:00 on an early-close day). Entries flagged `after_close` (EOD) skip the close test. |
| `XETRA` | D is a weekday. No XETRA holiday calendar exists; the XETRA sessions are deliberately **not** coupled to the US calendar (they still fire on Thanksgiving, as empty no-ops today). |

`session_runs_on(entry, day)` → `(bool, reason)`; `sessions_on(day)`,
`next_session_run(after)`, `last_session_run(before)`, `us_calendar_note(day)`
are built on it. Consumers:

* `scheduler/daily_runner.py` — `next_run_time`, `_run_for_time`,
  `current_session` (startup run) and a guard at the top of `_execute_run`
  (`_session_allowed_today`). A skipped session claims nothing and writes
  nothing: no `session_runs` row, no `signal_events`, no Telegram.
* `scripts/watchdog.py` — `_due_sessions` expects only sessions that exist
  today; the status block gets a `US calendar:` info line naming the
  reason and the sessions not expected (holiday → all six US sessions;
  early close → MIDDAY). Fallback when the repo module is missing: weekdays
  only, as before.
* `api/main.py` `/api/status` and `mcp_server/nts_mcp.py` `get_status` —
  `next_session` / `next_run_at` come from `next_session_run`, so on a
  weekend or holiday they name the real next firing (date included) instead
  of "today at 13:00". The MCP server's private schedule copy (still EOD
  22:15) is gone. Both expose `calendar_note`.

## Early closes (new in `data/market_calendar.py`)

`us_early_closes(year)`: Friday after Thanksgiving; July 3 and December 24
when they fall Mon–Thu (on a Friday they are the observed holiday). Verified
against the published NYSE calendars 2020–2027 in
`tests/test_market_calendar.py`. `us_rth_close(day)` / `is_us_rth` end the
regular session at 13:00 on those days; `PositionManager._is_market_hours`
and `PriceMonitor` use it, so the stale-feed escalation stops at the real
close instead of running three hours on a closed market.

Which sessions fall behind an early close: only **MIDDAY** (18:00 UTC =
13:00 EST in November/December, 14:00 EDT on a July 3). It is skipped —
it is a position monitor and there is nothing to monitor after the close;
the PositionManager covers the shortened session, EOD runs at 22:45 as on
any trading day and the daily bar exists.

## What happens now

| Day | Fires | Skipped (reason) | Watchdog |
|-----|-------|------------------|----------|
| Ordinary US trading day | all 8 | — | unchanged |
| NYSE holiday on a weekday (e.g. Labor Day, Thanksgiving, Good Friday) | XETRA_PRE, XETRA_OPEN (empty) | 6 US sessions ("US market holiday") | expects only the XETRA rows; status block: `US calendar: US market holiday — not expected today: …` |
| Early-close day (Fri after Thanksgiving, Dec 24 Mon–Thu, Jul 3 Mon–Thu) | 7 | MIDDAY ("US early close 13:00 ET") | does not expect MIDDAY; still expects EOD after 23:05 |
| Weekend | nothing (Sunday weekly job unchanged) | — | unchanged |

Unscheduled closures (day of mourning) are still not modelled — the
freshness gate and the watchdog OHLC check fire once, as documented in
`data/market_calendar.py`.

## Deploy

Code only, no schema, no unit-file change. Restart after `git pull` on the
VPS: `nts-trading` (scheduler + PositionManager), `nts-api` (status),
`nts-mcp` (status). The watchdog is a oneshot timer and picks the change
up on its next tick.
