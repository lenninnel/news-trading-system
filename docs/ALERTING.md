# Telegram Alerting — what reports, what deliberately doesn't, how the watchdog behaves

Goal: the system runs unattended; you look only when Telegram beeps.
Two layers:

1. **In-process alerts** (daemon, PositionManager, ingest script) report
   *events*. They all go through `notifications/telegram_bot.py`.
2. **The watchdog** (`scripts/watchdog.py`, systemd user timer) reports the
   *absence* of expected events and the failure of the units themselves.
   It does not depend on the daemon and cannot fail silently (see below).

## 1. Formatting contract — why there are no more Markdown 400s

`TelegramNotifier._send` is the single choke point. Every message goes out as
Telegram **HTML**:

* Legacy Markdown-styled text (all 13 direct `_send` callers in
  `scheduler/daily_runner.py`, the reviewer text, EOD summary) is converted by
  `legacy_markdown_to_html`: `*bold*` → `<b>`, `` `code` `` → `<code>`, only
  for non-empty spans closed on the same line. Everything else — underscores,
  stray `*`/backticks, `<`, `>`, `&` inside interpolated error strings — is
  escaped and shown verbatim. The converter only emits balanced, non-nested
  tags, so Telegram's parser accepts it for **any** input.
* The typed methods (`send_message`, `send_error`, `send_price_alert`,
  `send_trade_executed`, `send_signal`, `send_daily_summary`) build HTML and
  escape their dynamic fields.
* Messages over 4096 characters are sent plain and truncated.
* The plain-text retry stays as a net; if it ever fires, the
  `Telegram API error` warning in the journal is the signal.

`TelegramNotifier.from_env()` builds the notifier from `TELEGRAM_BOT_TOKEN` /
`TELEGRAM_CHAT_ID` regardless of `telegram.enabled` in `watchlist.yaml` (which
ships `false`). The kill switch, the drawdown CLI and the health monitor now
use it — before, they called a non-existent `send_message` on a notifier that
`from_config` never returned, i.e. the kill-switch alert was dead.

## 2. Event matrix

| Class | Reports? | Where | Notes |
|---|---|---|---|
| Daemon started | yes | daemon | "🟢 News Trading Daemon started" |
| Daemon dead / unit failed | **yes** | watchdog (`daemon`) + `nts-alert@` drop-in | watchdog: within 15 min, reminder every 6 h, recovery line. nts-alert@: when systemd gives up restarting |
| Daemon restart loop | **yes** | watchdog (`restarts` event) | NRestarts delta since last check, reported every time |
| Scheduled session missing | **yes** | watchdog (`session:<date>:<NAME>`) | session time + 20 min grace without a `session_runs` row; per-day keys expire silently at midnight |
| Session started / completed | yes | daemon | not for MIDDAY (monitor) |
| Session crashed | yes | daemon | "🚨 Scheduler error in …" |
| Scheduler loop crashed | yes | daemon | restarts in 60 s |
| Duplicate session skipped | yes | daemon | not for XETRA sessions |
| OHLC ingest failed (gate, fetch, key) | yes | `scripts/ingest_ohlc.py` (own alert) + `nts-alert@` (journal tail) | two messages on a real failure day: script's summary + unit failure with journal |
| OHLC ingest **did not run / no new data** | **yes** | watchdog (`ohlc`) | `MAX(date)` in `daily_ohlc` must be the last US trading day once it is ≥ 23:00 UTC; uses `data/market_calendar.py` (Labor Day etc.) |
| Ingest / backup timer disabled | **yes** | watchdog (`timer:*`) | |
| Backup missing (> 26 h) | **yes** | watchdog (`backup`) + `nts-alert@` on unit failure | |
| IB Gateway unreachable | **yes** | watchdog (`gateway`) | TCP to `IBKR_HOST:IBKR_PORT`, after 2 consecutive misses (tolerates the nightly Gateway restart) |
| IBKR connection lost in the daemon | **yes** | PositionManager | market hours only: after 3 consecutive failed reconnect cycles, once per outage, plus "✅ restored" |
| IBKR reconnect failed before a session | **yes** | daemon | "🚨 IBKR reconnect failed before US_OPEN" |
| PositionManager failed to start | yes | daemon | |
| Entry order rejected / cancelled / timed out | **yes** | daemon run summary | "🚨 Orders NOT filled (n)" with the broker reason; pre-broker skips (non-US, unsupported) stay quiet |
| Exit (stop/TP) order rejected by broker | **yes** | PositionManager | once per ticker until a SELL fills |
| Exit order stuck (PreSubmitted > 5 min) | yes | PositionManager | |
| Stop-loss / take-profit / trailing hit | yes | PositionManager | |
| Stale price feed during RTH | yes | PositionManager | every stale poll (unchanged, by decision) |
| Kill switch activated / resumed | **yes** (fixed) | `emergency_stop.py` | was dead: `send_message` missing + yaml `enabled: false` |
| Kill switch still active | yes | watchdog status block | info line, daily |
| Drawdown halt triggered | yes | daemon | one-shot; watchdog status block shows "HALTED" daily |
| Drawdown unlock (CLI) | **yes** (fixed) | `execution/drawdown_halt.py` | |
| Disk < 1 GB on DB volume | **yes** | watchdog (`disk`) | |
| DB missing / unreadable | **yes** | watchdog (`db`) | dependent checks are skipped, not faked |
| Watchdog itself broken | **yes** | `nts-alert@` via `OnFailure` + daily status block | exit 2 when a due message could not be delivered; exit 1 on a traceback |
| Ghost cleanup at startup | yes | daemon | |
| Weekly sector correlation | yes | daemon | |
| Pre-market scanner result | yes | daemon | |

### Deliberately NOT reported

* MIDDAY session start/summary (existing design: the monitor session is too noisy). A MIDDAY crash still reports via "Scheduler error", and a missed MIDDAY via the watchdog.
* XETRA duplicate-run skips (deploy overlap noise).
* A single failed IBKR reconnect cycle, a single Gateway port miss
  (Gateway restarts nightly), a single missing session before its grace.
* Exit orders that *time out* (not rejected): the PositionManager retries
  next cycle; only broker-side cancellation is alerted.
* Watchdog runs where nothing changed: no message at all. Silence between
  the daily status blocks means "nothing is wrong".
* Weekend session expectations, and the Sunday weekly job.
* Health-monitor daemon (`monitoring/health_monitor.py --daemon`) is not
  deployed; the watchdog covers disk/DB. Its `--notify` path is fixed anyway.

## 3. Watchdog behaviour

Runs every 15 minutes (`nts-watchdog.timer`). Each run:

1. Evaluates all checks (stdlib only: `systemctl --user show`, TCP connect,
   SQLite read-only, file mtimes, `shutil.disk_usage`).
2. Diffs against `watchdog_state.json` (next to the DB):
   * check flips to failing → **NEW** line;
   * still failing and last alert ≥ 6 h ago → **STILL** line;
   * failing → ok → **OK … recovered after N h** line;
   * events (auto-restarts) → **EVENT** line every time.
3. Once per day at/after 06:00 UTC adds the **🫀 status block** (every check
   with ✓/✗, kill switch, drawdown halt, open positions).
4. Sends everything as **one plain-text message** (no parse mode → no 400).

What you see in the failure case, e.g. daemon dead at 10:42 UTC:

```
🚨 NTS watchdog — claw — 2026-09-01 10:45 UTC
NEW  daemon: nts-trading.service ActiveState=failed SubState=failed Result=exit-code exit=1 NRestarts=5
```
then, if unattended, a `STILL daemon (since 2026-09-01 10:45): …` line at
16:45, 22:45, …, and `✅ … OK   daemon recovered after 6.3 h` once fixed. If
systemd's own restart limit was hit, `nts-alert@` additionally posts
`❌ nts-trading.service FAILED on claw …` with the last 15 journal lines.

**How you know the watchdog itself is alive:** the 🫀 status block arrives
every morning at 06:00 UTC (08:00 Berlin summer). If it does not, one of
these is broken — watchdog script, its timer, `.env` credentials, Telegram,
or the host — and that absence is the one thing this design cannot page
you for. Check `systemctl --user list-timers` and
`journalctl --user -u nts-watchdog`. A delivery failure exits 2, which
triggers `nts-alert@` (a second, independent send attempt), and the state
file is rolled back so the next run retries the same alert.

Tunables (env, all optional): `NTS_WATCHDOG_DB` (default `DB_PATH` →
`/home/trading/trading-data/news_trading.db`), `NTS_WATCHDOG_STATE`,
`NTS_WATCHDOG_SESSION_GRACE_MIN=20`, `NTS_WATCHDOG_REALERT_HOURS=6`,
`NTS_WATCHDOG_HEARTBEAT_UTC_HOUR=6`, `NTS_WATCHDOG_BACKUP_MAX_AGE_H=26`,
`NTS_WATCHDOG_DISK_MIN_GB=1`, `NTS_BACKUP_DST_DIR`, `IBKR_HOST`, `IBKR_PORT`,
`ENABLE_PRE_SESSIONS` (XETRA_PRE expectation follows the daemon's flag).

## 4. Install on the VPS (as `trading`, after `git pull` of the merged main)

```bash
cd ~/news-trading-system

# units + drop-in
cp deployment/systemd/nts-alert@.service deployment/systemd/nts-watchdog.service deployment/systemd/nts-watchdog.timer ~/.config/systemd/user/
cp deployment/systemd/nts-ohlc-ingest.service deployment/systemd/nts-backup.service deployment/systemd/nts-preann-estimates.service ~/.config/systemd/user/
mkdir -p ~/.config/systemd/user/nts-trading.service.d
cp deployment/systemd/nts-trading.service.d/10-alerting.conf ~/.config/systemd/user/nts-trading.service.d/
systemctl --user daemon-reload

# daemon picks up the Telegram / PositionManager changes
systemctl --user restart nts-trading

# watchdog
systemctl --user enable --now nts-watchdog.timer

# verify — preview (sends nothing), then a real status block (one message)
set -a; . ./.env; set +a
/usr/bin/python3 scripts/watchdog.py --dry-run --heartbeat
systemctl --user start nts-watchdog.service && journalctl --user -u nts-watchdog -n 20 --no-pager

# verify the OnFailure chain end-to-end (one "❌ selftest FAILED" message)
systemctl --user start nts-alert@selftest.service
systemctl --user list-timers --no-pager | grep nts-
```

Expected after install: "🟢 News Trading Daemon started" from the restart,
one 🫀 status block from the manual `--heartbeat` run (all ✓ except possibly
`session …` lines for sessions earlier today, which expire at midnight),
and the ❌ selftest message. The first automatic 🫀 arrives the next morning
at 06:00 UTC.
