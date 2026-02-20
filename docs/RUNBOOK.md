# Operations Runbook — News Trading System

> **Audience**: Operators responsible for keeping the system running in production.
> **Purpose**: Step-by-step procedures for routine operations and emergencies.

---

## Table of Contents

1. [Daily Tasks](#1-daily-tasks)
2. [Weekly Tasks](#2-weekly-tasks)
3. [Monthly Tasks](#3-monthly-tasks)
4. [Emergency Procedures](#4-emergency-procedures)
5. [Health Check Commands](#5-health-check-commands)
6. [Backup and Restore](#6-backup-and-restore)

---

## 1. Daily Tasks

Estimated time: **5–10 minutes** on trading days.

### 1.1 Check health monitor status

```bash
# Option A: HTTP health endpoint (if running)
curl -s http://localhost:9090/health | python3 -m json.tool

# Option B: CLI check
python3 -c "
from utils.api_recovery import APIRecovery
from data.fallback_coordinator import FallbackCoordinator
import json
circuits = APIRecovery.get_status()
fallbacks = FallbackCoordinator.get_status()
degraded_circuits  = {k: v for k, v in circuits.items()  if v['state'] != 'CLOSED'}
degraded_fallbacks = {k: v for k, v in fallbacks.items() if v['level'] > 0}
print('Degraded circuits: ', degraded_circuits  or 'none')
print('Active fallbacks:  ', degraded_fallbacks or 'none')
"

# Option C: Railway dashboard (cloud)
railway logs --service worker --tail 20
```

**What to look for:**
- `"state": "CLOSED"` for all circuits = healthy
- `"level": 0` for all services = primary sources working
- No `[DEGRADED]` or `[FALLBACK ALERT]` lines in logs

### 1.2 Verify scheduler ran

```bash
# Check log file exists for today
LOG_DATE=$(date +%Y-%m-%d)
LOG_FILE="scheduler/logs/${LOG_DATE}.log"

if [ -f "$LOG_FILE" ]; then
  echo "Scheduler ran. Last lines:"
  tail -10 "$LOG_FILE"
else
  echo "WARNING: No log file for today. Run manually:"
  echo "  python3 scheduler/daily_runner.py --now"
fi

# Check DB for today's signals
sqlite3 news_trading.db \
  "SELECT ticker, combined_signal, confidence, created_at
   FROM combined_signals
   WHERE date(created_at) = date('now')
   ORDER BY created_at DESC;"
```

**Expected output:** At least one row per watchlist ticker.

**If scheduler did not run:** See TROUBLESHOOTING.md § 6.

### 1.3 Review yesterday's trades

```bash
# Paper trades from yesterday
sqlite3 news_trading.db \
  "SELECT ticker, action, shares, entry_price, stop_loss, take_profit, trade_date
   FROM trade_history
   WHERE date(trade_date) >= date('now', '-1 day')
   ORDER BY trade_date DESC;"

# Current open positions and unrealised P&L
sqlite3 news_trading.db \
  "SELECT ticker, action, shares, entry_price, current_price,
          round((current_price - entry_price) * shares, 2) AS unrealised_pnl
   FROM portfolio
   WHERE status = 'open'
   ORDER BY unrealised_pnl ASC;"
```

**Red flags:**
- Trades with `CONFLICTING` signal in the same session → review manually
- More than 3 losing positions open simultaneously → consider tightening thresholds
- Any `action = 'STRONG SELL'` with high conviction → verify the news source

### 1.4 Check fallback coordinator alerts

```bash
python3 -c "
from data.fallback_coordinator import FallbackCoordinator
alerts = FallbackCoordinator.check_and_alert()
if alerts:
    for a in alerts:
        print(f\"ALERT: {a['service']} on L{a['level']} ({a['source']}) for {a['hours_elapsed']}h\")
else:
    print('No fallback alerts.')
"
```

**Action on alert:** Investigate the failing primary source (see TROUBLESHOOTING.md § 2 or § 7).

---

## 2. Weekly Tasks

Estimated time: **20–30 minutes** once per week (suggested: Monday morning).

### 2.1 Review performance metrics

```bash
# Win rate and average P&L for the past 7 days
sqlite3 news_trading.db \
  "SELECT
     count(*)                                          AS total_trades,
     round(avg(pnl), 2)                               AS avg_pnl,
     round(100.0 * sum(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / count(*), 1) AS win_rate_pct,
     round(sum(pnl), 2)                               AS total_pnl
   FROM trade_history
   WHERE date(trade_date) >= date('now', '-7 days')
     AND status = 'closed';"

# Breakdown by signal type
sqlite3 news_trading.db \
  "SELECT combined_signal, count(*) AS n,
          round(avg(confidence), 3) AS avg_confidence
   FROM combined_signals
   WHERE date(created_at) >= date('now', '-7 days')
   GROUP BY combined_signal
   ORDER BY n DESC;"
```

**Healthy benchmarks (paper trading):**
- Win rate ≥ 50% on STRONG signals
- Average confidence ≥ 0.60 for executed trades
- No more than 5% of runs in CONFLICTING state

### 2.2 Check API usage trends

```bash
# Recovery events this week
sqlite3 news_trading.db \
  "SELECT service, event_type, count(*) AS n
   FROM recovery_log
   WHERE date(created_at) >= date('now', '-7 days')
   GROUP BY service, event_type
   ORDER BY n DESC;"

# How many times each fallback level was used (news)
sqlite3 news_trading.db \
  "SELECT event_type, count(*) AS n
   FROM recovery_log
   WHERE service = 'newsapi'
     AND date(created_at) >= date('now', '-7 days')
   GROUP BY event_type;"
```

**If Level 1–3 used frequently:** Consider upgrading API plan or reducing `MAX_HEADLINES` to stay within free tier limits.

### 2.3 Review and update watchlist if needed

```bash
# Current watchlist
cat config/watchlist.yaml | grep -A20 'watchlist:'

# Tickers with consistently low signal confidence this week
sqlite3 news_trading.db \
  "SELECT ticker, round(avg(confidence), 3) AS avg_conf, count(*) AS runs
   FROM combined_signals
   WHERE date(created_at) >= date('now', '-7 days')
   GROUP BY ticker
   HAVING avg_conf < 0.4
   ORDER BY avg_conf ASC;"
```

**Candidates for removal:** Tickers with avg confidence < 0.40 and no trades in 14 days (low signal quality, wasting API quota).

To update the watchlist:
```bash
# Edit watchlist.yaml
nano config/watchlist.yaml

# Test the next run without executing trades
python3 scheduler/daily_runner.py --now --no-execute
```

### 2.4 Verify backups are current

```bash
# Check backup age
ls -lh backups/*.db 2>/dev/null | tail -5
# Should show a file from within the last 7 days

# If no recent backup, create one now
mkdir -p backups
cp news_trading.db "backups/news_trading_$(date +%Y%m%d).db"
echo "Backup created: backups/news_trading_$(date +%Y%m%d).db"
```

---

## 3. Monthly Tasks

Estimated time: **60–90 minutes** once per month.

### 3.1 Re-optimise parameters

Review and adjust thresholds in `config/settings.py` based on 30-day performance:

```bash
# 30-day signal distribution
sqlite3 news_trading.db \
  "SELECT combined_signal, count(*) AS n,
          round(avg(confidence), 3) AS avg_conf
   FROM combined_signals
   WHERE date(created_at) >= date('now', '-30 days')
   GROUP BY combined_signal ORDER BY n DESC;"

# Check if BUY_THRESHOLD needs adjustment
sqlite3 news_trading.db \
  "SELECT round(avg_score, 2) AS avg_score, signal, count(*) AS n
   FROM analysis_runs
   WHERE date(created_at) >= date('now', '-30 days')
   GROUP BY signal ORDER BY n DESC;"
```

**Adjustment guidelines:**

| Observation | Action |
|---|---|
| > 60% signals are HOLD | Lower `BUY_THRESHOLD` from 0.30 → 0.25 |
| Too many CONFLICTING signals | Raise `BUY_THRESHOLD` from 0.30 → 0.35 |
| Win rate < 45% on STRONG signals | Raise minimum confidence gate in `risk_agent.py` |
| Position sizes too small | Increase `MAX_RISK_PER_TRADE_PCT` (default 2%) |

### 3.2 Review and tighten risk limits

```bash
# Largest single-trade losses in the past 30 days
sqlite3 news_trading.db \
  "SELECT ticker, action, shares, entry_price, exit_price,
          round((exit_price - entry_price) * shares, 2) AS pnl,
          trade_date
   FROM trade_history
   WHERE date(trade_date) >= date('now', '-30 days')
     AND status = 'closed'
   ORDER BY pnl ASC LIMIT 10;"

# Check stop-loss hit rate
sqlite3 news_trading.db \
  "SELECT
     count(*) AS total,
     sum(CASE WHEN exit_reason = 'stop_loss' THEN 1 ELSE 0 END) AS stopped_out,
     sum(CASE WHEN exit_reason = 'take_profit' THEN 1 ELSE 0 END) AS took_profit
   FROM trade_history
   WHERE date(trade_date) >= date('now', '-30 days')
     AND status = 'closed';"
```

**Red flags requiring immediate action:**
- Stop-loss hit rate > 60% → strategy not working; activate kill switch and review
- Single trade loss > 5% of account → check Kelly fraction cap enforcement

### 3.3 Archive old database rows

Keep the DB lean to avoid memory issues and slow queries:

```bash
# Check current row counts
sqlite3 news_trading.db \
  "SELECT 'analysis_runs',    count(*) FROM analysis_runs    UNION ALL
   SELECT 'headline_scores',  count(*) FROM headline_scores  UNION ALL
   SELECT 'technical_signals',count(*) FROM technical_signals UNION ALL
   SELECT 'combined_signals', count(*) FROM combined_signals  UNION ALL
   SELECT 'recovery_log',     count(*) FROM recovery_log;"

# Archive rows older than 90 days (creates archive file first)
ARCHIVE="backups/archive_$(date +%Y%m).db"
cp news_trading.db "$ARCHIVE"
echo "Archive saved to $ARCHIVE"

# Delete old rows (keep last 90 days)
sqlite3 news_trading.db <<'SQL'
DELETE FROM headline_scores
  WHERE run_id IN (
    SELECT id FROM analysis_runs
    WHERE date(created_at) < date('now', '-90 days')
  );
DELETE FROM analysis_runs    WHERE date(created_at) < date('now', '-90 days');
DELETE FROM technical_signals WHERE date(created_at) < date('now', '-90 days');
DELETE FROM combined_signals  WHERE date(created_at) < date('now', '-90 days');
DELETE FROM recovery_log      WHERE date(created_at) < date('now', '-90 days');
VACUUM;
SQL
echo "Archive complete. New DB size: $(du -sh news_trading.db)"
```

### 3.4 Backup database

```bash
# Full monthly backup with timestamp
BACKUP="backups/monthly_$(date +%Y%m).db"
mkdir -p backups
sqlite3 news_trading.db ".backup $BACKUP"
echo "Monthly backup: $BACKUP ($(du -sh $BACKUP | cut -f1))"

# If on Railway with PostgreSQL, use pg_dump
# pg_dump $DATABASE_URL > backups/monthly_$(date +%Y%m).sql
```

---

## 4. Emergency Procedures

### 4.1 System Down — Recovery Steps

```
SEVERITY: HIGH   TIME TO RESOLVE TARGET: < 30 minutes
```

**Step 1: Assess the situation**
```bash
# What's running?
ps aux | grep -E "(python|streamlit|gunicorn)"

# Any recent errors?
tail -100 scheduler/logs/$(date +%Y-%m-%d).log 2>/dev/null | grep -i error

# DB accessible?
sqlite3 news_trading.db "SELECT count(*) FROM analysis_runs;" 2>&1
```

**Step 2: Activate kill switch to prevent bad state**
```bash
python3 main.py --kill-switch on
```

**Step 3: Restart services**
```bash
# Local
pkill -f daily_runner.py
pkill -f "streamlit run"
sleep 5
python3 scheduler/daily_runner.py --now --no-execute   # test run first
streamlit run dashboard/app.py &

# Railway (cloud)
railway service restart worker
railway service restart dashboard
```

**Step 4: Verify recovery**
```bash
# Run a single ticker without execution
python3 main.py AAPL --agent sentiment

# Check health
curl -s http://localhost:9090/health | python3 -m json.tool
```

**Step 5: Deactivate kill switch**
```bash
python3 main.py --kill-switch off
```

**Step 6: Log the incident**
- Record: what failed, time of failure, time of recovery, root cause
- Update `CHANGELOG.md` if a code fix was applied

---

### 4.2 Bad Trades — Kill Switch → Investigate

```
SEVERITY: HIGH   USE WHEN: Unexpected trades, wrong direction, excessive size
```

**Step 1: Stop all new trades immediately**
```bash
python3 main.py --kill-switch on
echo "Kill switch activated at $(date)"
```

**Step 2: Identify the problematic trades**
```bash
# Last 24 hours of trades with signal context
sqlite3 news_trading.db \
  "SELECT t.id, t.ticker, t.action, t.shares, t.entry_price,
          c.combined_signal, c.confidence, c.sentiment_signal, c.technical_signal,
          t.trade_date
   FROM trade_history t
   LEFT JOIN combined_signals c ON t.ticker = c.ticker
     AND date(t.trade_date) = date(c.created_at)
   WHERE date(t.trade_date) >= date('now', '-1 day')
   ORDER BY t.trade_date DESC;"
```

**Step 3: Check for data quality issues**
```bash
# Were any signals based on degraded data?
sqlite3 news_trading.db \
  "SELECT service, event_type, error_msg, created_at
   FROM recovery_log
   WHERE date(created_at) >= date('now', '-1 day')
   ORDER BY created_at DESC;"

# Were headlines from a fallback source?
python3 -c "
from data.fallback_coordinator import FallbackCoordinator
print(FallbackCoordinator.get_status())
"
```

**Step 4: Decide on action**

| Finding | Action |
|---|---|
| Trade based on Level 3 cached news | Acceptable — but add `prefer_cached_over_fallback: true` to watchlist.yaml |
| Trade based on sentiment only (technical N/A) | Review `technical_agent.py` fallback config |
| Wrong signal direction | Check if Anthropic was using rule-based fallback (`degraded: True`) |
| Position size too large | Verify Kelly fraction cap and `ACCOUNT_BALANCE` setting |
| All looks correct | False alarm — deactivate kill switch |

**Step 5: Reactivate when resolved**
```bash
python3 main.py --kill-switch off
```

---

### 4.3 Data Corruption — Restore from Backup

```
SEVERITY: CRITICAL   USE WHEN: PRAGMA integrity_check fails
```

**Step 1: Verify corruption**
```bash
sqlite3 news_trading.db "PRAGMA integrity_check;"
# If not "ok", proceed with restore
```

**Step 2: Stop all writes**
```bash
python3 main.py --kill-switch on
pkill -f daily_runner.py
```

**Step 3: Back up the corrupted file for analysis**
```bash
cp news_trading.db "news_trading_corrupted_$(date +%Y%m%d_%H%M).db"
```

**Step 4: Restore from most recent backup**
```bash
ls -lt backups/*.db | head -5   # find most recent

# Restore
cp backups/news_trading_YYYYMMDD.db news_trading.db

# Verify restored DB
sqlite3 news_trading.db "PRAGMA integrity_check;"
```

**Step 5: Re-run schema migration to catch any new tables**
```bash
python3 -c "from storage.database import Database; Database()"
```

**Step 6: Restart and verify**
```bash
python3 main.py --kill-switch off
python3 scheduler/daily_runner.py --now --no-execute
```

**Step 7: Investigate corruption cause**
- Check for disk full: `df -h`
- Check for power loss / force-kill: system uptime logs
- Check SQLite WAL mode: `sqlite3 news_trading.db "PRAGMA journal_mode;"`

---

## 5. Health Check Commands

Quick-reference one-liners for common checks:

```bash
# System overview
python3 -c "
from utils.api_recovery import APIRecovery
from data.fallback_coordinator import FallbackCoordinator
from utils.network_recovery import get_cache
print('=== Circuit Breakers ===')
for svc, info in APIRecovery.get_status().items():
    print(f'  {svc}: {info[\"state\"]} (failures={info[\"failures\"]})')
print('=== Fallback Levels ===')
for svc, info in FallbackCoordinator.get_status().items():
    print(f'  {svc}: L{info[\"level\"]} ({info[\"source\"]})')
print('=== Cache ===')
print(' ', get_cache().stats())
"

# DB quick stats
sqlite3 news_trading.db \
  "SELECT 'Signals today:',  count(*) FROM combined_signals WHERE date(created_at)=date('now')
   UNION ALL
   SELECT 'Trades today:',   count(*) FROM trade_history   WHERE date(trade_date)=date('now')
   UNION ALL
   SELECT 'Open positions:', count(*) FROM portfolio       WHERE status='open';"

# Test full pipeline (no trades, no output clutter)
python3 main.py AAPL --agent sentiment 2>/dev/null | grep -E "(signal|confidence|degraded)"

# Fallback chain test (network calls)
python3 -m data.news_aggregator --test-fallbacks --ticker AAPL --max 3
```

---

## 6. Backup and Restore

### Automated daily backup (local cron)

Add to crontab (`crontab -e`):
```cron
# Daily backup at 23:55 — keep 30 days
55 23 * * * cd /path/to/news-trading-system && \
  cp news_trading.db backups/news_trading_$(date +\%Y\%m\%d).db && \
  find backups/ -name "news_trading_*.db" -mtime +30 -delete
```

### Railway PostgreSQL backup (cloud)

```bash
# Manual pg_dump (requires DATABASE_URL)
pg_dump $DATABASE_URL > backups/pg_$(date +%Y%m%d).sql

# Restore
psql $DATABASE_URL < backups/pg_YYYYMMDD.sql
```

### Backup verification

```bash
# Verify a backup is readable and passes integrity check
BACKUP="backups/news_trading_YYYYMMDD.db"
sqlite3 "$BACKUP" "PRAGMA integrity_check; SELECT count(*) FROM analysis_runs;"
```
