# Troubleshooting Guide — News Trading System

> **Audience**: Operators who encounter errors or unexpected behaviour in production.
> **Scope**: Self-service diagnosis and resolution for the most common issues.

---

## Quick Reference Table

| Symptom | Most Likely Cause | First Action |
|---|---|---|
| System not trading | Kill switch active or scheduler not running | `python3 main.py --kill-switch status` |
| API errors (NewsAPI/Anthropic) | Rate limit or key expired | Check fallback status, verify keys |
| Database locked | Another process holds a write lock | Activate kill switch → wait → restart |
| Out of memory | Large portfolio or unconstrained cache | Prune cache, reduce watchlist size |
| Telegram alerts not arriving | Bot token or chat ID misconfigured | Verify token with curl test below |
| Scheduler did not run | Cron/daemon stopped or crashed | Check process and logs |
| Prices stuck / stale | yfinance circuit breaker open | Check fallback coordinator status |
| Dashboard blank / error | DB not found or port conflict | Verify DB path, restart Streamlit |

---

## 1. System Not Trading

### Symptoms
- No paper trades logged despite `execution.enabled: true`
- `python3 main.py AAPL` produces signals but no trade ID
- Dashboard Portfolio page is empty

### Diagnostic commands

```bash
# Check kill switch status
python3 main.py --kill-switch status

# Check execution flag in config
grep -A2 'execution:' config/watchlist.yaml

# Verify last scheduled run
ls -lt scheduler/logs/ | head -5
cat scheduler/logs/$(date +%Y-%m-%d).log | grep -E "(SIGNAL|TRADE|ERROR)"

# Check if confidence threshold is being met (need ≥ 30%)
sqlite3 news_trading.db \
  "SELECT ticker, combined_signal, confidence FROM combined_signals \
   ORDER BY created_at DESC LIMIT 10;"
```

### Solutions

| Cause | Fix |
|---|---|
| Kill switch is ON | `python3 main.py --kill-switch off` then restart |
| `execution.enabled: false` | Edit `config/watchlist.yaml` → set `enabled: true` |
| All signals are HOLD | Widen watchlist or adjust `BUY_THRESHOLD` in `config/settings.py` |
| Confidence always < 30% | Review sentiment scoring; check if Anthropic API is degraded |
| Scheduler didn't run | See section 6 (Scheduler) |

### When to escalate
Escalate if kill switch is off, execution is enabled, signals are non-HOLD, and trades are still not appearing — this may indicate a database write failure.

---

## 2. API Errors

### Symptoms
- Log lines: `[DEGRADED L1]`, `[DEGRADED L2]`, `Circuit OPEN for 'newsapi'`
- Sentiment fallback active (`degraded: True` in results)
- No headlines fetched for one or more tickers

### Diagnostic commands

```bash
# Check which fallback level is active for each service
python3 -c "
from data.fallback_coordinator import FallbackCoordinator
import json
print(json.dumps(FallbackCoordinator.get_status(), indent=2))
"

# Check circuit breaker states
python3 -c "
from utils.api_recovery import APIRecovery
import json
print(json.dumps(APIRecovery.get_status(), indent=2))
"

# Test NewsAPI manually
curl -s "https://newsapi.org/v2/everything?q=AAPL&apiKey=$NEWSAPI_KEY&pageSize=1" \
  | python3 -m json.tool | head -20

# Test Anthropic API
curl -s https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-haiku-4-5-20251001","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' \
  | python3 -m json.tool

# Check rate limit usage (NewsAPI free tier: 100 req/day)
sqlite3 news_trading.db \
  "SELECT service, event_type, error_msg, created_at FROM recovery_log \
   WHERE date(created_at) = date('now') ORDER BY created_at DESC LIMIT 20;"

# Test all news fallback levels (real network calls)
python3 -m data.news_aggregator --test-fallbacks --ticker AAPL
```

### Solutions

| Cause | Fix |
|---|---|
| NewsAPI 429 rate limit | Wait until midnight UTC; free tier = 100 req/day |
| Anthropic 429 rate limit | Reduce `MAX_HEADLINES` in `config/settings.py` |
| Invalid API key | Re-export key: `export NEWSAPI_KEY=newkey` and restart |
| Circuit breaker OPEN | Wait 5 min for auto-reset, or: `python3 -c "from utils.api_recovery import APIRecovery; APIRecovery.reset_circuit('newsapi')"` |
| Alpha Vantage not set | Add `ALPHA_VANTAGE_KEY` to `.env` for Level 1 price fallback |

### Fallback chain status explained

```
Level 0 (Primary)  — normal operation
Level 1 (Degraded) — primary failed; using RSS / Alpha Vantage
Level 2 (Degraded) — all paid sources failed; using Google News / Yahoo JSON
Level 3 (Degraded) — all live sources failed; using cached data (<24h)
Level 4 (Failed)   — all sources exhausted; result is empty / None price
```

If any service is stuck on Level 1+ for >24 h, `FallbackCoordinator.check_and_alert()` will emit a WARNING in the logs.

### When to escalate
Escalate if the circuit breaker remains OPEN after manual reset, or if all four fallback levels fail simultaneously — this usually indicates a network-level outage.

---

## 3. Database Locked

### Symptoms
- Error: `sqlite3.OperationalError: database is locked`
- Dashboard fails to load with database error
- Scheduler exits early with lock error

### Diagnostic commands

```bash
# Check which processes have the DB open
lsof news_trading.db 2>/dev/null || fuser news_trading.db 2>/dev/null

# Check for WAL files (indicate an uncommitted transaction)
ls -lh news_trading.db news_trading.db-wal news_trading.db-shm 2>/dev/null

# Check DB integrity
sqlite3 news_trading.db "PRAGMA integrity_check;"

# See recent write activity
sqlite3 news_trading.db \
  "SELECT name FROM sqlite_master WHERE type='table';" \
  ".tables"
```

### Resolution steps (in order)

1. **Activate kill switch** to stop new writes:
   ```bash
   python3 main.py --kill-switch on
   ```

2. **Identify and stop the locking process**:
   ```bash
   kill $(lsof -t news_trading.db)
   ```

3. **Wait 10 seconds** for SQLite WAL checkpoint to complete.

4. **Verify DB integrity**:
   ```bash
   sqlite3 news_trading.db "PRAGMA integrity_check;"
   # Should output: ok
   ```

5. **Deactivate kill switch and restart**:
   ```bash
   python3 main.py --kill-switch off
   python3 scheduler/daily_runner.py --now
   ```

### Prevention
- Never run two scheduler instances simultaneously.
- Use the kill switch before stopping the process forcibly (SIGKILL).

### When to escalate
Escalate if `PRAGMA integrity_check` returns anything other than `ok` — restore from backup (see RUNBOOK.md § Emergency Procedures).

---

## 4. Out of Memory

### Symptoms
- Process killed by OS (`OOMKilled` in Railway logs)
- Dashboard slow or unresponsive
- Python `MemoryError` in logs

### Diagnostic commands

```bash
# Current process memory
ps aux | grep -E "(python|streamlit)" | awk '{print $11, $6"KB"}'

# DB file size
du -sh news_trading.db

# Cache stats
python3 -c "
from utils.network_recovery import get_cache
stats = get_cache().stats()
print(stats)
"

# Count rows per table
sqlite3 news_trading.db \
  ".tables" \
  "SELECT 'headline_scores', count(*) FROM headline_scores UNION ALL \
   SELECT 'technical_signals', count(*) FROM technical_signals UNION ALL \
   SELECT 'combined_signals', count(*) FROM combined_signals;"
```

### Solutions

| Cause | Fix |
|---|---|
| Large response cache | `python3 -c "from utils.network_recovery import get_cache; get_cache().clear()"` |
| Many tickers in watchlist | Reduce watchlist to ≤ 10 tickers in `config/watchlist.yaml` |
| Old DB rows accumulating | Archive rows older than 90 days (see RUNBOOK.md § Monthly Tasks) |
| Streamlit + yfinance both loaded | Separate dashboard process from scheduler |
| Railway free tier RAM (512 MB) | Upgrade to Hobby plan or reduce concurrent agents |

### When to escalate
If OOM occurs on a watchlist of fewer than 5 tickers with an empty cache, the issue is likely in a third-party library (yfinance, pandas). Open an issue with the full traceback.

---

## 5. Telegram Not Working

### Symptoms
- No alerts received during trading session
- `send_signal()` calls succeed locally but no message arrives
- Dashboard shows signals but phone shows nothing

### Diagnostic commands

```bash
# Verify environment variables are set
echo "Token: ${TELEGRAM_BOT_TOKEN:0:10}..."
echo "Chat ID: $TELEGRAM_CHAT_ID"

# Send a direct test message via Telegram API
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
  -d "chat_id=$TELEGRAM_CHAT_ID&text=Test+from+news-trading-system"

# Check if Telegram is enabled in config
grep -A3 'telegram:' config/watchlist.yaml

# Check notification logs
sqlite3 news_trading.db \
  "SELECT * FROM recovery_log WHERE service='telegram' ORDER BY created_at DESC LIMIT 5;"
```

### Solutions

| Cause | Fix |
|---|---|
| Wrong `TELEGRAM_BOT_TOKEN` | Get correct token from @BotFather on Telegram |
| Wrong `TELEGRAM_CHAT_ID` | Send `/start` to your bot, then: `curl https://api.telegram.org/bot$TOKEN/getUpdates` to find chat ID |
| `telegram.enabled: false` | Set to `true` in `config/watchlist.yaml` |
| Bot was blocked | Unblock the bot in your Telegram app |
| Railway env vars missing | `railway variables set TELEGRAM_BOT_TOKEN=...` |
| Bot not started by user | Message the bot first with `/start` |

### When to escalate
If `curl` returns `{"ok":true}` but messages still don't arrive, the issue is in Telegram's infrastructure. Check [Telegram Status](https://core.telegram.org/status).

---

## 6. Scheduler Did Not Run

### Symptoms
- No log file for today in `scheduler/logs/`
- No signals in DB with today's date
- Telegram end-of-day summary not received

### Diagnostic commands

```bash
# Check if cron job is installed
crontab -l | grep daily_runner

# Check if daemon process is running
ps aux | grep daily_runner

# Check last log entry
ls -lt scheduler/logs/ | head -3
tail -50 scheduler/logs/$(ls -t scheduler/logs/ | head -1)

# Check Railway worker logs (cloud deployment)
railway logs --service worker --tail 100

# Run manually to verify it works
python3 scheduler/daily_runner.py --now --no-execute
```

### Solutions

| Cause | Fix |
|---|---|
| Cron not installed | `bash scheduler/install_cron.sh` |
| Daemon crashed | `python3 scheduler/daily_runner.py` (restart daemon) |
| Railway worker stopped | Redeploy: `railway up` |
| Wrong time zone | Check `schedule.time` in `watchlist.yaml` — it's local machine time |
| Weekend (weekdays_only: true) | Expected — scheduler skips Sat/Sun |

---

## 7. Stale / Wrong Prices

### Symptoms
- `[CACHE FALLBACK]` in logs
- `degraded: True` in PriceResult
- Price data is hours or days old

### Diagnostic commands

```bash
# Check yfinance circuit status
python3 -c "
from utils.api_recovery import APIRecovery
s = APIRecovery.get_status()
print('yfinance:', s.get('yfinance', 'no circuit'))
"

# Test yfinance directly
python3 -c "
import yfinance as yf
t = yf.Ticker('AAPL')
print(t.fast_info)
"

# Check price fallback levels
python3 -m data.price_fallback AAPL 2>/dev/null || \
python3 -c "
from data.price_fallback import PriceFallback
r = PriceFallback().get_price('AAPL')
print(f'Price: {r.price}  Source: {r.source}  Level: {r.level}  Fresh: {r.is_fresh}')
"
```

### Solutions

| Cause | Fix |
|---|---|
| yfinance circuit OPEN | Reset: `python3 -c "from utils.api_recovery import APIRecovery; APIRecovery.reset_circuit('yfinance')"` |
| Yahoo Finance rate-limited | Wait 15 min; yfinance auto-resets |
| No `ALPHA_VANTAGE_KEY` set | Add key to `.env` for Level 1 price fallback |
| Market closed | Expected — prices are previous-close during off-hours |

---

## Escalation Guide

| Situation | Action |
|---|---|
| Single API intermittently failing | Self-fix (see sections above) |
| Circuit breaker stuck open after reset | Open GitHub issue with log excerpt |
| `PRAGMA integrity_check` fails | Restore from backup immediately — see RUNBOOK.md |
| All 4 fallback levels fail simultaneously | Network outage — wait and retry in 30 min |
| Railway OOMKilled repeatedly on small watchlist | Upgrade plan or open GitHub issue |
| Unexpected trades in paper portfolio | Activate kill switch → investigate combined_signals table |

### Collecting diagnostic info for a bug report

```bash
# Capture system snapshot
python3 -c "
import sys, platform
from utils.api_recovery import APIRecovery
from utils.network_recovery import get_cache
from data.fallback_coordinator import FallbackCoordinator
print('Python:', sys.version)
print('Platform:', platform.platform())
print('Circuits:', APIRecovery.get_status())
print('Cache stats:', get_cache().stats())
print('Fallbacks:', FallbackCoordinator.get_status())
" 2>&1 | tee /tmp/nts_diagnostic.txt

echo "Attach /tmp/nts_diagnostic.txt to your GitHub issue."
```
