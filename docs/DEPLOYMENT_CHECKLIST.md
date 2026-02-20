# Deployment Checklist — News Trading System

> Run through this checklist before every production deployment.
> Check each item and note any failures — do not proceed if a required item fails.

---

## Pre-Deployment (Before Pushing)

### Code Quality

- [ ] All unit tests pass locally:
  ```bash
  python3 -m pytest tests/ -v
  # Expected: 202 passed, 0 failed
  ```
- [ ] No linting errors:
  ```bash
  python3 -m flake8 . --max-line-length=120 --exclude=.venv,__pycache__
  ```
- [ ] No obvious secrets committed:
  ```bash
  git diff HEAD --name-only | xargs grep -l "sk-ant\|apikey\|password" 2>/dev/null
  # Expected: no output
  ```
- [ ] `requirements.txt` is up to date:
  ```bash
  pip freeze | grep -E "(yfinance|anthropic|requests|streamlit)" | sort
  ```

### Resilience test

- [ ] Resilience test passes:
  ```bash
  python3 scheduler/daily_runner.py --resilience-test
  # Expected: 6/6 tests pass
  ```

---

## Environment Variables

Set all required variables in your deployment target (`.env` locally, Railway variables in cloud).

### Required

- [ ] `ANTHROPIC_API_KEY` — get at [console.anthropic.com](https://console.anthropic.com)
  ```bash
  # Verify it's set and non-empty
  [ -n "$ANTHROPIC_API_KEY" ] && echo "OK" || echo "MISSING"
  ```
- [ ] `NEWSAPI_KEY` — get at [newsapi.org](https://newsapi.org)
  ```bash
  [ -n "$NEWSAPI_KEY" ] && echo "OK" || echo "MISSING"
  ```

### Optional (but recommended)

- [ ] `ALPHA_VANTAGE_KEY` — price fallback Level 1; get at [alphavantage.co](https://www.alphavantage.co)
- [ ] `TELEGRAM_BOT_TOKEN` — real-time alerts via Telegram
- [ ] `TELEGRAM_CHAT_ID` — your Telegram chat ID
- [ ] `ACCOUNT_BALANCE` — paper-trading account size in USD (default: 10000)
- [ ] `DB_PATH` — custom SQLite path (default: `news_trading.db`)

### Cloud / production only

- [ ] `DATABASE_URL` — PostgreSQL DSN (Railway auto-injects when Postgres plugin is attached)
- [ ] `ENVIRONMENT=production` — enables production-mode logging and safety checks
- [ ] `HEALTH_PORT=8080` — port for the JSON health endpoint

### Verify all required variables are present

```bash
python3 -c "
import os
required = ['ANTHROPIC_API_KEY', 'NEWSAPI_KEY']
optional = ['ALPHA_VANTAGE_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID',
            'ACCOUNT_BALANCE', 'DB_PATH', 'DATABASE_URL', 'ENVIRONMENT']
print('Required:')
for k in required:
    status = 'OK' if os.environ.get(k) else 'MISSING <<<'
    print(f'  {k}: {status}')
print('Optional:')
for k in optional:
    status = 'set' if os.environ.get(k) else 'not set'
    print(f'  {k}: {status}')
"
```

---

## Database

- [ ] Database initialises without errors:
  ```bash
  python3 -c "from storage.database import Database; db = Database(); print('DB OK')"
  ```
- [ ] Schema migration passes (all expected tables exist):
  ```bash
  python3 -c "
  from storage.database import Database
  db = Database()
  tables = {r['name'] for r in db._select(\"SELECT name FROM sqlite_master WHERE type='table'\")}
  expected = {
      'analysis_runs', 'headline_scores', 'technical_signals', 'combined_signals',
      'risk_calculations', 'portfolio', 'trade_history', 'strategy_signals',
      'strategy_performance', 'recovery_log'
  }
  missing = expected - tables
  print('Missing tables:', missing or 'none')
  print('Schema check:', 'PASS' if not missing else 'FAIL')
  "
  ```
- [ ] Database integrity check passes:
  ```bash
  sqlite3 news_trading.db "PRAGMA integrity_check;"
  # Expected: ok
  ```
- [ ] Backup exists before deployment:
  ```bash
  mkdir -p backups
  cp news_trading.db "backups/pre_deploy_$(date +%Y%m%d_%H%M).db"
  echo "Backup created"
  ```

---

## Health Checks

- [ ] Health endpoint responds (if health service is running):
  ```bash
  curl -s http://localhost:9090/health | python3 -m json.tool
  # Expected: {"status": "healthy", ...}
  ```
- [ ] Single-ticker analysis completes without error:
  ```bash
  python3 main.py AAPL --agent sentiment --no-execute 2>&1 | grep -E "(signal|ERROR|Traceback)"
  # Expected: signal line present, no ERROR or Traceback
  ```
- [ ] News aggregator fallback chain operational:
  ```bash
  python3 -c "
  from data.news_aggregator import NewsAggregator
  from unittest.mock import patch, MagicMock
  # Test with NewsAPI (mocked to avoid spending quota)
  resp = MagicMock()
  resp.raise_for_status = lambda: None
  resp.json.return_value = {'articles': [{'title': 'Test headline'}]}
  with patch('data.news_aggregator.requests.get', return_value=resp):
      r = NewsAggregator().fetch_with_metadata('AAPL')
  print(f'News chain: L{r.level} ({r.source}) — {r.count} headline(s) — PASS')
  "
  ```
- [ ] Price fallback chain operational:
  ```bash
  python3 -c "
  from data.price_fallback import PriceFallback
  from unittest.mock import patch, MagicMock
  t = MagicMock()
  t.info = {'currentPrice': 100.0, 'currency': 'USD', 'longName': 'Test'}
  with patch('data.price_fallback.yf.Ticker', return_value=t), \
       patch('data.price_fallback.requests.get', side_effect=Exception('blocked')):
      r = PriceFallback().get_price('AAPL')
  print(f'Price chain: L{r.level} ({r.source}) price={r.price} — PASS')
  "
  ```

---

## Tests Passing

- [ ] Full test suite passes:
  ```bash
  python3 -m pytest tests/ --tb=short -q
  # Expected: 202 passed, 0 failed
  ```
- [ ] Recovery tests pass:
  ```bash
  python3 -m pytest tests/test_recovery.py tests/test_fallbacks.py -v -q
  # Expected: all pass
  ```

---

## Monitoring Active

- [ ] Health monitor is configured in `config/watchlist.yaml`:
  ```bash
  grep -A5 'health_monitor:' config/watchlist.yaml
  # Expected: enabled: true
  ```
- [ ] Log directory exists and is writable:
  ```bash
  mkdir -p scheduler/logs
  touch scheduler/logs/.write_test && rm scheduler/logs/.write_test && echo "Writable"
  ```
- [ ] Fallback coordinator has no stale alerts:
  ```bash
  python3 -c "
  from data.fallback_coordinator import FallbackCoordinator
  alerts = FallbackCoordinator.check_and_alert()
  print(f'Fallback alerts: {len(alerts)} (expected: 0)')
  "
  ```

---

## Kill Switch

- [ ] Kill switch module is importable:
  ```bash
  python3 -c "from scheduler.daily_runner import KillSwitch; print('KillSwitch: OK')" 2>/dev/null || \
  python3 -c "import main; print('Kill switch accessible via main.py')"
  ```
- [ ] Kill switch is OFF before deploying (do not deploy with it active):
  ```bash
  python3 main.py --kill-switch status
  # Expected: Kill switch is OFF (or similar — OFF is required to proceed)
  ```
- [ ] Verify kill switch can be toggled:
  ```bash
  python3 main.py --kill-switch on  && \
  python3 main.py --kill-switch status && \
  python3 main.py --kill-switch off  && \
  python3 main.py --kill-switch status
  # Expected: on → OFF transition works correctly
  ```

---

## Backups Configured

- [ ] Backup directory exists:
  ```bash
  mkdir -p backups && echo "backups/ exists"
  ```
- [ ] Pre-deployment backup created (see Database section above):
  ```bash
  ls -lh backups/pre_deploy_*.db | tail -1
  ```
- [ ] Cron backup job installed (local deployment):
  ```bash
  crontab -l | grep -q "backups/" && echo "Cron backup: configured" || \
  echo "Cron backup: NOT configured — add to crontab"
  ```
  If not configured:
  ```bash
  (crontab -l 2>/dev/null; echo "55 23 * * * cd $(pwd) && cp news_trading.db backups/news_trading_\$(date +\\%Y\\%m\\%d).db && find backups/ -name '*.db' -mtime +30 -delete") | crontab -
  ```

---

## Telegram Notifications (if enabled)

- [ ] Bot token is valid:
  ```bash
  curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe" | python3 -m json.tool | grep '"ok"'
  # Expected: "ok": true
  ```
- [ ] Test message delivered:
  ```bash
  curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
    -d "chat_id=$TELEGRAM_CHAT_ID&text=Deployment+check+✓" | python3 -m json.tool | grep '"ok"'
  # Expected: "ok": true — and a message appears in Telegram
  ```
- [ ] `telegram.enabled: true` in `config/watchlist.yaml`

---

## Scheduler (Daily Runner)

- [ ] Scheduler dry-run completes without error:
  ```bash
  python3 scheduler/daily_runner.py --now --no-execute 2>&1 | tail -5
  # Expected: completion message, no unhandled exceptions
  ```
- [ ] Cron or Railway worker is configured to auto-start:

  **Local (cron):**
  ```bash
  crontab -l | grep daily_runner
  # Or reinstall: bash scheduler/install_cron.sh
  ```

  **Railway:**
  ```bash
  railway service list   # confirm 'worker' service exists
  railway logs --service worker --tail 20
  ```

---

## Post-Deployment Verification

Run these checks **after** deploying:

- [ ] First scheduled run completes successfully:
  ```bash
  # Trigger manually and watch logs
  python3 scheduler/daily_runner.py --now 2>&1 | tail -20
  ```
- [ ] Signals appear in the database:
  ```bash
  sqlite3 news_trading.db \
    "SELECT ticker, combined_signal, confidence, created_at
     FROM combined_signals ORDER BY created_at DESC LIMIT 5;"
  ```
- [ ] Dashboard loads without error:
  ```bash
  # Start dashboard, open http://localhost:8501 in browser
  streamlit run dashboard/app.py
  ```
- [ ] Health endpoint returns healthy status:
  ```bash
  curl -s http://localhost:9090/health
  ```
- [ ] No errors in logs for first 10 minutes after deployment:
  ```bash
  tail -f scheduler/logs/$(date +%Y-%m-%d).log | grep -E "(ERROR|CRITICAL|Traceback)"
  ```

---

## Sign-off

| Check | Completed by | Notes |
|---|---|---|
| All tests passing | | |
| Environment variables verified | | |
| Database migrated and healthy | | |
| Kill switch tested | | |
| Backups configured | | |
| Monitoring active | | |
| Post-deploy verification | | |

**Deployment approved:** _______________________ **Date:** _______________
