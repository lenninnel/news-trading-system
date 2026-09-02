# NTS systemd unit definitions (user scope)

These are copies of the unit files that live in
`~/.config/systemd/user/` on the VPS. The repo is the source of
truth for the definitions; the VPS copies must be updated when
these change.

## Deploy a new or updated unit

```bash
# nts-backup
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-backup.service ~/.config/systemd/user/'
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-backup.timer ~/.config/systemd/user/'
ssh trading-vps 'systemctl --user daemon-reload && systemctl --user enable --now nts-backup.timer'

# nts-ohlc-ingest
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-ohlc-ingest.service ~/.config/systemd/user/'
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-ohlc-ingest.timer ~/.config/systemd/user/'
ssh trading-vps 'systemctl --user daemon-reload && systemctl --user enable --now nts-ohlc-ingest.timer'

# nts-preann-estimates (Q-013 — recorded-only Benzinga T-1 estimate snapshot)
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-preann-estimates.service ~/.config/systemd/user/'
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-preann-estimates.timer ~/.config/systemd/user/'
ssh trading-vps 'systemctl --user daemon-reload && systemctl --user enable --now nts-preann-estimates.timer'
```

# nts-alert@ (OnFailure → Telegram, user scope) + nts-watchdog (absence alerting)
ssh trading-vps 'cd ~/news-trading-system && cp deployment/systemd/nts-alert@.service deployment/systemd/nts-watchdog.service deployment/systemd/nts-watchdog.timer ~/.config/systemd/user/'
ssh trading-vps 'cd ~/news-trading-system && mkdir -p ~/.config/systemd/user/nts-trading.service.d && cp deployment/systemd/nts-trading.service.d/10-alerting.conf ~/.config/systemd/user/nts-trading.service.d/'
ssh trading-vps 'systemctl --user daemon-reload && systemctl --user enable --now nts-watchdog.timer'
# re-copy the oneshot units above afterwards — they now carry OnFailure=nts-alert@%n.service
```

See `docs/ALERTING.md` for what alerts, what deliberately does not, and how
the watchdog behaves.

## Verify

```bash
ssh trading-vps 'systemctl --user list-timers --no-pager | grep -E "nts-backup|nts-ohlc-ingest|nts-preann-estimates|nts-watchdog"'
ssh trading-vps 'cd ~/news-trading-system && set -a && . ./.env && set +a && /usr/bin/python3 scripts/watchdog.py --dry-run --heartbeat'   # preview, sends nothing
ssh trading-vps 'systemctl --user start nts-alert@selftest.service'   # end-to-end OnFailure chain → one "❌ selftest FAILED" message
ssh trading-vps 'systemctl --user status nts-backup.service --no-pager'
ssh trading-vps 'systemctl --user status nts-ohlc-ingest.service --no-pager'
ssh trading-vps 'journalctl --user -u nts-backup -n 30 --no-pager'
ssh trading-vps 'journalctl --user -u nts-ohlc-ingest -n 30 --no-pager'
```

## Existing services (already deployed)

- nts-api.service
- nts-trading.service
- nts-mcp.service
- nts-dashboard.service
- nts-backup.timer (daily 00:30 UTC)
- nts-ohlc-ingest.timer (daily 22:30 UTC — Polygon daily-bar incremental ingest into `daily_ohlc`.
  22:30 UTC is after the US close in both DST and winter, and the ingest window includes
  *today* past 22:00 UTC, so after a run on trading day T the store's MAX(date) must be T.
  A freshness gate enforces that per ticker against the US trading calendar
  (`data/market_calendar.py`); a stale store exits 1 and sends a Telegram alert via the
  app credentials — a silent "ok" on zero new rows is not possible.)
- nts-watchdog.timer (every 15 min — `scripts/watchdog.py`, stdlib-only, /usr/bin/python3. Checks
  daemon active + auto-restarts, every scheduled session claimed its `session_runs` slot, `daily_ohlc`
  holds the last completed US trading day, ingest/backup timers active, backup < 26 h, IB Gateway port,
  disk, DB readable. One message per run, re-alert every 6 h, recovery message, daily 06:00 UTC status
  block as its own liveness signal. Its failure routes through nts-alert@.)
- nts-alert@.service (template; `OnFailure=nts-alert@%n.service` on every unit above and, via the
  drop-in `nts-trading.service.d/10-alerting.conf`, on the daemon. Posts unit result + journal tail.)
- nts-preann-estimates.timer (daily 11:00 UTC — Q-013 recorded-only Benzinga T-1 pre-announcement estimate snapshot into `benzinga_estimate_preann_snapshot`; standalone, off the trading path)

The unit files for the four existing nts-* services are NOT
currently captured in this repo (only the `nts-trading.service.d/`
drop-in is). Adding them is a follow-up.
