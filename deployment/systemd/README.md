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
```

## Verify

```bash
ssh trading-vps 'systemctl --user list-timers --no-pager | grep -E "nts-backup|nts-ohlc-ingest"'
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
- nts-ohlc-ingest.timer (daily 22:30 UTC — Polygon daily-bar incremental ingest into `daily_ohlc`)

The unit files for the four existing nts-* services are NOT
currently captured in this repo. Adding them is a follow-up.
