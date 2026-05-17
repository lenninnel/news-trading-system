# NTS systemd unit definitions (user scope)

These are copies of the unit files that live in
`~/.config/systemd/user/` on the VPS. The repo is the source of
truth for the definitions; the VPS copies must be updated when
these change.

## Deploy a new or updated unit

```bash
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-backup.service ~/.config/systemd/user/'
ssh trading-vps 'cp /home/trading/news-trading-system/deployment/systemd/nts-backup.timer ~/.config/systemd/user/'
ssh trading-vps 'systemctl --user daemon-reload && systemctl --user enable --now nts-backup.timer'
```

## Verify

```bash
ssh trading-vps 'systemctl --user list-timers --no-pager | grep nts-backup'
ssh trading-vps 'systemctl --user status nts-backup.service --no-pager'
ssh trading-vps 'journalctl --user -u nts-backup -n 30 --no-pager'
```

## Existing services (already deployed)

- nts-api.service
- nts-trading.service
- nts-mcp.service
- nts-dashboard.service
- nts-backup.timer (daily 00:30 UTC)

The unit files for the four existing nts-* services are NOT
currently captured in this repo. Adding them is a follow-up.
