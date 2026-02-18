# Telegram Notifications

Real-time trading alerts delivered to a Telegram chat via the Bot API.

## What gets sent?

| Event | Trigger | Example |
|---|---|---|
| Signal alert | Every ticker analysed by the scheduler | 🚀 *AAPL* — `STRONG BUY` — Confidence: **82%** |
| Trade executed | Every paper trade logged | 🟢 *Paper Trade Executed* — `NVDA` — BUY 14 shares @ $820.50 |
| Daily summary | End of each scheduler run | ✅ *Daily Trading Summary* — 5 signals, 2 trades |
| Error | Any unhandled exception | ❗ Trading System Error |

## Setup

### 1 — Create a Telegram bot

1. Open Telegram and search for **@BotFather**.
2. Send `/newbot` and follow the prompts.
3. Copy the **HTTP API token** (format: `123456789:ABC-DEF...`).

### 2 — Find your chat ID

1. Send any message to your new bot.
2. Open this URL in a browser (replace `<TOKEN>` with your token):

   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```

3. In the JSON response, find `"chat": {"id": <YOUR_CHAT_ID>}`.
   For a group/channel the ID is negative (e.g. `-1001234567890`).

### 3 — Set environment variables

Add to your shell profile or `.env` file (never commit these to git):

```bash
export TELEGRAM_BOT_TOKEN="123456789:ABC-DEF..."
export TELEGRAM_CHAT_ID="-1001234567890"
```

### 4 — Enable in watchlist.yaml

```yaml
telegram:
  enabled: true
  bot_token: "${TELEGRAM_BOT_TOKEN}"   # reads from env var
  chat_id: "${TELEGRAM_CHAT_ID}"       # reads from env var
  dashboard_url: "http://localhost:8501"   # shown as inline button (optional)
```

## Usage

### Scheduler (recommended)

```bash
# Run immediately with notifications
python3 scheduler/daily_runner.py --now --notify

# Daemon mode with notifications
python3 scheduler/daily_runner.py --notify
```

### CLI (single ticker)

```bash
python3 main.py AAPL --execute --notify
```

## Message formatting

Messages use **Markdown** (`parse_mode=Markdown`).
A `📊 View Dashboard` inline button is added automatically when `dashboard_url` is set.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Telegram enabled but bot_token/chat_id missing` | Check that env vars are exported in the same shell session |
| `Telegram API error 401` | Invalid token — re-check with BotFather |
| `Telegram API error 400 Bad Request: chat not found` | Send at least one message to the bot first, then re-fetch `getUpdates` |
| No button appears | Set `dashboard_url` in `watchlist.yaml` |
| Works in terminal but not in cron | Add `export TELEGRAM_BOT_TOKEN=...` to `/etc/environment` or the cron job itself |
