# Setup Guide — News Trading System

> **Audience**: Anyone installing the system for the first time.
> Covers macOS, Linux, and Windows.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
   - [macOS](#21-macos)
   - [Linux (Ubuntu / Debian)](#22-linux-ubuntu--debian)
   - [Windows](#23-windows)
3. [API Key Setup](#3-api-key-setup)
   - [NewsAPI](#31-newsapi)
   - [Anthropic (Claude)](#32-anthropic-claude)
   - [Telegram (optional)](#33-telegram-optional)
4. [Configuration](#4-configuration)
5. [First Run Verification](#5-first-run-verification)
6. [Setting Up the Daily Scheduler](#6-setting-up-the-daily-scheduler)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

| Requirement | Minimum version | How to check |
|---|---|---|
| Python | 3.9 | `python3 --version` |
| pip | 21+ | `pip3 --version` |
| Git | Any recent | `git --version` |
| Internet access | — | Required for API calls |

You also need accounts at:
- [newsapi.org](https://newsapi.org) (free tier: 100 requests/day)
- [console.anthropic.com](https://console.anthropic.com) (pay-per-token)

---

## 2. Installation

### 2.1 macOS

**Step 1 — Install Python (if not present)**

```bash
# Check if Python 3.9+ is already installed
python3 --version

# If not, install via Homebrew (recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

**Step 2 — Clone the repository**

```bash
git clone https://github.com/your-username/news-trading-system.git
cd news-trading-system
```

**Step 3 — Create and activate a virtual environment** (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You will see `(.venv)` in your terminal prompt when the environment is active.

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5 — Create your `.env` file** (see [API Key Setup](#3-api-key-setup))

```bash
cp .env.example .env   # if an example file exists
# or create it manually:
nano .env
```

---

### 2.2 Linux (Ubuntu / Debian)

**Step 1 — Install Python and pip**

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git
```

**Step 2 — Clone and enter the project**

```bash
git clone https://github.com/your-username/news-trading-system.git
cd news-trading-system
```

**Step 3 — Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Step 4 — Install dependencies**

```bash
pip install -r requirements.txt
```

> If you see a compiler error for `ta`, install the build tools first:
> ```bash
> sudo apt install -y build-essential python3-dev
> ```

---

### 2.3 Windows

**Step 1 — Install Python**

Download Python 3.11+ from [python.org](https://www.python.org/downloads/windows/).

During installation, check **"Add Python to PATH"**.

**Step 2 — Open a terminal**

Use **PowerShell** or **Windows Terminal** (not Command Prompt).

**Step 3 — Clone and enter the project**

```powershell
git clone https://github.com/your-username/news-trading-system.git
cd news-trading-system
```

**Step 4 — Create and activate a virtual environment**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> If you see an execution policy error:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Step 5 — Install dependencies**

```powershell
pip install -r requirements.txt
```

---

## 3. API Key Setup

### 3.1 NewsAPI

1. Go to [newsapi.org](https://newsapi.org) and click **Get API Key**
2. Create a free account
3. Copy your API key from the dashboard
4. Add it to `.env`:

```env
NEWSAPI_KEY=abc123yourkeyhere
```

**Free tier limits**: 100 requests per day, headlines up to 1 month old.
For production use, consider the [Developer plan](https://newsapi.org/pricing) (500 requests/day).

---

### 3.2 Anthropic (Claude)

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account and add a payment method (pay-per-use)
3. Go to **API Keys** and create a new key
4. Add it to `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...yourkeyhere
```

**Cost estimate**: Each ticker analysis sends approximately 10 messages to Claude (one per headline). Using `claude-sonnet-4-6`, cost is roughly $0.003–0.005 per ticker. Running 5 tickers daily costs approximately $0.50–1.00 per month.

---

### 3.3 Telegram (optional)

Telegram notifications are optional. Skip this section if you don't need them.

**Create a bot**:

1. Open Telegram and search for **@BotFather**
2. Send `/newbot` and follow the prompts
3. Copy the **bot token** (format: `1234567890:ABCdef...`)
4. Add to `.env`:

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
```

**Get your chat ID**:

1. Start a conversation with your new bot
2. Send any message to it
3. Open in a browser (replace `TOKEN` with your bot token):
   ```
   https://api.telegram.org/botTOKEN/getUpdates
   ```
4. Find `"chat":{"id":...}` in the response
5. Add to `.env`:

```env
TELEGRAM_CHAT_ID=987654321
```

**Enable in config**:

```yaml
# config/watchlist.yaml
telegram:
  enabled: true
  bot_token: "${TELEGRAM_BOT_TOKEN}"
  chat_id: "${TELEGRAM_CHAT_ID}"
  dashboard_url: "http://localhost:8501"  # optional inline button
```

For a full walkthrough, see [notifications/README.md](../notifications/README.md).

---

## 4. Configuration

### The `.env` file

Create `.env` in the project root. This file is git-ignored and should never be committed.

```env
# Required
NEWSAPI_KEY=your_newsapi_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional — change where the database is stored
# DB_PATH=/path/to/your/news_trading.db

# Optional — Telegram notifications
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional — Email failure alerts
SMTP_PASSWORD=your_smtp_app_password
```

### The `config/watchlist.yaml` file

Controls which tickers the scheduler analyses daily.

```yaml
# Tickers to analyse every day
watchlist:
  - AAPL        # Apple (US)
  - NVDA        # NVIDIA (US)
  - BAYN.DE     # Bayer (Germany — note the .DE suffix)
  - ASML.AS     # ASML (Netherlands)

# Paper-trading account size in USD
account:
  balance: 10000.0

# Set to false to disable paper-trade logging (analysis only)
execution:
  enabled: true

# Scheduler fires at this local machine time on weekdays
schedule:
  time: "09:30"
  weekdays_only: true

# Email alerts on scheduler failures (optional)
email:
  enabled: false
  smtp_host: smtp.gmail.com
  smtp_port: 587
  from_address: "you@example.com"
  to_address: "you@example.com"
  # Set SMTP_PASSWORD in .env

# Stock screener configuration
screener:
  markets: [US, DE, EU]
  focus_market: DE
  top_candidates: 40
```

### The `config/settings.py` file

Signal thresholds and model settings. Modify only if you want to tune the system.

```python
CLAUDE_MODEL   = "claude-sonnet-4-6"
MAX_HEADLINES  = 10        # headlines fetched per ticker
BUY_THRESHOLD  = 0.30     # avg_score ≥ +0.30 → BUY
SELL_THRESHOLD = -0.30    # avg_score ≤ −0.30 → SELL
DB_PATH        = "news_trading.db"
```

---

## 5. First Run Verification

After completing installation and configuration, verify everything works:

```bash
# 1. Activate your virtual environment (if not already active)
source .venv/bin/activate        # macOS / Linux
# or
.venv\Scripts\Activate.ps1      # Windows

# 2. Run a quick analysis (no paper trade)
python main.py AAPL

# Expected: ticker info, 10 headline analyses, a combined signal report

# 3. Open the dashboard
streamlit run dashboard/app.py
# Expected: browser opens to http://localhost:8501

# 4. Test the screener (German market only, 3 results)
python3 -m agents.screener_agent --markets DE --top 3 --no-db
```

If all three commands complete without errors, the system is correctly installed.

---

## 6. Setting Up the Daily Scheduler

### Option A — Cron job (macOS / Linux, recommended)

The project includes a helper script:

```bash
bash scheduler/install_cron.sh
```

This installs a cron entry that runs the scheduler at 09:30 on weekdays and logs output to `scheduler/logs/`.

To verify the cron entry was created:

```bash
crontab -l
# Should show a line like:
# 30 9 * * 1-5 /usr/bin/python3 /path/to/scheduler/daily_runner.py --now >> ...
```

To remove it:

```bash
crontab -e  # then delete the relevant line
```

### Option B — Daemon mode (any platform)

Run in the background; it will fire at the configured time automatically:

```bash
# macOS / Linux
nohup python3 scheduler/daily_runner.py &
echo "PID: $!"

# Windows (PowerShell)
Start-Process python -ArgumentList "scheduler\daily_runner.py" -WindowStyle Hidden
```

To stop the daemon, find the process ID and kill it:

```bash
ps aux | grep daily_runner.py
kill <PID>
```

### Option C — Manual cron (custom schedule)

```bash
crontab -e
# Add a line (adjust path and time):
30 9 * * 1-5 /usr/bin/python3 /Users/you/news-trading-system/scheduler/daily_runner.py --now >> /Users/you/news-trading-system/scheduler/logs/cron.log 2>&1
```

---

## 7. Troubleshooting

### "ModuleNotFoundError: No module named 'anthropic'"

Your virtual environment is not active, or dependencies were not installed.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### "NEWSAPI_KEY not set" or "ANTHROPIC_API_KEY not set"

The `.env` file is missing or not in the project root.

```bash
ls -la .env           # should exist
cat .env              # should contain your keys
```

---

### "yfinance: 404 No data found for TICKER"

The ticker symbol is incorrect or the stock is no longer listed.

- For US stocks: use the standard symbol (`AAPL`, `NVDA`)
- For German stocks: append `.DE` (`BAYN.DE`, `SAP.DE`)
- For UK stocks: append `.L` (`SHEL.L`)
- Verify on [finance.yahoo.com](https://finance.yahoo.com) by searching the symbol

---

### "NewsAPI: 429 Too Many Requests"

You have hit the free-tier limit of 100 requests per day.

- Wait until the next day (resets at midnight UTC)
- Reduce your watchlist
- Upgrade to a paid NewsAPI plan

---

### "sqlite3.OperationalError: database is locked"

Another process (e.g. the dashboard) is writing to the database at the same time.

- The system has a 5-second retry built in; brief locks usually resolve automatically
- If persistent: close the dashboard or stop the scheduler before running the other

---

### "streamlit: command not found"

Streamlit is not installed or your virtual environment is not active.

```bash
source .venv/bin/activate
pip install streamlit
streamlit run dashboard/app.py
```

---

### Screener returns "0 candidates passed filters"

This is expected when markets are closed. Screener filters require:
- Price change of at least 3–4% today
- Volume at least 1.5–2× the 20-day average

These thresholds are typically only met during market hours when there is news. Run the screener between 09:30–16:00 Eastern for US stocks, or 09:00–17:30 CET for German stocks.

---

### Dashboard shows "no data" on all pages

The database has not been populated yet. Run one analysis first:

```bash
python main.py AAPL --execute
```

Then refresh the dashboard.

---

### "NotOpenSSLWarning" message at startup

This is a harmless warning from the `urllib3` library on macOS systems using LibreSSL. It does not affect functionality. You can suppress it by upgrading to a newer Python build that uses OpenSSL, or simply ignore it.
