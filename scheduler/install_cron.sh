#!/usr/bin/env bash
# install_cron.sh — Install daily runner cron job (09:30 CET, weekdays only)
#
# Usage:
#   bash scheduler/install_cron.sh

set -euo pipefail

PROJECT_DIR="/Users/lennartwehrheim/news-trading-system"
CRON_ENTRY="30 9 * * 1-5 cd ${PROJECT_DIR} && python3 -m scheduler.daily_runner --now >> logs/daily_runner.log 2>&1"

# Avoid duplicate entries
(crontab -l 2>/dev/null | grep -v "scheduler.daily_runner" ; echo "$CRON_ENTRY") | crontab -

echo "Cron job installed successfully."
echo ""
echo "  Schedule : 09:30 CET, Monday–Friday"
echo "  Command  : python3 -m scheduler.daily_runner --now"
echo "  Log file : ${PROJECT_DIR}/logs/daily_runner.log"
echo ""
echo "Verify with: crontab -l"
