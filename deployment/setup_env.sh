#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  setup_env.sh — Create a .env file from the template
#
#  Usage:
#    ./deployment/setup_env.sh
#
#  Copies .env.template → .env if .env does not already exist, then prompts
#  for the three required values that have no defaults.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

TEMPLATE=".env.template"
ENV_FILE=".env"

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: $TEMPLATE not found. Run this script from the project root."
  exit 1
fi

if [ -f "$ENV_FILE" ]; then
  echo ".env already exists — skipping copy."
else
  cp "$TEMPLATE" "$ENV_FILE"
  echo "Created $ENV_FILE from template."
fi

echo ""
echo "Please edit $ENV_FILE and set the following required values:"
echo ""
echo "  ANTHROPIC_API_KEY  — from https://console.anthropic.com/"
echo "  NEWSAPI_KEY        — from https://newsapi.org/"
echo "  DATABASE_URL       — PostgreSQL DSN (leave blank for local SQLite)"
echo ""
echo "Optional:"
echo "  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  — for trade alerts"
echo ""
echo "When done, run:  docker-compose up"
