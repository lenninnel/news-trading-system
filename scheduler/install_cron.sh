#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# install_cron.sh — install the News Trading System daily cron job
#
# What it does:
#   1. Detects Python and project paths automatically
#   2. Adds a Mon–Fri 9:30 AM cron entry that calls daily_runner.py --now
#   3. Appends stdout/stderr to scheduler/logs/cron.log
#   4. Is idempotent — safe to run more than once (won't add duplicates)
#
# Usage:
#   chmod +x scheduler/install_cron.sh
#   ./scheduler/install_cron.sh
#
# To remove the cron job later:
#   crontab -e   (delete the line containing daily_runner.py)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${PROJECT_ROOT}/scheduler/daily_runner.py"
LOG_DIR="${PROJECT_ROOT}/scheduler/logs"
LOG_FILE="${LOG_DIR}/cron.log"

mkdir -p "${LOG_DIR}"

# ── Detect Python ─────────────────────────────────────────────────────────────
# Prefer the Python in the active venv; fall back to system python3.
if command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    echo "ERROR: python3 not found on PATH. Install Python 3 and try again." >&2
    exit 1
fi

PYTHON_VER="$("${PYTHON}" --version 2>&1)"
echo "Using Python : ${PYTHON_VER} (${PYTHON})"
echo "Project root : ${PROJECT_ROOT}"
echo "Runner       : ${RUNNER}"
echo "Log file     : ${LOG_FILE}"
echo

# ── Cron expression ───────────────────────────────────────────────────────────
# 30 9 * * 1-5  →  09:30 AM, Monday through Friday (local machine time)
# IMPORTANT: if your machine is NOT in US/Eastern time, adjust the hour/minute
#            to match 9:30 AM Eastern in your local timezone.
#            e.g. UTC+1 (London winter): 14 30 * * 1-5
CRON_TIME="30 9"
CRON_DOW="1-5"
CRON_CMD="${PYTHON} ${RUNNER} --now >> ${LOG_FILE} 2>&1"
CRON_ENTRY="${CRON_TIME} * * ${CRON_DOW} ${CRON_CMD}"
CRON_MARKER="daily_runner.py"

# ── Install (idempotent) ──────────────────────────────────────────────────────
EXISTING="$(crontab -l 2>/dev/null || true)"

if echo "${EXISTING}" | grep -qF "${CRON_MARKER}"; then
    echo "Cron job already installed. No changes made."
    echo
    echo "Current entry:"
    echo "${EXISTING}" | grep "${CRON_MARKER}"
else
    # Append new entry preserving existing crontab
    ( echo "${EXISTING}"; echo "${CRON_ENTRY}" ) | crontab -
    echo "Cron job installed successfully."
    echo
    echo "Entry added:"
    echo "  ${CRON_ENTRY}"
fi

echo
echo "Verify with:  crontab -l"
echo "Test now:     ${PYTHON} ${RUNNER} --now"
echo "View logs:    tail -f ${LOG_FILE}"
