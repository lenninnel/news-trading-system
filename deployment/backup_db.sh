#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  backup_db.sh — SQLite snapshot backup
#
#  Usage:
#    ./deployment/backup_db.sh                   # backup news_trading.db
#    DB_PATH=/path/to/other.db ./deployment/backup_db.sh
#
#  Creates a timestamped copy in ./backups/ and keeps the last 30 files.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

DB_PATH="${DB_PATH:-news_trading.db}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
MAX_BACKUPS="${MAX_BACKUPS:-30}"

if [ ! -f "$DB_PATH" ]; then
  echo "ERROR: SQLite database not found: $DB_PATH"
  exit 1
fi

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
BASENAME=$(basename "$DB_PATH" .db)
DEST="${BACKUP_DIR}/${BASENAME}_${TIMESTAMP}.db"

# Use SQLite online backup API via sqlite3 CLI (safe with concurrent writes)
sqlite3 "$DB_PATH" ".backup '$DEST'"

SIZE=$(du -sh "$DEST" | cut -f1)
echo "Backup created: $DEST ($SIZE)"

# Prune old backups
COUNT=$(ls -1 "${BACKUP_DIR}/${BASENAME}_"*.db 2>/dev/null | wc -l)
if [ "$COUNT" -gt "$MAX_BACKUPS" ]; then
  TO_DELETE=$(( COUNT - MAX_BACKUPS ))
  ls -1t "${BACKUP_DIR}/${BASENAME}_"*.db | tail -n "$TO_DELETE" | xargs rm -f
  echo "Pruned $TO_DELETE old backup(s) (keeping last $MAX_BACKUPS)."
fi
