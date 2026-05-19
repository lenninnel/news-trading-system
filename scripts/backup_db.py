#!/usr/bin/env python3
"""Daily SQLite backup for news_trading.db.

Uses sqlite3.connect().backup() (online backup API) — safe during
concurrent writes from the trading daemon. Writes a dated file to
/home/trading/backups/ and prunes anything older than RETENTION_DAYS.

Designed to run as a systemctl --user timer. Logs to stdout/stderr
which systemd captures to journalctl --user -u nts-backup.

Exit codes:
    0  backup succeeded (and prune succeeded)
    1  source DB missing or backup failed
    2  prune failed but backup succeeded (still considered failure
       so it's visible; backup file is preserved)
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SRC_PATH = Path(
    os.environ.get(
        "NTS_BACKUP_SRC",
        "/home/trading/trading-data/news_trading.db",
    )
)
DST_DIR = Path(
    os.environ.get(
        "NTS_BACKUP_DST_DIR",
        "/home/trading/backups",
    )
)
RETENTION_DAYS = int(os.environ.get("NTS_BACKUP_RETENTION_DAYS", "14"))
MIN_BYTES = int(os.environ.get("NTS_BACKUP_MIN_BYTES", "1000000"))

# R2 off-host replication config. rclone is installed user-local at
# ~/bin/rclone (no sudo); reference via explicit path so systemd
# --user context doesn't depend on PATH inheritance.
RCLONE_BIN = os.environ.get(
    "NTS_RCLONE_BIN",
    str(Path.home() / "bin" / "rclone"),
)
R2_ENABLED = os.environ.get("NTS_BACKUP_R2_ENABLED", "true").lower() == "true"
R2_REMOTE = os.environ.get("NTS_BACKUP_R2_REMOTE", "r2:nts-backup/daily")
R2_RETENTION_DAYS = int(os.environ.get("NTS_BACKUP_R2_RETENTION_DAYS", "14"))


def _log(msg: str) -> None:
    """Single-line timestamped log — journalctl-friendly."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{ts} {msg}", flush=True)


def main() -> int:
    if not SRC_PATH.exists():
        _log(f"ERROR source DB missing: {SRC_PATH}")
        return 1
    if not SRC_PATH.is_file():
        _log(f"ERROR source path is not a file: {SRC_PATH}")
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)

    today = date.today().isoformat()
    dst_path = DST_DIR / f"news_trading_{today}.db"

    _log(f"backup start src={SRC_PATH} dst={dst_path}")

    try:
        src = sqlite3.connect(str(SRC_PATH))
        try:
            dst = sqlite3.connect(str(dst_path))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except Exception as exc:
        _log(f"ERROR backup failed: {exc!r}")
        # Don't leave a half-written file behind
        try:
            dst_path.unlink(missing_ok=True)
        except Exception:
            pass
        return 1

    # Verify the backup is non-trivial
    try:
        size = dst_path.stat().st_size
    except FileNotFoundError:
        _log("ERROR backup file vanished after write")
        return 1

    if size < MIN_BYTES:
        _log(
            f"ERROR backup too small ({size} bytes < {MIN_BYTES}) — "
            f"refusing to keep"
        )
        try:
            dst_path.unlink(missing_ok=True)
        except Exception:
            pass
        return 1

    _log(f"backup ok size={size} bytes")

    # ── R2 replication (fail-soft) ──────────────────────────────
    # R2 push failures log WARN but never fail the backup. Local
    # integrity is the primary success criterion; off-host is
    # secondary disaster-recovery insurance.
    #
    # --s3-no-check-bucket: bucket-scoped tokens deny CreateBucket
    # (which rclone calls by default before copy). The bucket exists;
    # skip the check.
    if R2_ENABLED:
        import subprocess
        _log(f"R2 push start: {dst_path.name} -> {R2_REMOTE}/")
        try:
            result = subprocess.run(
                [RCLONE_BIN, "copy", str(dst_path), R2_REMOTE + "/",
                 "--s3-no-check-bucket",
                 "--retries", "2", "--low-level-retries", "2",
                 "--timeout", "60s"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                _log(f"WARN R2 push failed (rc={result.returncode}): "
                     f"{result.stderr.strip()[:300]}")
            else:
                _log("R2 push ok")
                prune_result = subprocess.run(
                    [RCLONE_BIN, "delete", R2_REMOTE + "/",
                     "--s3-no-check-bucket",
                     "--min-age", f"{R2_RETENTION_DAYS}d",
                     "--include", "news_trading_*.db"],
                    capture_output=True, text=True, timeout=60,
                )
                if prune_result.returncode != 0:
                    _log(f"WARN R2 prune failed: "
                         f"{prune_result.stderr.strip()[:200]}")
                else:
                    _log("R2 prune ok")
        except subprocess.TimeoutExpired:
            _log("WARN R2 push timed out after 120s")
        except FileNotFoundError:
            _log(f"WARN R2 push skipped: rclone not found at {RCLONE_BIN}")
        except Exception as exc:
            _log(f"WARN R2 push exception: {exc!r}")
    else:
        _log("R2 push disabled (NTS_BACKUP_R2_ENABLED=false)")

    # Prune older backups
    cutoff = date.today() - timedelta(days=RETENTION_DAYS)
    pruned = 0
    prune_errors = 0
    try:
        for f in DST_DIR.glob("news_trading_*.db"):
            # Parse date from filename
            stem = f.stem.replace("news_trading_", "")
            try:
                f_date = date.fromisoformat(stem)
            except ValueError:
                # Don't touch files that don't match our naming
                continue
            if f_date < cutoff:
                try:
                    f.unlink()
                    pruned += 1
                    _log(f"pruned {f.name}")
                except Exception as exc:
                    prune_errors += 1
                    _log(f"WARN failed to prune {f.name}: {exc!r}")
    except Exception as exc:
        _log(f"ERROR prune scan failed: {exc!r}")
        return 2

    _log(
        f"backup complete: 1 new, {pruned} pruned, "
        f"{prune_errors} prune errors"
    )

    if prune_errors > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
