#!/usr/bin/env python3
"""
Standalone entry point for the signal outcome tracker (cron safety net).

Runs nightly at 23:00 UTC from the ``trading`` user's crontab on the VPS
so 3d/5d/10d outcome columns on ``signal_events`` get filled even when the
EOD session (22:45 UTC, which runs the same tracker) did not complete.
Both runs are idempotent and write from the same source: the
``daily_ohlc`` store (see ``analytics/outcome_tracker.py``).

History: until 2026-09-23 this script carried three yfinance-based fixups
of its own (sealing sentinel rows with price=0, backfilling missing entry
prices from *adjusted* Yahoo closes, recomputing percentages).  All three
now live inside the tracker, on the store's raw closes, with an explicit
``outcome_status`` / ``outcome_note`` instead of a 0.0 sentinel value.

Usage::

    python3 scripts/update_outcomes.py            # one-shot backfill
    python3 scripts/update_outcomes.py --dry-run  # report only
    python3 scripts/update_outcomes.py --quiet    # summary line only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when invoked as a bare script
# (e.g. from cron) rather than via ``python -m``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics.outcome_tracker import run_outcome_tracker  # noqa: E402

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fill 3d/5d/10d price outcomes on signal_events from daily_ohlc.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Backfill all pending outcomes (default behaviour — flag kept "
             "for CLI clarity / the existing cron line).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change; write nothing.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logs; only print the summary line.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from storage.database import Database
    db = Database()

    try:
        result = run_outcome_tracker(db, dry_run=args.dry_run)
    except Exception as exc:
        # Non-zero exit so cron mail / the log shows the failure instead
        # of a silent empty run.
        log.error("update_outcomes FAILED: %s", exc, exc_info=True)
        print(f"update_outcomes: FAILED {exc}")
        return 1

    print(f"update_outcomes: {result.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
