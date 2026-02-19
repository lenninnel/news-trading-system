"""
State Recovery — checkpoint system for crash-resistant long-running operations.

CheckpointManager saves operation state to a JSON file every N operations
(default: 10) so that a crashed or interrupted run can resume from where it
left off rather than starting over.

State integrity validation
--------------------------
On load, every checkpoint is validated against three rules:
  1. Schema version matches (prevents loading stale format).
  2. _saved_at is a valid ISO-8601 timestamp.
  3. Age does not exceed max_age_hours (default: 24 h).

Usage
-----
    from utils.state_recovery import CheckpointManager

    mgr = CheckpointManager("daily_run")
    mgr.set_db(db)  # optional — enables DB logging

    # Resume or start fresh
    pending_tickers = mgr.get_pending(all_tickers)
    completed = set(all_tickers) - set(pending_tickers)

    for ticker in pending_tickers:
        process(ticker)
        completed.add(ticker)
        mgr.update("completed_tickers", list(completed))

    mgr.clear()  # remove checkpoint after successful completion
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Checkpoint files live in <project_root>/.checkpoints/
_DEFAULT_DIR    = Path(__file__).resolve().parent.parent / ".checkpoints"
_SCHEMA_VERSION = 1


class CheckpointManager:
    """
    Saves and restores operation state for crash recovery.

    Auto-saves to disk every *save_interval* update() calls.  State is
    stored as JSON so it survives process restarts.

    Args:
        name:           Human-readable run name; becomes the filename stem.
        checkpoint_dir: Directory for checkpoint files (created if absent).
        save_interval:  Auto-save after every N update() calls.
        max_age_hours:  Reject checkpoints older than this on load.
    """

    def __init__(
        self,
        name:           str              = "run",
        checkpoint_dir: "Path | str | None" = None,
        save_interval:  int              = 10,
        max_age_hours:  float            = 24.0,
    ) -> None:
        self.name        = name
        self._dir        = Path(checkpoint_dir or _DEFAULT_DIR)
        self._save_every = save_interval
        self._max_age_h  = max_age_hours
        self._op_count   = 0
        self._state:     dict = {}
        self._lock       = threading.Lock()
        self._db:        Any  = None
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Absolute path to the checkpoint JSON file."""
        return self._dir / f"{self.name}.json"

    def set_db(self, db: Any) -> None:
        """Attach a Database instance for recovery_log persistence."""
        self._db = db

    # -- Writes ----------------------------------------------------------------

    def update(self, key: str, value: Any) -> None:
        """
        Update an in-memory state field and auto-save every *save_interval* calls.

        Args:
            key:   State field name (e.g. "completed_tickers").
            value: JSON-serialisable value.
        """
        with self._lock:
            self._state[key]  = value
            self._op_count   += 1
            if self._op_count % self._save_every == 0:
                self._save_unlocked()
                log.debug(
                    "Checkpoint '%s': auto-saved at %d ops",
                    self.name, self._op_count,
                )

    def save(self) -> None:
        """Explicitly flush current state to disk immediately."""
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        """Write state to disk; must be called while holding self._lock."""
        payload = {
            "_version":  _SCHEMA_VERSION,
            "_name":     self.name,
            "_saved_at": datetime.now(timezone.utc).isoformat(),
            "_op_count": self._op_count,
            **self._state,
        }
        try:
            # Write to a temp file first, then atomic rename
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            tmp.replace(self.path)
            self._log_db(
                "checkpoint_save",
                f"Checkpoint saved ({self._op_count} ops completed)",
                success=True,
            )
        except OSError as exc:
            log.error("Checkpoint '%s': save failed: %s", self.name, exc)

    # -- Reads -----------------------------------------------------------------

    def load(self) -> "dict | None":
        """
        Load the last checkpoint, validating its integrity.

        Returns:
            dict of application state (internal ``_*`` keys stripped), or
            None when the file is absent, stale, or corrupt.
        """
        if not self.path.exists():
            return None

        try:
            raw: dict = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Checkpoint '%s': could not read file: %s", self.name, exc)
            return None

        if not self._validate(raw):
            return None

        # Strip internal metadata before returning to caller
        state = {k: v for k, v in raw.items() if not k.startswith("_")}
        log.info(
            "Checkpoint '%s': resuming from save at %s (%d ops previously done)",
            self.name, raw.get("_saved_at", "?"), raw.get("_op_count", 0),
        )
        self._log_db(
            "checkpoint_resume",
            f"Resumed from checkpoint saved at {raw.get('_saved_at', '?')}",
            success=True,
        )
        return state

    def _validate(self, raw: dict) -> bool:
        """Return True when *raw* passes all integrity checks."""
        # 1. Schema version
        if raw.get("_version") != _SCHEMA_VERSION:
            log.warning(
                "Checkpoint '%s': version mismatch (got %s, expected %s) — discarding",
                self.name, raw.get("_version"), _SCHEMA_VERSION,
            )
            return False

        # 2. Valid timestamp
        saved_at_str = raw.get("_saved_at", "")
        try:
            saved_at = datetime.fromisoformat(saved_at_str)
        except (ValueError, TypeError):
            log.warning(
                "Checkpoint '%s': invalid _saved_at '%s' — discarding",
                self.name, saved_at_str,
            )
            return False

        # Make offset-aware for arithmetic
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=timezone.utc)

        # 3. Not too old
        age_h = (datetime.now(timezone.utc) - saved_at).total_seconds() / 3600.0
        if age_h > self._max_age_h:
            log.info(
                "Checkpoint '%s': stale (%.1f h old, limit %.1f h) — discarding",
                self.name, age_h, self._max_age_h,
            )
            return False

        return True

    # -- Resume helper ---------------------------------------------------------

    def get_pending(
        self,
        all_items:     list[str],
        completed_key: str = "completed_tickers",
    ) -> list[str]:
        """
        Return items from *all_items* not yet recorded in the checkpoint.

        If no valid checkpoint exists, returns *all_items* unchanged (full run).

        Args:
            all_items:     Full ordered list to process (e.g. watchlist tickers).
            completed_key: Key in checkpoint state that holds completed items.

        Returns:
            Subset of all_items that still need processing.
        """
        state = self.load()
        if state is None:
            return all_items

        done    = set(state.get(completed_key, []))
        pending = [t for t in all_items if t not in done]

        if pending and done:
            log.info(
                "Checkpoint '%s': %d/%d items already done, %d remaining",
                self.name, len(done), len(all_items), len(pending),
            )
        return pending

    # -- Lifecycle -------------------------------------------------------------

    def clear(self) -> None:
        """Delete the checkpoint file (call after successful completion)."""
        with self._lock:
            if self.path.exists():
                self.path.unlink()
                log.info("Checkpoint '%s': cleared after successful completion", self.name)

    # -- DB logging ------------------------------------------------------------

    def _log_db(self, event_type: str, msg: str, success: bool = True) -> None:
        if self._db is None:
            return
        try:
            self._db.log_recovery_event(
                service="checkpoint",
                event_type=event_type,
                error_msg=None if success else msg,
                recovery_action=msg if success else None,
                success=success,
            )
        except Exception:
            pass
