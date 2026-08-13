"""SQLite task registry with leases and explicit terminal states."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TaskRegistry:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    lease_until TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def add(self, task_id: str, kind: str, payload: dict[str, Any]) -> None:
        now = _now()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO tasks VALUES (?, ?, ?, 'PENDING', 0, NULL, NULL, ?, ?)",
                (task_id, kind, json.dumps(payload, sort_keys=True), now, now),
            )

    def claim(self, task_id: str, lease_minutes: int = 30) -> bool:
        now = datetime.now(UTC)
        lease = (now + timedelta(minutes=lease_minutes)).isoformat()
        with self.connect() as db:
            row = db.execute(
                "SELECT status, lease_until FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None or row["status"] == "COMPLETED":
                return False
            if (
                row["status"] == "RUNNING"
                and row["lease_until"]
                and row["lease_until"] > now.isoformat()
            ):
                return False
            updated = db.execute(
                """UPDATE tasks SET status='RUNNING', attempts=attempts+1,
                   lease_until=?, updated_at=? WHERE task_id=?""",
                (lease, now.isoformat(), task_id),
            )
            return updated.rowcount == 1

    def renew(self, task_id: str, lease_minutes: int = 30) -> bool:
        """Extend an owned running lease after durable progress is recorded."""

        now = datetime.now(UTC)
        lease = (now + timedelta(minutes=lease_minutes)).isoformat()
        with self.connect() as db:
            updated = db.execute(
                """UPDATE tasks SET lease_until=?, updated_at=?
                   WHERE task_id=? AND status='RUNNING'""",
                (lease, now.isoformat(), task_id),
            )
            return updated.rowcount == 1

    def finish(self, task_id: str, *, error: str | None = None) -> None:
        status = "FAILED_RETRYABLE" if error else "COMPLETED"
        with self.connect() as db:
            db.execute(
                "UPDATE tasks SET status=?, lease_until=NULL, last_error=?, "
                "updated_at=? WHERE task_id=?",
                (status, error, _now(), task_id),
            )

    def row(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return dict(row) if row else None
