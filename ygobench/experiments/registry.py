"""SQLite task registry with leases and explicit terminal states."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _local_pid_alive(pid: int) -> bool:
    # Windows implements os.kill via TerminateProcess, including signal zero.
    # A Windows worker therefore waits for its short lease to expire.
    if os.name == "nt":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TaskRegistry:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
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
                    lease_generation INTEGER NOT NULL DEFAULT 0,
                    lease_until TEXT,
                    owner_host TEXT,
                    owner_pid INTEGER,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(tasks)")}
            if "lease_generation" not in columns:
                db.execute(
                    "ALTER TABLE tasks ADD COLUMN lease_generation INTEGER NOT NULL DEFAULT 0"
                )
            if "owner_host" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN owner_host TEXT")
            if "owner_pid" not in columns:
                db.execute("ALTER TABLE tasks ADD COLUMN owner_pid INTEGER")

    def add(self, task_id: str, kind: str, payload: dict[str, Any]) -> None:
        now = _now()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO tasks "
                "(task_id, kind, payload, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'PENDING', ?, ?)",
                (task_id, kind, json.dumps(payload, sort_keys=True), now, now),
            )
            existing = db.execute(
                "SELECT kind,payload FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if existing["kind"] != kind or json.loads(existing["payload"]) != payload:
                raise ValueError(f"task {task_id} has a different frozen configuration")

    def claim(self, task_id: str, lease_minutes: int = 2) -> bool:
        now = datetime.now(UTC)
        lease = (now + timedelta(minutes=lease_minutes)).isoformat()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, lease_until, owner_host, owner_pid FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None or row["status"] == "COMPLETED":
                return False
            owner_alive = False
            if row["owner_host"] == socket.gethostname() and row["owner_pid"]:
                owner_alive = _local_pid_alive(int(row["owner_pid"]))
            if (
                row["status"] == "RUNNING"
                and row["lease_until"]
                and row["lease_until"] > now.isoformat()
                and (row["owner_host"] != socket.gethostname() or owner_alive)
            ):
                return False
            updated = db.execute(
                """UPDATE tasks SET status='RUNNING', attempts=attempts+1,
                   lease_generation=lease_generation+1,
                   lease_until=?, owner_host=?, owner_pid=?, updated_at=?
                   WHERE task_id=?""",
                (lease, socket.gethostname(), os.getpid(), now.isoformat(), task_id),
            )
            return updated.rowcount == 1

    def renew(
        self, task_id: str, lease_minutes: int = 2, *, generation: int | None = None
    ) -> bool:
        """Extend an owned running lease after durable progress is recorded."""

        now = datetime.now(UTC)
        lease = (now + timedelta(minutes=lease_minutes)).isoformat()
        with self.connect() as db:
            updated = db.execute(
                """UPDATE tasks SET lease_until=?, updated_at=?
                   WHERE task_id=? AND status='RUNNING'
                   AND (? IS NULL OR lease_generation=?)""",
                (lease, now.isoformat(), task_id, generation, generation),
            )
            return updated.rowcount == 1

    def finish(
        self,
        task_id: str,
        *,
        error: str | None = None,
        status: str | None = None,
        generation: int | None = None,
    ) -> bool:
        status = status or ("FAILED_RETRYABLE" if error else "COMPLETED")
        with self.connect() as db:
            updated = db.execute(
                "UPDATE tasks SET status=?, lease_until=NULL, owner_host=NULL, "
                "owner_pid=NULL, last_error=?, "
                "updated_at=? WHERE task_id=? AND (? IS NULL OR lease_generation=?)",
                (status, error, _now(), task_id, generation, generation),
            )
            return updated.rowcount == 1

    def row(self, task_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return dict(row) if row else None


class LeaseHeartbeat:
    """Renew a worker lease even while a slow provider request is in flight."""

    def __init__(
        self, registry: TaskRegistry, task_id: str, generation: int, interval: float = 30.0
    ) -> None:
        self.registry = registry
        self.task_id = task_id
        self.generation = generation
        self.interval = interval
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if not self.registry.renew(self.task_id, generation=self.generation):
                    self._lost.set()
                    return
            except sqlite3.Error:
                self._lost.set()
                return

    def start(self) -> None:
        self._thread.start()

    def assert_owned(self) -> None:
        if self._lost.is_set() or not self.registry.renew(
            self.task_id, generation=self.generation
        ):
            raise RuntimeError(f"lease lost for task {self.task_id}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
