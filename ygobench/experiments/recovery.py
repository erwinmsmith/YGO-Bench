"""Durable API turns and atomic duel evidence for interrupted experiments.

SQLite is the commit authority. JSONL files remain readable projections for the
existing replay and metric pipeline; they can be rebuilt after a torn append.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ygobench.agents.provider_limits import scrub_reasoning_content
from ygobench.experiments.io import JsonlJournal, content_hash, read_json


class RecoveryConflict(RuntimeError):
    """A resumed request or decision differs from its frozen predecessor."""


def pause_status(error: Exception) -> str:
    """Classify technical API failures without charging model-action attempts."""
    if isinstance(error, RecoveryConflict):
        return "PAUSED_CONFIG"
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    if status == 402 or any(
        phrase in message for phrase in ("insufficient balance", "insufficient funds")
    ):
        return "PAUSED_BILLING"
    if status in {401, 403}:
        return "PAUSED_AUTH"
    if status == 429:
        return "PAUSED_RATE_LIMIT"
    if status == 400:
        return "PAUSED_REQUEST_CONFIG"
    return "PAUSED_NETWORK"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _turn_snapshot(turn: Any) -> dict[str, Any]:
    # Provider reasoning text and credentials are deliberately excluded. Tool
    # call IDs and arguments are sufficient to continue the game conversation.
    return {
        "text": str(getattr(turn, "text", "")),
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in getattr(turn, "tool_calls", [])
        ],
        "stop_reason": str(getattr(turn, "stop_reason", "")),
        "usage": getattr(turn, "usage", {}) or {},
        "wallclock_seconds": float(getattr(turn, "wallclock_seconds", 0.0)),
        "response_headers": {
            key: value
            for key, value in (getattr(turn, "response_headers", {}) or {}).items()
            if key.lower() in {"x-request-id", "x-oneapi-request-id", "request-id"}
        },
        "provider_data": scrub_reasoning_content(
            getattr(turn, "provider_data", {}) or {}
        ),
    }


def _restored_turn(snapshot: dict[str, Any]) -> Any:
    return SimpleNamespace(
        text=snapshot["text"],
        tool_calls=[SimpleNamespace(**call) for call in snapshot["tool_calls"]],
        stop_reason=snapshot["stop_reason"],
        usage=snapshot["usage"],
        wallclock_seconds=snapshot["wallclock_seconds"],
        response_headers=snapshot.get("response_headers", {}),
        provider_data=snapshot.get("provider_data", {}),
    )


@dataclass(frozen=True)
class CallResult:
    turn: Any
    reused: bool
    attempt: int


class RecoveryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_calls (
                    scope_id TEXT NOT NULL,
                    call_index INTEGER NOT NULL,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    response_json TEXT,
                    last_error_type TEXT,
                    PRIMARY KEY (scope_id, call_index)
                );
                CREATE TABLE IF NOT EXISTS duel_decisions (
                    game_id TEXT NOT NULL,
                    decision_index INTEGER NOT NULL,
                    public_json TEXT NOT NULL,
                    oracle_json TEXT NOT NULL,
                    PRIMARY KEY (game_id, decision_index)
                );
                """
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def call(
        self,
        *,
        scope_id: str,
        call_index: int,
        request: dict[str, Any],
        invoke: Callable[[], Any],
    ) -> CallResult:
        request_hash = content_hash(request)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM api_calls WHERE scope_id=? AND call_index=?",
                (scope_id, call_index),
            ).fetchone()
            if row is None:
                db.execute(
                    "INSERT INTO api_calls(scope_id,call_index,request_hash,status) "
                    "VALUES(?, ?, ?, 'PREPARED')",
                    (scope_id, call_index, request_hash),
                )
                attempt = 0
            else:
                if row["request_hash"] != request_hash:
                    raise RecoveryConflict(
                        f"request changed at {scope_id} call {call_index}"
                    )
                if row["status"] == "RESPONSE_SAVED":
                    return CallResult(
                        _restored_turn(json.loads(row["response_json"])),
                        True,
                        row["attempt_count"],
                    )
                attempt = int(row["attempt_count"])
            db.execute(
                "UPDATE api_calls SET status='IN_FLIGHT', attempt_count=? "
                "WHERE scope_id=? AND call_index=?",
                (attempt + 1, scope_id, call_index),
            )
        try:
            turn = invoke()
        except Exception as exc:
            with self.connect() as db:
                db.execute(
                    "UPDATE api_calls SET status='REQUEST_ERROR', last_error_type=? "
                    "WHERE scope_id=? AND call_index=? AND status='IN_FLIGHT'",
                    (type(exc).__name__, scope_id, call_index),
                )
            raise
        snapshot = _turn_snapshot(turn)
        with self.connect() as db:
            db.execute(
                "UPDATE api_calls SET status='RESPONSE_SAVED', response_json=?, "
                "last_error_type=NULL WHERE scope_id=? AND call_index=?",
                (_json(snapshot), scope_id, call_index),
            )
        return CallResult(turn, False, attempt + 1)

    def committed_decisions(self, game_id: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT public_json,oracle_json FROM duel_decisions WHERE game_id=? "
                "ORDER BY decision_index",
                (game_id,),
            ).fetchall()
        return [(json.loads(row[0]), json.loads(row[1])) for row in rows]

    def commit_decision(
        self,
        game_id: str,
        public_row: dict[str, Any],
        oracle_row: dict[str, Any],
        *,
        first_index: int = 1,
    ) -> None:
        index = int(public_row["decision_index"])
        if public_row["decision_id"] != oracle_row["decision_id"]:
            raise RecoveryConflict("public and oracle decision IDs differ")
        if public_row["commit_hash"] != oracle_row["public_commit_hash"]:
            raise RecoveryConflict("public and oracle commit hashes differ")
        unsigned_public = {
            key: value for key, value in public_row.items() if key != "commit_hash"
        }
        if content_hash(unsigned_public) != public_row["commit_hash"]:
            raise RecoveryConflict("public decision content hash is invalid")
        for stage in ("before", "after"):
            public_hash = public_row.get(f"oracle_{stage}_hash")
            oracle_hash = oracle_row.get(stage, {}).get("state_hash")
            if public_hash is not None and oracle_hash != public_hash:
                raise RecoveryConflict(f"oracle {stage} state hash differs from public record")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT decision_index,public_json FROM duel_decisions WHERE game_id=? "
                "ORDER BY decision_index DESC LIMIT 1",
                (game_id,),
            ).fetchone()
            if previous is not None and index != int(previous[0]) + 1:
                raise RecoveryConflict("decision index is not consecutive")
            if previous is None and index != first_index:
                raise RecoveryConflict("first committed decision has an unexpected index")
            previous_hash = json.loads(previous[1])["commit_hash"] if previous else ""
            if public_row.get("previous_commit_hash") != previous_hash:
                raise RecoveryConflict("decision hash chain is broken")
            db.execute(
                "INSERT INTO duel_decisions VALUES (?, ?, ?, ?)",
                (game_id, index, _json(public_row), _json(oracle_row)),
            )

    def reconcile_journals(
        self,
        game_id: str,
        public: JsonlJournal,
        oracle: JsonlJournal,
        *,
        first_index: int = 1,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Rebuild projections from the canonical transaction after a crash."""
        committed = self.committed_decisions(game_id)
        public_rows = public.recover()
        oracle_rows = oracle.recover()
        if not committed and (public_rows or oracle_rows):
            common = min(len(public_rows), len(oracle_rows))
            for index in range(common):
                if public_rows[index].get("decision_id") != oracle_rows[index].get("decision_id"):
                    break
                self.commit_decision(
                    game_id, public_rows[index], oracle_rows[index], first_index=first_index
                )
            committed = self.committed_decisions(game_id)
        canonical_public = [pair[0] for pair in committed]
        canonical_oracle = [pair[1] for pair in committed]
        if public_rows != canonical_public:
            public.rewrite(canonical_public)
        if oracle_rows != canonical_oracle:
            oracle.rewrite(canonical_oracle)
        return canonical_public, canonical_oracle


def inspect_run(run_dir: Path) -> dict[str, Any]:
    """Read-only, secret-free progress and integrity summary for a run."""
    def read_prefix(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.is_file():
            return rows
        with path.open("rb") as handle:
            for line in handle:
                try:
                    value = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    break
                if not isinstance(value, dict):
                    break
                rows.append(value)
        return rows

    tasks: list[dict[str, Any]] = []
    registry_path = run_dir / "task_state.sqlite"
    if registry_path.is_file():
        with sqlite3.connect(registry_path) as db:
            db.row_factory = sqlite3.Row
            tasks = [dict(row) for row in db.execute(
                "SELECT task_id,kind,status,attempts,lease_until,owner_host,owner_pid "
                "FROM tasks ORDER BY kind,task_id"
            )]
    api_counts: Counter[str] = Counter()
    recovery_path = run_dir / "recovery.sqlite"
    if recovery_path.is_file():
        with sqlite3.connect(recovery_path) as db:
            api_counts.update({status: count for status, count in db.execute(
                "SELECT status,COUNT(*) FROM api_calls GROUP BY status"
            )})
    games: list[dict[str, Any]] = []
    for game_dir in sorted((run_dir / "games").glob("*/manifest.json")):
        directory = game_dir.parent
        public = read_prefix(directory / "trajectory.jsonl")
        oracle = read_prefix(directory / "oracle_trajectory.jsonl")
        canonical_match: bool | None = None
        if recovery_path.is_file():
            with sqlite3.connect(recovery_path) as db:
                committed = db.execute(
                    "SELECT public_json,oracle_json FROM duel_decisions "
                    "WHERE game_id=? ORDER BY decision_index",
                    (directory.name,),
                ).fetchall()
            canonical_match = (
                public == [json.loads(pair[0]) for pair in committed]
                and oracle == [json.loads(pair[1]) for pair in committed]
            )
        outcome = read_json(directory / "outcome.json") or {}
        status = read_json(directory / "status.json") or {}
        lp = (
            oracle[-1].get("after", {}).get("tracked", {}).get("lp")
            if oracle else None
        )
        games.append({
            "game_id": directory.name,
            "status": status.get("status", "UNKNOWN"),
            "committed_decisions": min(len(public), len(oracle)),
            "projection_lengths_match": len(public) == len(oracle),
            "projection_matches_commit": canonical_match,
            "current_turn": public[-1].get("turn") if public else None,
            "current_lp": outcome.get("final_lp", lp),
            "termination": outcome.get("termination"),
        })
    return {
        "run_id": run_dir.name,
        "task_count": len(tasks),
        "task_status_counts": dict(sorted(Counter(task["status"] for task in tasks).items())),
        "unfinished_tasks": [task for task in tasks if task["status"] != "COMPLETED"],
        "api_call_status_counts": dict(sorted(api_counts.items())),
        "games": games,
    }
