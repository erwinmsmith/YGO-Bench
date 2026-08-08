"""Privacy migrations for persisted experiment evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ygobench.agents.provider_limits import scrub_reasoning_content
from ygobench.experiments.io import JsonlJournal, atomic_write_json, content_hash


def _count_reasoning_fields(value: Any) -> int:
    if isinstance(value, dict):
        return int("reasoning_content" in value) + sum(
            _count_reasoning_fields(item) for item in value.values()
        )
    if isinstance(value, list):
        return sum(_count_reasoning_fields(item) for item in value)
    return 0


def scrub_game_reasoning(game_dir: Path) -> dict[str, Any]:
    """Remove stored reasoning and rebuild public/oracle commit links."""

    public = JsonlJournal(game_dir / "trajectory.jsonl")
    oracle = JsonlJournal(game_dir / "oracle_trajectory.jsonl")
    public_rows = public.recover()
    oracle_rows = oracle.recover()
    if len(public_rows) != len(oracle_rows):
        raise ValueError(
            f"public/oracle length mismatch in {game_dir}: "
            f"{len(public_rows)} != {len(oracle_rows)}"
        )
    removed = sum(_count_reasoning_fields(row) for row in public_rows)
    old_root = public_rows[-1].get("commit_hash") if public_rows else None
    cleaned_rows: list[dict[str, Any]] = []
    previous = ""
    for original in public_rows:
        row = scrub_reasoning_content(original)
        row.pop("commit_hash", None)
        row["previous_commit_hash"] = previous
        row["commit_hash"] = content_hash(row)
        previous = row["commit_hash"]
        cleaned_rows.append(row)
    cleaned_oracle = []
    for row, public_row in zip(oracle_rows, cleaned_rows, strict=True):
        updated = scrub_reasoning_content(row)
        updated["public_commit_hash"] = public_row["commit_hash"]
        cleaned_oracle.append(updated)
    public.rewrite(cleaned_rows)
    oracle.rewrite(cleaned_oracle)
    report = {
        "migration": "remove_reasoning_content_v1",
        "reasoning_fields_removed": removed,
        "records": len(cleaned_rows),
        "old_root_commit_hash": old_root,
        "new_root_commit_hash": previous or None,
    }
    atomic_write_json(game_dir / "privacy_migration.json", report)
    return report
