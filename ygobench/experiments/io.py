"""Crash-safe JSON/JSONL primitives and content hashes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ygobench.experiments.config import canonical_json


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


class JsonlJournal:
    """Append-only JSONL with recovery to the last valid newline/record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    def recover(self) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        valid_bytes = 0
        with self.path.open("rb") as handle:
            for raw in handle:
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    break
                if not isinstance(payload, dict):
                    break
                valid.append(payload)
                valid_bytes += len(raw)
        if valid_bytes != self.path.stat().st_size:
            with self.path.open("r+b") as handle:
                handle.truncate(valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
        return valid

    def append(self, payload: dict[str, Any]) -> None:
        line = canonical_json(payload) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def rewrite(self, records: list[dict[str, Any]]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
