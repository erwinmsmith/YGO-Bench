"""Metric artifact provenance without credentials or mutable absolute paths."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _dirty(root: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def metric_provenance(run_dir: Path, *, statistics_seed: int = 0) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    inputs = {}
    patterns = (
        "games/*/manifest.json",
        "games/*/outcome.json",
        "games/*/trajectory.jsonl",
        "games/*/oracle_trajectory.jsonl",
        "derived/*.jsonl",
        "derived/probe_results/*/*.jsonl",
        "games/*/exp6_offline_report.json",
    )
    for pattern in patterns:
        for path in sorted(run_dir.glob(pattern)):
            inputs[path.relative_to(run_dir).as_posix()] = _sha256(path)
    source_hashes = {}
    for pattern in (
        "ygobench/experiments/*.py",
        "ygobench/agents/*.py",
        "ygobench/agents/prompts/*.md",
        "ygobench/engine/*.py",
        "ygobench/bench/glicko2.py",
        "ygobench/config.py",
    ):
        for path in sorted(project_root.glob(pattern)):
            source_hashes[path.relative_to(project_root).as_posix()] = _sha256(path)
    return {
        "code_commit": _commit(project_root),
        "tracked_worktree_dirty": _dirty(project_root),
        "analysis_source_hashes": source_hashes,
        "input_artifact_hashes": inputs,
        "statistics_seed": statistics_seed,
        "contains_credentials": False,
    }
