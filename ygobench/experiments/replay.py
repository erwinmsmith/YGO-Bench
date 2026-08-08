"""Deterministic action-prefix replay and reversible-decision checks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ygobench.config import PROJECT_ROOT
from ygobench.engine.protocol import ActionChoice
from ygobench.experiments.io import JsonlJournal, atomic_write_json, content_hash, read_json
from ygobench.experiments.oracle import build_oracle_state
from ygobench.experiments.session import DuelSession


def verify_reversible_decisions(
    game_dir: Path,
    *,
    sample_size: int = 10,
    output: Path | None = None,
) -> dict[str, Any]:
    manifest = read_json(game_dir / "manifest.json")
    rows = JsonlJournal(game_dir / "trajectory.jsonl").recover()
    config = manifest["config"]
    if not rows:
        raise ValueError(f"no decisions in {game_dir}")
    count = min(sample_size, len(rows))
    indices = sorted(random.Random(config["seed"]).sample(range(len(rows)), count))
    deck_root = PROJECT_ROOT / "resources" / "decks"
    session = DuelSession(
        deck_root / f"{config['deck1']}.ydk",
        deck_root / f"{config['deck2']}.ydk",
        seed=int(config["seed"]),
    )
    results = []
    recent_actions: list[dict[str, Any]] = []
    try:
        for index, row in enumerate(rows[: max(indices) + 1]):
            view = session.view(recent_actions)
            before = view["oracle"]
            before_match = before["state_hash"] == row["oracle_before_hash"]
            legal_match = content_hash(view["legal"]) == content_hash(row["legal"])
            session.execute(ActionChoice(**row["executed_action"]))
            after = build_oracle_state(session)
            after_match = after["state_hash"] == row["oracle_after_hash"]
            if index in indices:
                results.append(
                    {
                        "decision_id": row["decision_id"],
                        "decision_index": index + 1,
                        "before_match": before_match,
                        "legal_match": legal_match,
                        "after_match": after_match,
                        "passed": before_match and legal_match and after_match,
                    }
                )
            if not before_match or not legal_match or not after_match:
                break
            recent_actions.append(row["action_summary"])
    finally:
        session.close()
    report = {
        "phase": 4,
        "sample_size": len(results),
        "requested_sample_size": sample_size,
        "passed": len(results) == count and all(row["passed"] for row in results),
        "results": results,
    }
    if output is not None:
        atomic_write_json(output, report)
    return report
