"""Frozen configuration and deterministic identifiers for experiment runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(prefix: str, value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode()).hexdigest()[:length]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    game_id: str
    deck1: str
    deck2: str
    agent1: str
    agent2: str
    seed: int
    max_decisions: int = 0
    # v1.1 permits a terminal decision record with no executed engine action
    # when model-action retries are exhausted and the acting player forfeits.
    schema_version: str = "1.1.0"
    checkpoint_interval: int = 25
    resume: bool = True

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        deck1: str,
        deck2: str,
        agent1: str,
        agent2: str,
        seed: int,
        max_decisions: int = 0,
        game_id: str | None = None,
        checkpoint_interval: int = 25,
        resume: bool = True,
    ) -> ExperimentConfig:
        identity = {
            "run_id": run_id,
            "deck1": deck1,
            "deck2": deck2,
            "agent1": agent1,
            "agent2": agent2,
            "seed": seed,
        }
        return cls(
            run_id=run_id,
            game_id=game_id or stable_id("game", identity),
            deck1=deck1,
            deck2=deck2,
            agent1=agent1,
            agent2=agent2,
            seed=seed,
            max_decisions=max_decisions,
            checkpoint_interval=checkpoint_interval,
            resume=resume,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def game_dir(self, root: Path) -> Path:
        return root / self.run_id / "games" / self.game_id
