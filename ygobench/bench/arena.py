"""Round-robin full-duel benchmark runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations, product
from pathlib import Path

from ygobench.agents.factory import create_agent
from ygobench.bench.duel_metrics import summarize_duels
from ygobench.config import PROJECT_ROOT
from ygobench.engine.full_duel import run_duel


@dataclass(frozen=True)
class ArenaConfig:
    agents: tuple[str, ...]
    decks: tuple[str, ...]
    seeds: tuple[int, ...]
    max_decisions: int = 2000
    run_name: str | None = None


def run_arena(config: ArenaConfig) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = config.run_name or f"{stamp}_full_duel_arena"
    run_dir = PROJECT_ROOT / "bench_data" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    deck_root = PROJECT_ROOT / "resources" / "decks"
    missing = [deck for deck in config.decks if not (deck_root / f"{deck}.ydk").is_file()]
    if missing:
        raise ValueError(f"Missing deck files: {', '.join(missing)}")

    pairs = list(combinations(config.agents, 2))
    if not pairs and config.agents:
        pairs = [(config.agents[0], config.agents[0])]
    games = []
    game_id = 0
    for agent_a, agent_b in pairs:
        for deck_a, deck_b in product(config.decks, repeat=2):
            for seed in config.seeds:
                for swapped in (False, True):
                    p1, p2 = (agent_b, agent_a) if swapped else (agent_a, agent_b)
                    d1, d2 = (deck_b, deck_a) if swapped else (deck_a, deck_b)
                    replay_name = f"game_{game_id:04d}_seed_{seed}_{d1}_vs_{d2}.jsonl"
                    result = run_duel(
                        deck_root / f"{d1}.ydk",
                        deck_root / f"{d2}.ydk",
                        agent1=create_agent(p1, seed=seed * 2),
                        agent2=create_agent(p2, seed=seed * 2 + 1),
                        seed=seed,
                        max_decisions=config.max_decisions,
                        output_dir=run_dir,
                        replay_name=replay_name,
                    )
                    game = result.to_dict()
                    game.update(
                        {
                            "game_id": game_id,
                            "seed": seed,
                            "agents": [p1, p2],
                            "decks": [d1, d2],
                            "decisions_by_player": _outcome_field(
                                result.replay_path, "decisions_by_player", [0, 0]
                            ),
                            "model_usage_totals": _outcome_field(
                                result.replay_path, "model_usage_totals", {}
                            ),
                        }
                    )
                    games.append(game)
                    game_id += 1

    metrics = summarize_duels(games)
    (run_dir / "games.json").write_text(json.dumps(games, indent=2) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    summary = {
        "benchmark_type": "full_duel_arena",
        "config": {
            "agents": list(config.agents),
            "decks": list(config.decks),
            "seeds": list(config.seeds),
            "max_decisions": config.max_decisions,
        },
        "counts": {"games": len(games), "engine_completed": metrics["engine_completed"]},
        "per_instance": {f"game_{game['game_id']:04d}": game for game in games},
    }
    (run_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir


def _outcome_field(path: Path, key: str, default):
    with path.open() as handle:
        for raw in reversed(handle.readlines()):
            payload = json.loads(raw)
            if payload.get("type") == "outcome":
                return payload.get(key, default)
    return default
