#!/usr/bin/env python3
"""Export experiment evidence into the JSONL replay format used by the web UI.

The experiment directory remains the source of truth. This exporter creates a
derived, public-only replay projection under bench_data/runs. It never reads
oracle state into the generated replay stream.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ygobench.config import PROJECT_ROOT

EXPERIMENTS_ROOT = PROJECT_ROOT / "bench_data" / "experiments"
RUNS_ROOT = PROJECT_ROOT / "bench_data" / "runs"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                break
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _atomic_write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _agent_configs(manifest: dict[str, Any], config: dict[str, Any]) -> list[Any]:
    agents = manifest.get("agents")
    if isinstance(agents, list) and agents:
        return agents
    return [
        {"name": config.get("agent1", "unknown")},
        {"name": config.get("agent2", "unknown")},
    ]


def _tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    trace = row.get("trace")
    turns = trace.get("turns", []) if isinstance(trace, dict) else []
    for turn in turns:
        for call in turn.get("tool_calls", []) if isinstance(turn, dict) else []:
            if isinstance(call, dict):
                calls.append(
                    {
                        key: call[key]
                        for key in ("id", "name", "arguments")
                        if key in call
                    }
                )
    if not calls:
        action = row.get("attempted_action") or row.get("executed_action")
        if isinstance(action, dict) and action.get("tool"):
            calls.append(
                {
                    "name": action["tool"],
                    "arguments": action.get("arguments", {}),
                }
            )
    return calls


def _model_turn(row: dict[str, Any], agents: list[Any]) -> dict[str, Any]:
    player = int(row.get("player", 0))
    validation = row.get("validation") or {}
    trace = row.get("trace") or {}
    turns = trace.get("turns", []) if isinstance(trace, dict) else []
    text = ""
    if turns and isinstance(turns[-1], dict):
        text = turns[-1].get("text", "") or ""
    attempted = row.get("attempted_action") or {}
    if not text:
        text = attempted.get("label", "") or attempted.get("tool", "")
    agent = agents[player] if player < len(agents) else "unknown"
    if isinstance(agent, dict):
        agent = agent.get("name") or agent.get("id") or agent.get("model") or "unknown"
    return {
        "type": "model_turn",
        "player": player,
        "agent": agent,
        "text": text,
        "tool_calls": _tool_calls(row),
        "elapsed_seconds": row.get("elapsed_seconds", 0),
        "agent_error": (
            validation.get("agent_error")
            or validation.get("pre_engine_error")
            or validation.get("engine_error")
        ),
        "trace": trace,
    }


def _invalid_action(row: dict[str, Any]) -> dict[str, Any] | None:
    validation = row.get("validation") or {}
    if validation.get("valid", True) and not validation.get("attempted_invalid"):
        return None
    return {
        "type": "invalid_action",
        "player": row.get("player"),
        "attempted": row.get("attempted_action"),
        "executed": row.get("executed_action"),
        "error": (
            validation.get("agent_error")
            or validation.get("pre_engine_error")
            or validation.get("engine_error")
            or "invalid_action"
        ),
        "resolution": validation.get("recovery", "none"),
    }


def _config_event(
    manifest: dict[str, Any],
    config: dict[str, Any],
    game_dir: Path,
) -> dict[str, Any]:
    randomization = manifest.get("randomization") or {}
    return {
        "type": "config",
        "schema_version": manifest.get("schema_version", config.get("schema_version")),
        "run_id": config.get("run_id"),
        "game_id": config.get("game_id"),
        "seed": config.get("seed"),
        "deck1": config.get("deck1"),
        "deck2": config.get("deck2"),
        "agents": [config.get("agent1"), config.get("agent2")],
        "agent_configs": _agent_configs(manifest, config),
        "rules": "MR5",
        "starting_lp": 8000,
        "starting_hand": 5,
        "draw_per_turn": 1,
        "prompt_template": "full_duel_system.md",
        "randomization": {
            key: randomization[key]
            for key in (
                "engine_seed",
                "engine_seeds",
                "deck_shuffle_seeds",
                "deck_order_hashes",
                "deck_shuffle_algorithm",
                "llm_sampling",
            )
            if key in randomization
        },
        "source_experiment_dir": str(game_dir),
        "public_only": True,
    }


def export_game(game_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = _read_json(game_dir / "manifest.json")
    config = manifest.get("config") or {}
    trajectory = _read_jsonl(game_dir / "trajectory.jsonl")
    outcome = _read_json(game_dir / "outcome.json")
    if not manifest or not config:
        raise ValueError(f"missing manifest/config: {game_dir}")
    if not trajectory:
        raise ValueError(f"missing trajectory: {game_dir}")

    agents = [config.get("agent1"), config.get("agent2")]
    events: list[dict[str, Any]] = [
        _config_event(manifest, config, game_dir),
        {
            "type": "start",
            "run_id": config.get("run_id"),
            "game_id": config.get("game_id"),
            "events": trajectory[0].get("observation", {}).get(
                "events_since_last_decision", []
            ),
        },
    ]
    for row in trajectory:
        events.append(
            {
                "type": "observation",
                "run_id": row.get("run_id"),
                "game_id": row.get("game_id"),
                "decision": row.get("decision_index"),
                "player": row.get("player"),
                "state": row.get("observation", {}),
            }
        )
        events.append(_model_turn(row, agents))
        if row.get("engine_events"):
            action = row.get("executed_action") or {}
            events.append(
                {
                    "type": "tool_result",
                    "player": row.get("player"),
                    "tool": action.get("tool"),
                    "events": row.get("engine_events", []),
                    "fallback": (row.get("validation") or {}).get("recovery")
                    in {
                        "deterministic_fallback_after_engine_error",
                        "corrected_retry",
                    },
                }
            )
        invalid = _invalid_action(row)
        if invalid is not None:
            events.append(invalid)

    if outcome:
        replay_outcome = {
            "type": "outcome",
            "benchmark_type": "full_duel",
            **outcome,
            "winner_agent": (
                agents[int(outcome["winner"])]
                if isinstance(outcome.get("winner"), int)
                and 0 <= int(outcome["winner"]) < len(agents)
                else None
            ),
        }
    else:
        replay_outcome = {
            "type": "outcome",
            "benchmark_type": "full_duel",
            "termination": "in_progress",
            "game_over": False,
            "winner": None,
        }
    events.append(replay_outcome)

    filename = f"{config['game_id']}.jsonl"
    output_path = output_dir / filename
    _atomic_write_jsonl(output_path, events)
    return {
        "run_id": config.get("run_id"),
        "game_id": config.get("game_id"),
        "source": str(game_dir),
        "output": str(output_path),
        "decisions": len(trajectory),
        "events": len(events),
        "completed": bool(outcome.get("game_over")) if outcome else False,
    }


def export_run(run_dir: Path, output_root: Path = RUNS_ROOT) -> dict[str, Any]:
    output_dir = output_root / run_dir.name
    game_dirs = sorted(
        path.parent for path in (run_dir / "games").glob("*/manifest.json")
    )
    if not game_dirs:
        raise ValueError(f"no game manifests found: {run_dir}")
    exported = [export_game(game_dir, output_dir) for game_dir in game_dirs]
    _atomic_write_json(
        output_dir / "_summary.json",
        {
            "run_id": run_dir.name,
            "source": str(run_dir),
            "exported_at": datetime.now(UTC).isoformat(),
            "counts": {
                "games": len(exported),
                "completed": sum(item["completed"] for item in exported),
                "decisions": sum(item["decisions"] for item in exported),
            },
            "games": exported,
        },
    )
    return {
        "run_id": run_dir.name,
        "output_dir": str(output_dir),
        "games": exported,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export experiment trajectories into web replay JSONL files."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--all-runs", action="store_true")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=EXPERIMENTS_ROOT,
        help="Override the experiment evidence root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUNS_ROOT,
        help="Override the derived web replay root.",
    )
    args = parser.parse_args()
    if args.run_id:
        run_dirs = [args.experiments_root / args.run_id]
    else:
        run_dirs = sorted(
            path for path in args.experiments_root.iterdir() if (path / "games").is_dir()
        )
    results = [export_run(run_dir, args.output_root) for run_dir in run_dirs]
    print(json.dumps({"exports": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
