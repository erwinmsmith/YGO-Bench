"""Read benchmark runs and append-only JSONL episode replays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ygobench.config import PROJECT_ROOT

RUNS_ROOT = PROJECT_ROOT / "bench_data" / "runs"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def list_runs() -> list[dict[str, Any]]:
    if not RUNS_ROOT.is_dir():
        return []
    runs = []
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        summary = _read_json(run_dir / "_summary.json")
        metrics = _read_json(run_dir / "metrics.json")
        replay_count = sum(1 for path in run_dir.glob("*.jsonl") if not path.name.startswith("_"))
        runs.append(
            {
                "id": run_dir.name,
                "mtime": run_dir.stat().st_mtime,
                "replay_count": replay_count,
                "metrics": metrics,
                "counts": summary.get("counts", {}),
            }
        )
    return sorted(runs, key=lambda item: item["mtime"], reverse=True)


def list_replays() -> list[dict[str, Any]]:
    replays = []
    if not RUNS_ROOT.is_dir():
        return replays
    for run_dir in RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        for path in run_dir.glob("*.jsonl"):
            if path.name.startswith("_"):
                continue
            replays.append(
                {
                    "run_id": run_dir.name,
                    "filename": path.name,
                    "puzzle_id": path.stem,
                    "size": path.stat().st_size,
                    "mtime": path.stat().st_mtime,
                }
            )
    return sorted(replays, key=lambda item: item["mtime"], reverse=True)


def _safe_replay_path(run_id: str, filename: str) -> Path:
    candidate = (RUNS_ROOT / run_id / filename).resolve()
    root = RUNS_ROOT.resolve()
    if candidate.suffix != ".jsonl" or root not in candidate.parents:
        raise ValueError("Invalid replay path")
    return candidate


def _compact_card(card: Any) -> Any:
    if not isinstance(card, dict):
        return card
    keep = (
        "zone_index",
        "empty",
        "face_down",
        "code",
        "name",
        "position",
        "attack",
        "defense",
        "level",
        "rank",
        "link",
        "attribute",
        "race",
        "type_flags",
        "location",
        "sequence",
    )
    return {key: card[key] for key in keep if key in card}


def _compact_side(side: Any) -> Any:
    if not isinstance(side, dict):
        return side
    compact = {
        key: side.get(key)
        for key in (
            "lp",
            "deck_count",
            "hand_count",
            "grave_count",
            "banished_count",
            "extra_deck_count",
        )
    }
    for key in (
        "monster_zone",
        "spell_trap_zone",
        "hand",
        "graveyard",
        "banished",
        "pendulum_zone",
    ):
        compact[key] = [_compact_card(card) for card in side.get(key, [])]
    compact["field_zone"] = _compact_card(side.get("field_zone"))
    return compact


def _compact_event(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("type") != "observation" or not isinstance(payload.get("state"), dict):
        return payload
    state = payload["state"]
    payload["state"] = {
        key: state.get(key)
        for key in (
            "perspective_player",
            "phase",
            "turn_player",
            "turn",
            "game_over",
            "decision",
            "chain",
            "events_since_last_decision",
            "recent_actions",
        )
    }
    payload["state"]["you"] = _compact_side(state.get("you"))
    payload["state"]["opponent"] = _compact_side(state.get("opponent"))
    return payload


def load_replay(run_id: str, filename: str, *, compact: bool = True) -> dict[str, Any] | None:
    try:
        path = _safe_replay_path(run_id, filename)
    except ValueError:
        return None
    if not path.is_file():
        return None
    events = []
    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                payload = {"type": "parse_error", "line": line_number}
            if compact:
                payload = _compact_event(payload)
            payload["index"] = len(events)
            events.append(payload)
    outcome = next((event for event in reversed(events) if event.get("type") == "outcome"), None)
    return {"run_id": run_id, "filename": filename, "events": events, "outcome": outcome}


def load_replay_frames(run_id: str, filename: str) -> dict[str, Any] | None:
    """Convert an evidence trace into decision-level visual replay frames."""

    try:
        path = _safe_replay_path(run_id, filename)
    except ValueError:
        return None
    if not path.is_file():
        return None

    frames: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    outcome: dict[str, Any] | None = None
    with path.open() as handle:
        for raw_line in handle:
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            event_type = payload.get("type")
            if event_type == "config":
                config = payload
            elif event_type == "observation" and isinstance(payload.get("state"), dict):
                compact = _compact_event(payload)
                frames.append(
                    {
                        "frame_index": len(frames),
                        "player": compact.get("player"),
                        "state": compact["state"],
                        "action": None,
                        "engine_events": [],
                        "invalid_action": None,
                    }
                )
            elif event_type == "model_turn" and frames:
                calls = payload.get("tool_calls") or []
                frames[-1]["fallback"] = bool(
                    (payload.get("trace") or {}).get("fallback", False)
                )
                frames[-1]["action"] = {
                    "player": payload.get("player", frames[-1].get("player")),
                    "agent": payload.get("agent", payload.get("policy", "unknown")),
                    "text": payload.get("text", ""),
                    "tool_calls": calls,
                    "elapsed_seconds": payload.get("elapsed_seconds", 0),
                    "agent_error": payload.get("agent_error"),
                }
            elif event_type == "tool_result" and frames:
                frames[-1]["engine_events"] = payload.get("events") or []
                frames[-1]["fallback"] = frames[-1].get("fallback", False) or bool(
                    payload.get("fallback", False)
                )
            elif event_type == "invalid_action" and frames:
                frames[-1]["invalid_action"] = payload
            elif event_type == "outcome":
                outcome = payload
    return {
        "run_id": run_id,
        "filename": filename,
        "config": config,
        "frames": frames,
        "outcome": outcome,
    }
