"""Phase-3 state reconstruction and interruption forecasting probes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ygobench.agents.llm_agent import compact_prompt_state
from ygobench.agents.provider_limits import omit_deepseek_token_limit
from ygobench.engine.upstream import UpstreamLayout
from ygobench.experiments.io import JsonlJournal, content_hash

_COMMITMENT_COMMANDS = {"activate", "summon", "sp_summon", "attack"}
_NEW_DECISION_BOUNDARIES = {"select_idlecmd", "select_battlecmd", "rock_paper_scissors"}


def _provider(model: str):
    layout = UpstreamLayout()
    source = str(layout.root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from providers import get_provider  # type: ignore[import-not-found]

    provider = get_provider("deepseek", model, temperature=0.0)
    omit_deepseek_token_limit(provider)
    provider.reasoning_effort = None
    provider.thinking_enabled = False
    return provider


def _state_truth(observation: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    field = oracle.get("field", {})
    players = field.get("players", [{}, {}])
    tracked = oracle.get("tracked", {})
    return {
        "turn": int(tracked.get("turn_count", observation.get("turn", 0))),
        "turn_player": int(tracked.get("turn_player", 0)),
        "phase": str(observation.get("phase", "")),
        "chain_depth": len(field.get("chain", [])),
        "player0_lp": int(players[0].get("lp", 0)),
        "player1_lp": int(players[1].get("lp", 0)),
        "player0_hand_count": int(players[0].get("hand_count_raw", 0)),
        "player1_hand_count": int(players[1].get("hand_count_raw", 0)),
        "player0_grave_count": int(players[0].get("grave_count_raw", 0)),
        "player1_grave_count": int(players[1].get("grave_count_raw", 0)),
    }


def _initial_public_summary(observation: dict[str, Any]) -> dict[str, Any]:
    perspective = int(observation.get("perspective_player", 0))
    sides = [None, None]
    sides[perspective] = observation.get("you", {})
    sides[1 - perspective] = observation.get("opponent", {})
    return {
        "turn": observation.get("turn"),
        "turn_player_relative": observation.get("turn_player"),
        "phase": observation.get("phase"),
        "perspective_player": perspective,
        "players": [
            {
                "lp": side.get("lp"),
                "deck_count": side.get("deck_count"),
                "hand_count": side.get("hand_count"),
                "grave_count": side.get("grave_count"),
                "banished_count": side.get("banished_count"),
            }
            for side in sides
        ],
        "events": observation.get("events_since_last_decision", []),
    }


def _response_window_after(
    public: list[dict[str, Any]], commitment_index: int
) -> tuple[int, dict[str, Any]] | None:
    """Find the causal opponent chain window after a committed game action.

    Target/cost/position selections by the committing player may occur between
    the commitment and the opponent's response.  A new idle/battle decision,
    turn change, or another committed action closes the search so an unrelated
    later chain window cannot be attached to the sample.
    """

    commitment = public[commitment_index]
    actor = int(commitment["player"])
    turn = int(commitment["turn"])
    for response_index in range(commitment_index + 1, len(public)):
        row = public[response_index]
        if int(row["turn"]) != turn:
            return None
        responder = str(row["legal"].get("expected_responder", ""))
        if int(row["player"]) != actor and responder == "select_chain":
            return response_index, row
        command = row.get("executed_action", {}).get("arguments", {}).get("command")
        if responder in _NEW_DECISION_BOUNDARIES or command in _COMMITMENT_COMMANDS:
            return None
    return None


def extract_probe_samples(run_dir: Path, *, max_forecasts_per_game: int = 8) -> dict[str, int]:
    state_out = JsonlJournal(run_dir / "derived" / "state_probe_samples.jsonl")
    forecast_out = JsonlJournal(run_dir / "derived" / "forecast_samples.jsonl")
    state_out.rewrite([])
    forecast_out.rewrite([])
    state_count = forecast_count = 0
    for game_dir in sorted((run_dir / "games").glob("*")):
        if not (game_dir / "manifest.json").is_file():
            continue
        public = JsonlJournal(game_dir / "trajectory.jsonl").recover()
        oracle = JsonlJournal(game_dir / "oracle_trajectory.jsonl").recover()
        if not public:
            continue
        indices = sorted(
            {
                min(len(public) - 1, round(fraction * (len(public) - 1)))
                for fraction in (0.25, 0.5, 0.75, 0.9)
            }
        )
        for index in indices:
            prefix = {
                "initial_public_state": _initial_public_summary(public[0]["observation"]),
                "executed_history": [
                    {
                    "turn": row["turn"],
                    "phase": row["phase"],
                    "player": row["player"],
                    "executed_action": row["executed_action"],
                    "engine_events": row["engine_events"],
                    }
                    for row in public[:index]
                ],
                "query_context": {
                    "decision_id": public[index]["decision_id"],
                    "turn": public[index]["turn"],
                    "phase": public[index]["phase"],
                    "acting_player": public[index]["player"],
                },
            }
            state_out.append(
                {
                    "sample_id": f"{public[index]['game_id']}:{public[index]['decision_id']}:state",
                    "game_id": public[index]["game_id"],
                    "decision_id": public[index]["decision_id"],
                    "trajectory_progress": (index + 1) / len(public),
                    "prefix": prefix,
                    "ground_truth": _state_truth(
                        public[index]["observation"], oracle[index]["before"]
                    ),
                }
            )
            state_count += 1

        candidates = 0
        for index, row in enumerate(public):
            if not row.get("validation", {}).get("valid", False):
                continue
            action = row.get("executed_action")
            if not isinstance(action, dict):
                continue
            command = action.get("arguments", {}).get("command")
            commitment = command in _COMMITMENT_COMMANDS
            if not commitment or candidates >= max_forecasts_per_game:
                continue
            window = _response_window_after(public, index)
            if window is None:
                continue
            response_index, response_row = window
            legal_responses = response_row["legal"].get("actions", [])
            availability = any(
                candidate.get("arguments", {}).get("index") is not None
                for candidate in legal_responses
            )
            actual_response = response_row.get("executed_action")
            behavior = bool(
                isinstance(actual_response, dict)
                and actual_response.get("arguments", {}).get("index") is not None
                and response_row.get("validation", {}).get("valid", False)
            )
            forecast_out.append(
                {
                    "sample_id": f"{row['game_id']}:{row['decision_id']}:forecast",
                    "game_id": row["game_id"],
                    "decision_id": row["decision_id"],
                    "observation": compact_prompt_state(row["observation"]),
                    "commitment_action": action,
                    "response_window_id": response_row["decision_id"],
                    "response_window_index": response_index + 1,
                    "opponent_legal_response_set": legal_responses,
                    "actual_opponent_response": actual_response,
                    "availability_ground_truth": int(availability),
                    "behavior_ground_truth": int(behavior),
                }
            )
            forecast_count += 1
            candidates += 1
    return {"state_samples": state_count, "forecast_samples": forecast_count}


def run_probes(
    run_dir: Path,
    *,
    model: str = "deepseek-v4-flash",
    max_state_samples: int = 4,
    max_forecast_samples: int = 4,
) -> dict[str, int]:
    provider = _provider(model)
    state_samples = JsonlJournal(run_dir / "derived" / "state_probe_samples.jsonl").recover()[
        :max_state_samples
    ]
    forecast_samples = JsonlJournal(run_dir / "derived" / "forecast_samples.jsonl").recover()[
        :max_forecast_samples
    ]
    state_out = JsonlJournal(run_dir / "derived" / "state_probe_results.jsonl")
    forecast_out = JsonlJournal(run_dir / "derived" / "forecast_results.jsonl")
    state_hashes = {sample["sample_id"]: content_hash(sample) for sample in state_samples}
    forecast_hashes = {
        sample["sample_id"]: content_hash(sample) for sample in forecast_samples
    }
    retained_state = [
        row
        for row in state_out.recover()
        if row.get("sample_hash") == state_hashes.get(row.get("sample_id"))
    ]
    retained_forecast = [
        row
        for row in forecast_out.recover()
        if row.get("sample_hash") == forecast_hashes.get(row.get("sample_id"))
    ]
    state_out.rewrite(retained_state)
    forecast_out.rewrite(retained_forecast)
    completed_state = {row["sample_id"] for row in retained_state}
    completed_forecast = {row["sample_id"] for row in retained_forecast}
    state_tool = {
        "name": "report_state",
        "description": "Report the reconstructed public duel state.",
        "input_schema": {
            "type": "object",
            "properties": {
                key: {"type": "integer" if isinstance(value, int) else "string"}
                for key, value in state_samples[0]["ground_truth"].items()
            }
            if state_samples
            else {},
            "required": list(state_samples[0]["ground_truth"]) if state_samples else [],
        },
    }
    for sample in state_samples:
        if sample["sample_id"] in completed_state:
            continue
        turn = provider.respond(
            system=(
                "Reconstruct only objectively knowable state from the supplied "
                "full-duel prefix."
            ),
            messages=[
                {"role": "user", "content": json.dumps(sample["prefix"], ensure_ascii=False)}
            ],
            tools=[state_tool],
        )
        call = next((call for call in turn.tool_calls if call.name == "report_state"), None)
        state_out.append(
            {
                "sample_id": sample["sample_id"],
                "sample_hash": state_hashes[sample["sample_id"]],
                "ground_truth": sample["ground_truth"],
                "prediction": call.arguments if call else None,
                "usage": turn.usage,
                "elapsed_seconds": turn.wallclock_seconds,
            }
        )
    forecast_tool = {
        "name": "report_forecast",
        "description": (
            "Report separate probabilities for legal response availability and "
            "actual opponent response behavior."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "p_opponent_has_legal_response": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "p_opponent_will_respond": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "p_opponent_has_legal_response",
                "p_opponent_will_respond",
            ],
        },
    }
    for sample in forecast_samples:
        if sample["sample_id"] in completed_forecast:
            continue
        turn = provider.respond(
            system=(
                "Estimate response availability from the acting player's visible "
                "information only."
            ),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "observation": sample["observation"],
                            "action": sample["commitment_action"],
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            tools=[forecast_tool],
        )
        call = next((call for call in turn.tool_calls if call.name == "report_forecast"), None)
        availability_probability = (
            call.arguments.get("p_opponent_has_legal_response") if call else None
        )
        behavior_probability = call.arguments.get("p_opponent_will_respond") if call else None
        forecast_out.append(
            {
                "sample_id": sample["sample_id"],
                "sample_hash": forecast_hashes[sample["sample_id"]],
                "response_window_id": sample["response_window_id"],
                "availability_ground_truth": sample["availability_ground_truth"],
                "behavior_ground_truth": sample["behavior_ground_truth"],
                "availability_probability": availability_probability,
                "behavior_probability": behavior_probability,
                "usage": turn.usage,
                "elapsed_seconds": turn.wallclock_seconds,
            }
        )
    return {
        "state_results": len(state_out.recover()),
        "forecast_results": len(forecast_out.recover()),
    }
