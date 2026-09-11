"""Phase-3 state reconstruction and interruption forecasting probes."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal, TypeVar

from ygobench.agents.llm_agent import compact_prompt_state
from ygobench.agents.provider_limits import omit_reasoning_model_token_limit
from ygobench.config import default_model_config
from ygobench.engine.upstream import UpstreamLayout
from ygobench.experiments.identity import (
    model_configuration_id,
    policy_descriptor,
    policy_id,
)
from ygobench.experiments.io import JsonlJournal, content_hash

_COMMITMENT_COMMANDS = {"activate", "summon", "sp_summon", "attack"}
_NEW_DECISION_BOUNDARIES = {"select_idlecmd", "select_battlecmd", "rock_paper_scissors"}
ForecastSampling = Literal["chronological", "stratified"]
T = TypeVar("T")


def _probe_policy_descriptor(
    provider_name: str | None = None, model: str | None = None
) -> dict[str, Any]:
    config = default_model_config(provider=provider_name, model=model)
    descriptor = policy_descriptor(
        f"react-fast:{config.provider}:{config.model}",
        runtime={
            "provider": config.provider,
            "model": config.model,
            "backend": config.backend,
            "thinking_enabled": False,
            "reasoning_effort": None,
            "temperature": 0.0,
            "max_tokens": None,
        },
    )
    descriptor.update(
        {
            "task_profile": "post_hoc_public_state_and_forecast_v2",
            "thinking_enabled": False,
            "reasoning_mode": "disabled",
            "prompt_version": "exp3-exp5-public-prefix-v2",
            "tool_schema_version": "post-hoc-probe-tools-v2",
            "context_policy": "public-prefix-no-hidden-state-v2",
        }
    )
    return descriptor


def probe_evaluator_id(provider_name: str | None = None, model: str | None = None) -> str:
    """Stable path identifier covering the complete post-hoc evaluator policy."""
    config = default_model_config(provider=provider_name, model=model)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{config.provider}__{config.model}")
    return f"{readable}__{policy_id(_probe_policy_descriptor(provider_name, model))}"


def probe_evaluator_identity(
    provider_name: str | None = None, model: str | None = None
) -> dict[str, Any]:
    descriptor = _probe_policy_descriptor(provider_name, model)
    return {
        "evaluator_id": probe_evaluator_id(provider_name, model),
        "policy_id": policy_id(descriptor),
        "model_configuration_id": model_configuration_id(descriptor),
        "configuration": descriptor,
    }


def _provider(provider_name: str | None = None, model: str | None = None):
    layout = UpstreamLayout()
    source = str(layout.root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from providers import get_provider  # type: ignore[import-not-found]

    config = default_model_config(provider=provider_name, model=model)
    backend = config.backend or config.provider
    provider = get_provider(
        backend,
        config.model,
        temperature=0.0,
        **({"base_url": config.base_url} if config.base_url else {}),
        **({"api_key": config.api_key} if config.api_key else {}),
    )
    omit_reasoning_model_token_limit(provider)
    if hasattr(provider, "reasoning_effort"):
        provider.reasoning_effort = None
    if hasattr(provider, "thinking_enabled"):
        provider.thinking_enabled = False
    return provider


def _named_visible_cards(side: dict[str, Any]) -> list[str]:
    """Return only card identities objectively public in a rendered side."""

    cards: list[str] = []
    for zone in (
        "monster_zone",
        "spell_trap_zone",
        "field_zone",
        "pendulum_zone",
        "graveyard",
        "banished",
    ):
        values = side.get(zone, [])
        if isinstance(values, dict):
            values = [values]
        if not isinstance(values, list):
            continue
        for card in values:
            if isinstance(card, dict) and card.get("name") and not card.get("face_down"):
                cards.append(str(card["name"]))
    return sorted(cards)


def _historical_truth(public_prefix: list[dict[str, Any]]) -> dict[str, Any]:
    """Truth slots derivable from the same public action/event prefix as the probe.

    The engine does not expose an effect-use registry, so this deliberately
    measures auditable public history rather than fabricating hidden flags.
    """

    activation_counts = [0, 0]
    event_types: list[str] = []
    for row in public_prefix:
        action = row.get("executed_action") or {}
        arguments = action.get("arguments", {}) if isinstance(action, dict) else {}
        if arguments.get("command") == "activate":
            activation_counts[int(row.get("player", 0))] += 1
        for event in row.get("engine_events", []):
            if isinstance(event, dict) and event.get("msg_name"):
                event_types.append(str(event["msg_name"]))
    return {
        "player0_public_activation_count": activation_counts[0],
        "player1_public_activation_count": activation_counts[1],
        # Keeping the suffix bounded prevents a long duel from turning this
        # slot into an unbounded transcription task while retaining recent
        # chain/resolution history.
        "recent_public_engine_event_types": event_types[-12:],
    }


def _state_truth(
    observation: dict[str, Any], oracle: dict[str, Any], public_prefix: list[dict[str, Any]]
) -> dict[str, Any]:
    field = oracle.get("field", {})
    players = field.get("players", [{}, {}])
    tracked = oracle.get("tracked", {})
    truth = {
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
    truth.update(_public_zone_truth(observation))
    perspective = int(observation.get("perspective_player", 0))
    sides = {
        perspective: observation.get("you", {}),
        1 - perspective: observation.get("opponent", {}),
    }
    for player in (0, 1):
        # The observation is perspective-relative, so resolve absolute seats
        # before taking public zone/resource counts from it.
        side = sides[player]
        truth[f"player{player}_banished_count"] = int(side.get("banished_count", 0))
        truth[f"player{player}_extra_deck_count"] = int(side.get("extra_deck_count", 0))
        truth[f"player{player}_known_public_cards"] = _named_visible_cards(side)
    truth.update(_historical_truth(public_prefix))
    return truth


def _public_zone_truth(observation: dict[str, Any]) -> dict[str, list[str]]:
    """Canonical public face-up cards, expressed from absolute player seats."""
    perspective = int(observation.get("perspective_player", 0))
    sides = {
        perspective: observation.get("you", {}),
        1 - perspective: observation.get("opponent", {}),
    }

    def names(player: int, zone: str) -> list[str]:
        cards = sides[player].get(zone, []) if isinstance(sides[player], dict) else []
        return sorted(
            str(card["name"])
            for card in cards
            if isinstance(card, dict)
            and not card.get("empty")
            and not card.get("face_down")
            and card.get("name")
        )

    return {
        "player0_faceup_monsters": names(0, "monster_zone"),
        "player1_faceup_monsters": names(1, "monster_zone"),
        "player0_faceup_spells_traps": names(0, "spell_trap_zone"),
        "player1_faceup_spells_traps": names(1, "spell_trap_zone"),
    }


def _zone_transition_count(row: dict[str, Any]) -> int:
    names = {"MSG_MOVE", "MSG_POS_CHANGE", "MSG_SWAP", "MSG_SHUFFLE_HAND", "MSG_SHUFFLE_DECK"}
    return sum(event.get("msg_name") in names for event in row.get("engine_events", []))


def _checkpoint_indices(public: list[dict[str, Any]]) -> dict[int, list[str]]:
    """PDF-specified quartiles plus deterministic high-complexity checkpoints."""
    tags: dict[int, list[str]] = {}

    def add(index: int, tag: str) -> None:
        tags.setdefault(index, []).append(tag)

    for fraction, tag in (
        (0.25, "trajectory_25"),
        (0.5, "trajectory_50"),
        (0.75, "trajectory_75"),
        (0.9, "trajectory_90"),
    ):
        add(min(len(public) - 1, round(fraction * (len(public) - 1))), tag)
    special = {
        "chain_depth_ge_2": [
            index
            for index, row in enumerate(public)
            if len(row.get("observation", {}).get("chain", [])) >= 2
        ],
        "zone_transition_heavy": [
            index for index, row in enumerate(public) if _zone_transition_count(row) >= 2
        ],
        "opponent_response": [
            index
            for index, row in enumerate(public)
            if row.get("legal", {}).get("expected_responder") == "select_chain"
            and row.get("executed_action", {}).get("arguments", {}).get("index") is not None
        ],
    }
    for tag, candidates in special.items():
        if candidates:
            add(candidates[len(candidates) // 2], tag)
    return tags


def _forecast_strata(observation: dict[str, Any], deck_matchup: str) -> dict[str, Any]:
    you = observation.get("you", {})
    opponent = observation.get("opponent", {})
    backrow = opponent.get("spell_trap_zone", []) if isinstance(opponent, dict) else []
    set_cards = sum(bool(isinstance(card, dict) and card.get("face_down")) for card in backrow)
    opponent_hand = int(opponent.get("hand_count", 0)) if isinstance(opponent, dict) else 0
    return {
        "turn_stage": str(observation.get("phase", "unknown")),
        "hand_size": int(you.get("hand_count", 0)) if isinstance(you, dict) else 0,
        "opponent_set_card_count": set_cards,
        "chain_depth": len(observation.get("chain", [])),
        "deck_matchup": deck_matchup,
        "hidden_information_density": opponent_hand + set_cards,
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


def _evenly_spaced(items: list[T], count: int) -> list[T]:
    """Select a deterministic trajectory-spanning subset without replacement."""
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def _select_forecast_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_samples: int,
    sampling: ForecastSampling,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample joint Availability/Behavior strata with a frozen seed."""
    if sampling == "chronological":
        return candidates[:max_samples]
    if sampling != "stratified":
        raise ValueError(f"Unsupported forecast sampling policy: {sampling}")

    groups = {
        label: [
            candidate
            for candidate in candidates
            if (candidate["availability_ground_truth"], candidate["behavior_ground_truth"]) == label
        ]
        for label in ((0, 0), (1, 0), (1, 1))
    }
    present = [label for label in ((1, 1), (1, 0), (0, 0)) if groups[label]]
    if not present or max_samples <= 0:
        return []

    quota, remainder = divmod(max_samples, len(present))
    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for position, label in enumerate(present):
        shuffled = list(groups[label])
        rng.shuffle(shuffled)
        selected.extend(shuffled[: quota + int(position < remainder)])

    # If a small class cannot use its quota, fill from the remaining candidates
    # while retaining deterministic, whole-trajectory coverage.
    selected_ids = {candidate["sample_id"] for candidate in selected}
    if len(selected) < min(max_samples, len(candidates)):
        remainder_pool = [
            candidate for candidate in candidates if candidate["sample_id"] not in selected_ids
        ]
        rng.shuffle(remainder_pool)
        selected.extend(remainder_pool[: min(max_samples, len(candidates)) - len(selected)])
    return sorted(selected, key=lambda candidate: candidate["trajectory_index"])


def _select_across_games(samples: list[T], count: int) -> list[T]:
    """Round-robin deterministic selection so early-sorted games cannot dominate."""
    if count <= 0:
        return []
    groups: dict[str, list[T]] = {}
    for sample in samples:
        game_id = str(sample.get("game_id", "unknown"))  # type: ignore[union-attr]
        groups.setdefault(game_id, []).append(sample)
    selected: list[T] = []
    index = 0
    while len(selected) < min(count, len(samples)):
        progressed = False
        for game_id in sorted(groups):
            if index < len(groups[game_id]):
                selected.append(groups[game_id][index])
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
        index += 1
    return selected


def extract_probe_samples(
    run_dir: Path,
    *,
    max_forecasts_per_game: int = 8,
    forecast_sampling: ForecastSampling = "stratified",
) -> dict[str, int]:
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
        for index, checkpoint_tags in sorted(_checkpoint_indices(public).items()):
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
                    "checkpoint_tags": sorted(checkpoint_tags),
                    "state_complexity": {
                        "chain_depth": len(public[index]["observation"].get("chain", [])),
                        "zone_transition_events": _zone_transition_count(public[index]),
                        "historical_action_count": index,
                        "historical_public_activation_count": sum(
                            1
                            for row in public[:index]
                            if row.get("executed_action", {}).get("arguments", {}).get("command")
                            == "activate"
                        ),
                    },
                    "prefix": prefix,
                    "ground_truth": _state_truth(
                        public[index]["observation"], oracle[index]["before"], public[:index]
                    ),
                }
            )
            state_count += 1

        manifest = json.loads((game_dir / "manifest.json").read_text(encoding="utf-8"))
        config = manifest["config"]
        deck_matchup = f"{config['deck1']}__vs__{config['deck2']}"
        forecast_candidates: list[dict[str, Any]] = []
        for index, row in enumerate(public):
            if not row.get("validation", {}).get("valid", False):
                continue
            action = row.get("executed_action")
            if not isinstance(action, dict):
                continue
            command = action.get("arguments", {}).get("command")
            commitment = command in _COMMITMENT_COMMANDS
            if not commitment:
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
            forecast_candidates.append(
                {
                    "sample_id": f"{row['game_id']}:{row['decision_id']}:forecast",
                    "game_id": row["game_id"],
                    "decision_id": row["decision_id"],
                    "trajectory_index": index,
                    "observation": compact_prompt_state(row["observation"]),
                    "commitment_action": action,
                    "response_window_id": response_row["decision_id"],
                    "response_window_index": response_index + 1,
                    "opponent_legal_response_set": legal_responses,
                    "actual_opponent_response": actual_response,
                    "availability_ground_truth": int(availability),
                    "behavior_ground_truth": int(behavior),
                    "joint_stratum": f"A{int(availability)}B{int(behavior)}",
                    "strata": _forecast_strata(row["observation"], deck_matchup),
                }
            )

        impossible = [
            sample
            for sample in forecast_candidates
            if sample["availability_ground_truth"] == 0 and sample["behavior_ground_truth"] == 1
        ]
        if impossible:
            raise ValueError(f"{game_dir.name} contains logically impossible A0B1 forecast windows")

        selected_forecasts = _select_forecast_candidates(
            forecast_candidates,
            max_samples=max_forecasts_per_game,
            sampling=forecast_sampling,
            seed=int(config["seed"]),
        )
        candidate_strata = Counter(candidate["joint_stratum"] for candidate in forecast_candidates)
        selected_strata = Counter(candidate["joint_stratum"] for candidate in selected_forecasts)
        for sample in selected_forecasts:
            sample["forecast_sampling"] = forecast_sampling
            sample["sampling_seed"] = int(config["seed"])
            sample["inclusion_probability"] = (
                selected_strata[sample["joint_stratum"]] / candidate_strata[sample["joint_stratum"]]
            )
            sample["candidate_joint_stratum_counts"] = dict(sorted(candidate_strata.items()))
            sample["candidate_class_counts"] = {
                "availability_positive": sum(
                    candidate["availability_ground_truth"] for candidate in forecast_candidates
                ),
                "availability_negative": sum(
                    not candidate["availability_ground_truth"] for candidate in forecast_candidates
                ),
            }
            forecast_out.append(sample)
            forecast_count += 1
    return {"state_samples": state_count, "forecast_samples": forecast_count}


def run_probes(
    run_dir: Path,
    *,
    provider_name: str | None = None,
    model: str | None = None,
    max_state_samples: int = 4,
    max_forecast_samples: int = 4,
    experiments: tuple[str, ...] = ("exp3", "exp5"),
) -> dict[str, int]:
    unknown = set(experiments) - {"exp3", "exp5"}
    if unknown:
        raise ValueError(f"Unknown probe experiments: {sorted(unknown)}")
    evaluator_identity = probe_evaluator_identity(provider_name, model)
    evaluator_id = evaluator_identity["evaluator_id"]
    provider = _provider(provider_name, model)
    state_samples = (
        _select_across_games(
            JsonlJournal(run_dir / "derived" / "state_probe_samples.jsonl").recover(),
            max_state_samples,
        )
        if "exp3" in experiments
        else []
    )
    forecast_samples = (
        _select_across_games(
            JsonlJournal(run_dir / "derived" / "forecast_samples.jsonl").recover(),
            max_forecast_samples,
        )
        if "exp5" in experiments
        else []
    )
    output_dir = run_dir / "derived" / "probe_results" / evaluator_id
    state_out = JsonlJournal(output_dir / "state_probe_results.jsonl")
    forecast_out = JsonlJournal(output_dir / "forecast_results.jsonl")
    state_hashes = {sample["sample_id"]: content_hash(sample) for sample in state_samples}
    forecast_hashes = {sample["sample_id"]: content_hash(sample) for sample in forecast_samples}
    retained_state = state_out.recover()
    retained_forecast = forecast_out.recover()
    if "exp3" in experiments:
        retained_state = [
            row
            for row in retained_state
            if row.get("sample_hash") == state_hashes.get(row.get("sample_id"))
        ]
        state_out.rewrite(retained_state)
    if "exp5" in experiments:
        retained_forecast = [
            row
            for row in retained_forecast
            if row.get("sample_hash") == forecast_hashes.get(row.get("sample_id"))
        ]
        forecast_out.rewrite(retained_forecast)
    completed_state = {row["sample_id"] for row in retained_state}
    completed_forecast = {row["sample_id"] for row in retained_forecast}

    def schema_for(value: Any) -> dict[str, Any]:
        if isinstance(value, int):
            return {"type": "integer"}
        if isinstance(value, list):
            return {"type": "array", "items": {"type": "string"}}
        return {"type": "string"}

    state_tool = {
        "name": "report_state",
        "description": "Report the reconstructed public duel state.",
        "input_schema": {
            "type": "object",
            "properties": {
                key: schema_for(value) for key, value in state_samples[0]["ground_truth"].items()
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
                "Reconstruct only objectively knowable state from the supplied full-duel prefix."
            ),
            messages=[
                {"role": "user", "content": json.dumps(sample["prefix"], ensure_ascii=False)}
            ],
            tools=[state_tool],
        )
        call = next((call for call in turn.tool_calls if call.name == "report_state"), None)
        prediction = call.arguments if call else None
        schema_errors = []
        if not isinstance(prediction, dict):
            schema_errors.append("missing_or_non_object_prediction")
        else:
            for key, expected in sample["ground_truth"].items():
                if key not in prediction:
                    schema_errors.append(f"missing_field:{key}")
                elif isinstance(expected, list) and not isinstance(prediction[key], list):
                    schema_errors.append(f"wrong_type:{key}:array")
                elif isinstance(expected, int) and (
                    not isinstance(prediction[key], int) or isinstance(prediction[key], bool)
                ):
                    schema_errors.append(f"wrong_type:{key}:integer")
                elif isinstance(expected, str) and not isinstance(prediction[key], str):
                    schema_errors.append(f"wrong_type:{key}:string")
        state_out.append(
            {
                "sample_id": sample["sample_id"],
                "game_id": sample["game_id"],
                "sample_hash": state_hashes[sample["sample_id"]],
                "evaluator_identity": evaluator_identity,
                "ground_truth": sample["ground_truth"],
                "prediction": prediction,
                "schema_valid": not schema_errors,
                "schema_errors": schema_errors,
                "trajectory_progress": sample["trajectory_progress"],
                "checkpoint_tags": sample["checkpoint_tags"],
                "state_complexity": sample["state_complexity"],
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
                "Using only the acting player's visible information, estimate two "
                "separate probabilities: whether the opponent objectively has a legal "
                "response, and, conditional on such a response being available, whether "
                "the opponent will choose to use it."
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
        schema_errors = []
        for key, value in (
            ("p_opponent_has_legal_response", availability_probability),
            ("p_opponent_will_respond", behavior_probability),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                schema_errors.append(f"missing_or_non_numeric:{key}")
            elif not 0 <= float(value) <= 1:
                schema_errors.append(f"out_of_range:{key}")
        forecast_out.append(
            {
                "sample_id": sample["sample_id"],
                "game_id": sample["game_id"],
                "sample_hash": forecast_hashes[sample["sample_id"]],
                "evaluator_identity": evaluator_identity,
                "response_window_id": sample["response_window_id"],
                "availability_ground_truth": sample["availability_ground_truth"],
                "behavior_ground_truth": sample["behavior_ground_truth"],
                "availability_probability": availability_probability,
                "behavior_probability": behavior_probability,
                "schema_valid": not schema_errors,
                "schema_errors": schema_errors,
                "joint_stratum": sample["joint_stratum"],
                "inclusion_probability": sample.get("inclusion_probability"),
                "candidate_joint_stratum_counts": sample.get("candidate_joint_stratum_counts"),
                "strata": sample["strata"],
                "usage": turn.usage,
                "elapsed_seconds": turn.wallclock_seconds,
            }
        )
    return {
        "evaluator_id": evaluator_id,
        "state_results": len(state_out.recover()),
        "forecast_results": len(forecast_out.recover()),
    }
