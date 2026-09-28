"""Phase-3 state reconstruction and interruption forecasting probes."""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

from ygobench.agents.llm_agent import compact_prompt_state
from ygobench.agents.provider_limits import (
    adapt_gagawenai_gemini,
    force_gagawenai_gemini_thinking_low,
    force_provider_thinking_disabled,
    omit_provider_token_limit,
)
from ygobench.config import default_model_config
from ygobench.engine.upstream import UpstreamLayout
from ygobench.experiments.config import stable_id
from ygobench.experiments.identity import (
    model_configuration_id,
    policy_descriptor,
    policy_id,
)
from ygobench.experiments.io import JsonlJournal, atomic_write_json, content_hash, read_json
from ygobench.experiments.recovery import (
    RecoveryConflict,
    RecoveryStore,
    pause_status,
)
from ygobench.experiments.registry import LeaseHeartbeat, TaskRegistry

_COMMITMENT_COMMANDS = {"activate", "summon", "sp_summon", "attack"}
_NEW_DECISION_BOUNDARIES = {"select_idlecmd", "select_battlecmd", "rock_paper_scissors"}
ForecastSampling = Literal["chronological", "stratified"]
T = TypeVar("T")

FORMAL_STATE_SAMPLES_PER_MODEL = 48
FORMAL_FORECAST_SAMPLES_PER_MODEL = 90
PILOT_STATE_SAMPLES_PER_MODEL = 24
PILOT_FORECAST_SAMPLES_PER_MODEL = 45
PROBE_SAMPLING_VERSION = "budget-stratified-v1"


def _probe_policy_descriptor(
    provider_name: str | None = None, model: str | None = None
) -> dict[str, Any]:
    config = default_model_config(provider=provider_name, model=model)
    gemini_low = config.provider == "gagawenai-gemini"
    descriptor = policy_descriptor(
        f"react-fast:{config.provider}:{config.model}",
        runtime={
            "provider": config.provider,
            "model": config.model,
            "backend": config.backend,
            "thinking_enabled": gemini_low,
            "reasoning_effort": "low" if gemini_low else None,
            "temperature": 0.0,
            "max_tokens": None,
        },
    )
    descriptor.update(
        {
            "task_profile": "post_hoc_public_state_and_forecast_v2",
            "thinking_enabled": gemini_low,
            "reasoning_mode": "low" if gemini_low else "disabled",
            "thinking_control": (
                "gagawenai-gemini.reasoning_effort"
                if gemini_low
                else "explicit-provider-wire-disable-v1"
            ),
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
    if config.provider == "gagawenai-gemini":
        adapt_gagawenai_gemini(provider)
    omit_provider_token_limit(provider)
    if config.provider == "gagawenai-gemini":
        force_gagawenai_gemini_thinking_low(provider)
    else:
        force_provider_thinking_disabled(provider, provider_name=config.provider)
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


def _stable_round_robin(samples: list[dict[str, Any]], *, salt: str) -> list[dict[str, Any]]:
    """Order a stratum reproducibly while exhausting distinct games first."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(str(sample.get("game_id", "unknown")), []).append(sample)
    for game_samples in groups.values():
        game_samples.sort(
            key=lambda sample: content_hash(
                [PROBE_SAMPLING_VERSION, salt, str(sample.get("sample_id"))]
            )
        )
    ordered: list[dict[str, Any]] = []
    game_order = sorted(
        groups,
        key=lambda game_id: content_hash(
            [PROBE_SAMPLING_VERSION, salt, "game", game_id]
        ),
    )
    depth = 0
    while True:
        progressed = False
        for game_id in game_order:
            if depth < len(groups[game_id]):
                ordered.append(groups[game_id][depth])
                progressed = True
        if not progressed:
            return ordered
        depth += 1


def _interleaved_strata(
    groups: dict[str, list[dict[str, Any]]], order: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Build a prefix-stable balanced sequence across the requested strata."""

    sequence: list[dict[str, Any]] = []
    depth = 0
    while True:
        progressed = False
        for label in order:
            values = groups.get(label, [])
            if depth < len(values):
                sequence.append(values[depth])
                progressed = True
        if not progressed:
            return sequence
        depth += 1


def _spread_across_games(sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer one sample per duel before admitting a second or third sample."""

    ordered: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_game: Counter[str] = Counter()
    cap = 1
    while len(ordered) < len(sequence):
        progressed = False
        for sample in sequence:
            sample_id = str(sample.get("sample_id"))
            game_id = str(sample.get("game_id", "unknown"))
            if sample_id in selected_ids or per_game[game_id] >= cap:
                continue
            ordered.append(sample)
            selected_ids.add(sample_id)
            per_game[game_id] += 1
            progressed = True
        if not progressed:
            break
        cap += 1
    return ordered


def _progress_quartile(value: float) -> str:
    if value <= 0.25:
        return "Q1"
    if value <= 0.5:
        return "Q2"
    if value <= 0.75:
        return "Q3"
    return "Q4"


def _state_complexity_score(sample: dict[str, Any]) -> int:
    values = sample.get("state_complexity", {})
    public_cards = sum(
        len(value)
        for key, value in (sample.get("ground_truth") or {}).items()
        if key.endswith("known_public_cards") and isinstance(value, list)
    )
    return (
        3 * int(values.get("chain_depth", 0) or 0)
        + 2 * int(values.get("zone_transition_events", 0) or 0)
        + int(values.get("historical_public_activation_count", 0) or 0)
        + public_cards
    )


def _select_state_samples(
    candidates: list[dict[str, Any]], max_samples: int
) -> list[dict[str, Any]]:
    """Select Q1-Q4 x Low/High states with a prefix-stable budget sequence."""

    if max_samples <= 0 or not candidates:
        return []
    low_ids: set[str] = set()
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        ranked = sorted(
            [
                sample
                for sample in candidates
                if _progress_quartile(float(sample.get("trajectory_progress", 0))) == quartile
            ],
            key=lambda sample: (_state_complexity_score(sample), str(sample.get("sample_id"))),
        )
        low_ids.update(
            str(sample["sample_id"]) for sample in ranked[: (len(ranked) + 1) // 2]
        )
    annotated: list[dict[str, Any]] = []
    for source in candidates:
        sample = dict(source)
        band = "Low" if str(sample["sample_id"]) in low_ids else "High"
        quartile = _progress_quartile(float(sample.get("trajectory_progress", 0)))
        sample["state_complexity_score"] = _state_complexity_score(sample)
        sample["state_complexity_band"] = band
        sample["state_sampling_stratum"] = f"{quartile}_{band}"
        annotated.append(sample)
    order = tuple(f"Q{quartile}_{band}" for quartile in range(1, 5) for band in ("Low", "High"))
    groups = {
        label: _stable_round_robin(
            [sample for sample in annotated if sample["state_sampling_stratum"] == label],
            salt=f"state:{label}",
        )
        for label in order
    }
    selected = _spread_across_games(_interleaved_strata(groups, order))[
        : min(max_samples, len(annotated))
    ]
    pool_counts = Counter(sample["state_sampling_stratum"] for sample in annotated)
    selected_counts = Counter(sample["state_sampling_stratum"] for sample in selected)
    for sample in selected:
        label = sample["state_sampling_stratum"]
        sample["inclusion_probability"] = selected_counts[label] / pool_counts[label]
        sample["candidate_state_stratum_counts"] = dict(sorted(pool_counts.items()))
    return selected


def _select_forecast_samples(
    candidates: list[dict[str, Any]],
    *,
    max_samples: int,
    sampling: ForecastSampling,
) -> list[dict[str, Any]]:
    """Select global A0B0/A1B0/A1B1 strata with expansion-safe prefixes."""

    if max_samples <= 0 or not candidates:
        return []
    if sampling == "chronological":
        selected = _select_across_games(candidates, max_samples)
    elif sampling == "stratified":
        order = ("A0B0", "A1B0", "A1B1")
        groups = {
            label: _stable_round_robin(
                [sample for sample in candidates if sample.get("joint_stratum") == label],
                salt=f"forecast:{label}",
            )
            for label in order
        }
        selected = _spread_across_games(_interleaved_strata(groups, order))[
            : min(max_samples, len(candidates))
        ]
    else:
        raise ValueError(f"Unsupported forecast sampling policy: {sampling}")
    selected = [dict(sample) for sample in selected]
    pool_counts = Counter(str(sample.get("joint_stratum")) for sample in candidates)
    selected_counts = Counter(str(sample.get("joint_stratum")) for sample in selected)
    for sample in selected:
        label = str(sample["joint_stratum"])
        sample["inclusion_probability"] = selected_counts[label] / pool_counts[label]
        sample["global_candidate_joint_stratum_counts"] = dict(sorted(pool_counts.items()))
    return selected


def extract_probe_samples(
    run_dir: Path,
    *,
    max_forecasts_per_game: int = 0,
    forecast_sampling: ForecastSampling = "stratified",
) -> dict[str, int]:
    state_out = JsonlJournal(run_dir / "derived" / "state_probe_samples.jsonl")
    forecast_out = JsonlJournal(run_dir / "derived" / "forecast_samples.jsonl")
    state_records: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []
    state_count = forecast_count = 0
    for game_dir in sorted((run_dir / "games").glob("*")):
        if not (game_dir / "manifest.json").is_file():
            continue
        outcome = read_json(game_dir / "outcome.json")
        status = read_json(game_dir / "status.json") or {}
        if (
            not outcome
            or status.get("status") != "COMPLETED"
            or not outcome.get("game_over")
            or not outcome.get("competitive_eligible")
        ):
            raise RecoveryConflict(f"duel {game_dir.name} is not a complete rated game")
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
            state_records.append(
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

        selected_forecasts = (
            _select_forecast_candidates(
                forecast_candidates,
                max_samples=max_forecasts_per_game,
                sampling=forecast_sampling,
                seed=int(config["seed"]),
            )
            if max_forecasts_per_game > 0
            else forecast_candidates
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
            forecast_records.append(sample)
            forecast_count += 1
    state_out.rewrite(state_records)
    forecast_out.rewrite(forecast_records)
    return {"state_samples": state_count, "forecast_samples": forecast_count}


def _execute_probe_task(
    *,
    registry: TaskRegistry,
    recovery_store: RecoveryStore,
    evaluator_id: str,
    experiment: str,
    sample: dict[str, Any],
    sample_hash: str,
    request: dict[str, Any],
    invoke: Callable[[], Any],
    make_record: Callable[[Any], dict[str, Any]],
    journal: JsonlJournal,
) -> None:
    task_id = stable_id(
        "probe", [evaluator_id, experiment, sample["sample_id"]], length=24
    )
    registry.add(
        task_id,
        "probe",
        {
            "evaluator_id": evaluator_id,
            "experiment": experiment,
            "sample_id": sample["sample_id"],
            "sample_hash": sample_hash,
        },
    )
    if not registry.claim(task_id):
        raise RuntimeError(f"probe task {task_id} is already running or completed")
    generation = int(registry.row(task_id)["lease_generation"])
    heartbeat = LeaseHeartbeat(registry, task_id, generation)
    heartbeat.start()
    api_request_pending = True
    try:
        turn = recovery_store.call(
            scope_id=f"probe:{evaluator_id}:{experiment}:{sample['sample_id']}",
            call_index=0,
            request=request,
            invoke=invoke,
        ).turn
        api_request_pending = False
        record = make_record(turn)
        heartbeat.assert_owned()
        journal.append(record)
        if not registry.finish(task_id, generation=generation):
            raise RuntimeError(f"probe task {task_id} lost its lease")
    except Exception as exc:
        registry.finish(
            task_id,
            error=f"{type(exc).__name__}: {exc}",
            status=(
                pause_status(exc)
                if api_request_pending
                else "FAILED_RETRYABLE"
            ),
            generation=generation,
        )
        raise
    finally:
        heartbeat.stop()


def _state_result_record(
    sample: dict[str, Any], sample_hash: str, evaluator_identity: dict[str, Any], turn: Any
) -> dict[str, Any]:
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
    return {
        "sample_id": sample["sample_id"],
        "game_id": sample["game_id"],
        "sample_hash": sample_hash,
        "evaluator_identity": evaluator_identity,
        "ground_truth": sample["ground_truth"],
        "prediction": prediction,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "trajectory_progress": sample["trajectory_progress"],
        "checkpoint_tags": sample["checkpoint_tags"],
        "state_complexity": sample["state_complexity"],
        "state_complexity_score": sample["state_complexity_score"],
        "state_complexity_band": sample["state_complexity_band"],
        "state_sampling_stratum": sample["state_sampling_stratum"],
        "inclusion_probability": sample.get("inclusion_probability"),
        "candidate_state_stratum_counts": sample.get("candidate_state_stratum_counts"),
        "usage": turn.usage,
        "elapsed_seconds": turn.wallclock_seconds,
    }


def _forecast_result_record(
    sample: dict[str, Any], sample_hash: str, evaluator_identity: dict[str, Any], turn: Any
) -> dict[str, Any]:
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
    return {
        "sample_id": sample["sample_id"],
        "game_id": sample["game_id"],
        "sample_hash": sample_hash,
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
        "global_candidate_joint_stratum_counts": sample.get(
            "global_candidate_joint_stratum_counts"
        ),
        "strata": sample["strata"],
        "usage": turn.usage,
        "elapsed_seconds": turn.wallclock_seconds,
    }


def run_probes(
    run_dir: Path,
    *,
    provider_name: str | None = None,
    model: str | None = None,
    max_state_samples: int = FORMAL_STATE_SAMPLES_PER_MODEL,
    max_forecast_samples: int = FORMAL_FORECAST_SAMPLES_PER_MODEL,
    forecast_sampling: ForecastSampling = "stratified",
    experiments: tuple[str, ...] = ("exp3", "exp5"),
) -> dict[str, int]:
    unknown = set(experiments) - {"exp3", "exp5"}
    if unknown:
        raise ValueError(f"Unknown probe experiments: {sorted(unknown)}")
    evaluator_identity = probe_evaluator_identity(provider_name, model)
    evaluator_id = evaluator_identity["evaluator_id"]
    provider = _provider(provider_name, model)
    recovery_store = RecoveryStore(run_dir / "recovery.sqlite")
    registry = TaskRegistry(run_dir / "task_state.sqlite")
    state_samples = (
        _select_state_samples(
            JsonlJournal(run_dir / "derived" / "state_probe_samples.jsonl").recover(),
            max_state_samples,
        )
        if "exp3" in experiments
        else []
    )
    forecast_samples = (
        _select_forecast_samples(
            JsonlJournal(run_dir / "derived" / "forecast_samples.jsonl").recover(),
            max_samples=max_forecast_samples,
            sampling=forecast_sampling,
        )
        if "exp5" in experiments
        else []
    )
    output_dir = run_dir / "derived" / "probe_results" / evaluator_id
    sample_manifest = {
        "sampling_version": PROBE_SAMPLING_VERSION,
        "evaluator_identity": evaluator_identity,
        "forecast_sampling": forecast_sampling,
        "budget": {
            "state_samples_requested": max_state_samples,
            "forecast_samples_requested": max_forecast_samples,
            "pilot_defaults": {
                "state": PILOT_STATE_SAMPLES_PER_MODEL,
                "forecast": PILOT_FORECAST_SAMPLES_PER_MODEL,
            },
            "formal_defaults": {
                "state": FORMAL_STATE_SAMPLES_PER_MODEL,
                "forecast": FORMAL_FORECAST_SAMPLES_PER_MODEL,
            },
        },
        "prefix_extension_safe": True,
        "state_sample_ids": [sample["sample_id"] for sample in state_samples],
        "forecast_sample_ids": [sample["sample_id"] for sample in forecast_samples],
        "state_stratum_counts": dict(
            sorted(Counter(sample["state_sampling_stratum"] for sample in state_samples).items())
        ),
        "forecast_stratum_counts": dict(
            sorted(Counter(sample["joint_stratum"] for sample in forecast_samples).items())
        ),
        "state_pool_hash": content_hash(
            JsonlJournal(run_dir / "derived" / "state_probe_samples.jsonl").recover()
        ),
        "forecast_pool_hash": content_hash(
            JsonlJournal(run_dir / "derived" / "forecast_samples.jsonl").recover()
        ),
    }
    manifest_path = output_dir / "sample_manifest.json"
    frozen = read_json(manifest_path)
    if frozen:
        for key in (
            "sampling_version",
            "evaluator_identity",
            "forecast_sampling",
            "state_pool_hash",
            "forecast_pool_hash",
        ):
            if frozen.get(key) != sample_manifest[key]:
                raise RecoveryConflict(f"probe {key} changed during resume")
        for kind in ("state", "forecast"):
            old_ids = frozen.get(f"{kind}_sample_ids", [])
            new_ids = sample_manifest[f"{kind}_sample_ids"]
            if not new_ids:
                sample_manifest[f"{kind}_sample_ids"] = old_ids
            elif new_ids[: len(old_ids)] != old_ids:
                raise RecoveryConflict(f"probe {kind} selection is not a prefix extension")
    atomic_write_json(output_dir / "sample_manifest.json", sample_manifest)
    state_out = JsonlJournal(output_dir / "state_probe_results.jsonl")
    forecast_out = JsonlJournal(output_dir / "forecast_results.jsonl")
    sampling_fields = {
        "inclusion_probability",
        "candidate_state_stratum_counts",
        "global_candidate_joint_stratum_counts",
    }

    def payload_hash(sample: dict[str, Any]) -> str:
        return content_hash(
            {key: value for key, value in sample.items() if key not in sampling_fields}
        )

    state_hashes = {sample["sample_id"]: payload_hash(sample) for sample in state_samples}
    forecast_hashes = {sample["sample_id"]: payload_hash(sample) for sample in forecast_samples}
    state_by_id = {sample["sample_id"]: sample for sample in state_samples}
    forecast_by_id = {sample["sample_id"]: sample for sample in forecast_samples}
    retained_state = state_out.recover()
    retained_forecast = forecast_out.recover()
    if "exp3" in experiments:
        retained_state = [
            {
                **row,
                "inclusion_probability": state_by_id[row["sample_id"]].get(
                    "inclusion_probability"
                ),
                "candidate_state_stratum_counts": state_by_id[row["sample_id"]].get(
                    "candidate_state_stratum_counts"
                ),
            }
            for row in retained_state
            if row.get("sample_hash") == state_hashes.get(row.get("sample_id"))
        ]
        state_out.rewrite(retained_state)
    if "exp5" in experiments:
        retained_forecast = [
            {
                **row,
                "inclusion_probability": forecast_by_id[row["sample_id"]].get(
                    "inclusion_probability"
                ),
                "global_candidate_joint_stratum_counts": forecast_by_id[
                    row["sample_id"]
                ].get("global_candidate_joint_stratum_counts"),
            }
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
        system = "Reconstruct only objectively knowable state from the supplied full-duel prefix."
        messages = [
            {"role": "user", "content": json.dumps(sample["prefix"], ensure_ascii=False)}
        ]
        _execute_probe_task(
            registry=registry,
            recovery_store=recovery_store,
            evaluator_id=evaluator_id,
            experiment="exp3",
            sample=sample,
            sample_hash=state_hashes[sample["sample_id"]],
            request={
                "system": system,
                "messages": messages,
                "tools": [state_tool],
                "evaluator_identity": evaluator_identity,
            },
            invoke=lambda system=system, messages=messages: provider.respond(
                system=system, messages=messages, tools=[state_tool]
            ),
            make_record=lambda turn, sample=sample: _state_result_record(
                sample, state_hashes[sample["sample_id"]], evaluator_identity, turn
            ),
            journal=state_out,
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
        system = (
            "Using only the acting player's visible information, estimate two "
            "separate probabilities: whether the opponent objectively has a legal "
            "response, and, conditional on such a response being available, whether "
            "the opponent will choose to use it."
        )
        messages = [
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
        ]
        _execute_probe_task(
            registry=registry,
            recovery_store=recovery_store,
            evaluator_id=evaluator_id,
            experiment="exp5",
            sample=sample,
            sample_hash=forecast_hashes[sample["sample_id"]],
            request={
                "system": system,
                "messages": messages,
                "tools": [forecast_tool],
                "evaluator_identity": evaluator_identity,
            },
            invoke=lambda system=system, messages=messages: provider.respond(
                system=system, messages=messages, tools=[forecast_tool]
            ),
            make_record=lambda turn, sample=sample: _forecast_result_record(
                sample, forecast_hashes[sample["sample_id"]], evaluator_identity, turn
            ),
            journal=forecast_out,
        )
    return {
        "evaluator_id": evaluator_id,
        "state_results": len(state_out.recover()),
        "forecast_results": len(forecast_out.recover()),
    }
