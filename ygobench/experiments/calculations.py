"""Statistical calculations for Experiments 1 through 5."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ygobench.bench.glicko2 import Glicko2, GlickoPlayer
from ygobench.experiments.identity import outcome_policy_id
from ygobench.experiments.statistics import (
    binary_probability_metrics,
    cluster_bootstrap,
    kaplan_meier,
    nearest_rank,
    proportion,
    rate,
)

METRIC_SCHEMA_VERSION = "2.0.0"


def _identity(outcome: dict[str, Any], seat: int) -> tuple[str, str, str]:
    policy = outcome_policy_id(outcome, seat)
    identities = outcome.get("policy_identities") or []
    model_config = policy
    if seat < len(identities) and isinstance(identities[seat], dict):
        model_config = str(identities[seat].get("model_configuration_id") or policy)
    agent = str(outcome.get("agents", ["unknown", "unknown"])[seat])
    return policy, model_config, agent


def _validation_invalid(row: dict[str, Any]) -> bool:
    validation = row.get("validation", {})
    return bool(
        not validation.get("valid", False)
        or validation.get("attempted_invalid")
        or validation.get("protocol_errors")
        or validation.get("agent_error")
        or validation.get("pre_engine_error")
        or validation.get("engine_rejection_error")
        or validation.get("engine_error")
    )


def _tool_calls(row: dict[str, Any]) -> int:
    return sum(len(turn.get("tool_calls", [])) for turn in row.get("trace", {}).get("turns", []))


def _pair_cluster_id(outcome: dict[str, Any]) -> str:
    policies = sorted(outcome_policy_id(outcome, seat) for seat in (0, 1))
    decks = sorted(str(deck) for deck in outcome["decks"])
    return (
        f"seed={outcome.get('seed')}|policies={'__vs__'.join(policies)}"
        f"|decks={'__vs__'.join(decks)}"
    )


def arena_metrics(
    outcomes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    strict = [outcome for outcome in outcomes if outcome.get("competitive_eligible", False)]
    rows_by_game_seat: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_game_seat[(str(row["game_id"]), int(row["player"]))].append(row)

    by_policy: dict[str, dict[str, Any]] = {}
    policy_game_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        for seat in (0, 1):
            policy, model_config, agent = _identity(outcome, seat)
            item = by_policy.setdefault(
                policy,
                {
                    "policy": policy,
                    "model_configuration_id": model_config,
                    "agent": agent,
                    "games_observed": 0,
                    "engine_completed": 0,
                    "strict_games": 0,
                    "strict_wins": 0,
                    "first_games": 0,
                    "first_wins": 0,
                    "second_games": 0,
                    "second_wins": 0,
                    "illegal_actions": 0,
                    "decisions": 0,
                    "tool_calls": 0,
                    "model_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "decision_seconds": 0.0,
                    "duel_seconds": 0.0,
                },
            )
            item["games_observed"] += 1
            item["engine_completed"] += int(outcome.get("termination") == "game_over")
            eligible = bool(outcome.get("competitive_eligible", False))
            item["strict_games"] += int(eligible)
            if eligible:
                won = int(outcome.get("winner") == seat)
                item["strict_wins"] += won
                position = "first" if seat == 0 else "second"
                item[f"{position}_games"] += 1
                item[f"{position}_wins"] += won
                policy_game_rows[policy].append(
                    {"cluster_id": _pair_cluster_id(outcome), "won": won}
                )
            decision_rows = rows_by_game_seat[(str(outcome["game_id"]), seat)]
            item["illegal_actions"] += sum(_validation_invalid(row) for row in decision_rows)
            item["decisions"] += len(decision_rows)
            item["tool_calls"] += sum(_tool_calls(row) for row in decision_rows)
            usage = outcome.get("model_usage_totals", {})
            item["model_calls"] += int((usage.get("model_calls") or [0, 0])[seat])
            item["input_tokens"] += int((usage.get("input_tokens") or [0, 0])[seat])
            item["output_tokens"] += int((usage.get("output_tokens") or [0, 0])[seat])
            item["decision_seconds"] += float((outcome.get("decision_seconds") or [0, 0])[seat])
            item["duel_seconds"] += float(outcome.get("elapsed_seconds", 0.0))

    policies = sorted(by_policy)
    ratings = {policy: GlickoPlayer() for policy in policies}
    rating_games = [
        outcome
        for outcome in strict
        if outcome_policy_id(outcome, 0) != outcome_policy_id(outcome, 1)
    ]
    if len(policies) >= 2 and rating_games:
        snapshots = dict(ratings)
        for policy in policies:
            games = []
            for outcome in rating_games:
                participants = [outcome_policy_id(outcome, seat) for seat in (0, 1)]
                if policy not in participants:
                    continue
                seat = participants.index(policy)
                score = 1.0 if outcome.get("winner") == seat else 0.0
                games.append((snapshots[participants[1 - seat]], score))
            ratings[policy] = Glicko2.update(snapshots[policy], games)
        glicko = {
            "status": "COMPUTED",
            "rated_games": len(rating_games),
            "rating_period": "all strict games in deterministic game_id order",
            "initial": {"rating": 1500.0, "deviation": 350.0, "volatility": 0.06},
            "tau": 0.5,
        }
    else:
        glicko = {
            "status": "INSUFFICIENT_OPPONENT_DIVERSITY",
            "rated_games": 0,
            "reason": "at least two exact policy identities must meet in strict games",
        }

    leaderboard = []
    for policy in policies:
        item = by_policy[policy]
        decisions = int(item["decisions"])
        observed = int(item["games_observed"])
        strict_games = int(item["strict_games"])
        first_games = int(item["first_games"])
        second_games = int(item["second_games"])
        item.update(
            {
                "overall_win_rate": rate(item["strict_wins"], strict_games),
                "overall_win_rate_ci95": proportion(item["strict_wins"], strict_games)[
                    "confidence_interval_95"
                ],
                "overall_win_rate_cluster_bootstrap": (
                    cluster_bootstrap(
                        policy_game_rows[policy],
                        cluster_key="cluster_id",
                        statistic=lambda sampled: rate(
                            sum(row["won"] for row in sampled), len(sampled)
                        ),
                    )
                    if include_bootstrap
                    else None
                ),
                "first_player_win_rate": rate(item["first_wins"], first_games),
                "second_player_win_rate": rate(item["second_wins"], second_games),
                "engine_completion_rate": rate(item["engine_completed"], observed),
                "illegal_action_rate": rate(item["illegal_actions"], decisions),
                "tool_calls_per_decision": rate(item["tool_calls"], decisions),
                "model_calls_per_decision": rate(item["model_calls"], decisions),
                "input_tokens_per_decision": rate(item["input_tokens"], decisions),
                "output_tokens_per_decision": rate(item["output_tokens"], decisions),
                "mean_decision_latency_seconds": rate(item["decision_seconds"], decisions),
                "mean_duel_latency_seconds": rate(item["duel_seconds"], observed),
                "glicko2": (
                    {
                        "mu": ratings[policy].rating,
                        "rd_phi": ratings[policy].deviation,
                        "volatility": ratings[policy].volatility,
                    }
                    if glicko["status"] == "COMPUTED"
                    else None
                ),
            }
        )
        leaderboard.append(item)

    pair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        key = _pair_cluster_id(outcome)
        pair_groups[key].append(outcome)
    paired_units = []
    for key, games in sorted(pair_groups.items()):
        directions = Counter(
            tuple(outcome_policy_id(game, seat) for seat in (0, 1)) for game in games
        )
        paired_units.append(
            {
                "paired_unit_id": key,
                "games_observed": len(games),
                "strict_games": sum(game.get("competitive_eligible", False) for game in games),
                "seat_directions": {
                    "__vs__".join(direction): count
                    for direction, count in sorted(directions.items())
                },
                "reciprocal_pair_complete": len(directions) >= 2
                or (len(directions) == 1 and len(games) >= 2),
            }
        )

    matchup: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for outcome in strict:
        for seat in (0, 1):
            policy = outcome_policy_id(outcome, seat)
            key = (policy, str(outcome["decks"][seat]), str(outcome["decks"][1 - seat]))
            matchup[key]["games"] += 1
            matchup[key]["wins"] += int(outcome.get("winner") == seat)
            matchup[key]["first_games"] += int(seat == 0)
            matchup[key]["first_wins"] += int(seat == 0 and outcome.get("winner") == seat)
            matchup[key]["second_games"] += int(seat == 1)
            matchup[key]["second_wins"] += int(seat == 1 and outcome.get("winner") == seat)

    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "experiment": 1,
        "status": "COMPLETED" if outcomes else "INSUFFICIENT_DATA",
        "protocol": "full_duel_arena",
        "games_observed": len(outcomes),
        "strict_games": len(strict),
        "eligibility_exclusions": dict(
            Counter(
                outcome.get("termination", "unknown")
                for outcome in outcomes
                if not outcome.get("competitive_eligible", False)
            )
        ),
        "glicko2": glicko,
        "leaderboard": leaderboard,
        "paired_units": paired_units,
        "paired_unit_completion": proportion(
            sum(unit["reciprocal_pair_complete"] for unit in paired_units), len(paired_units)
        ),
        "model_deck_matchup_matrix": [
            {
                "policy": key[0],
                "deck": key[1],
                "opponent_deck": key[2],
                **counts,
                "win_rate": rate(counts["wins"], counts["games"]),
                "first_player_win_rate": rate(counts["first_wins"], counts["first_games"]),
                "second_player_win_rate": rate(counts["second_wins"], counts["second_games"]),
            }
            for key, counts in sorted(matchup.items())
        ],
        "side_swapped_win_rate": {
            "overall_by_policy": [
                {
                    "policy": item["policy"],
                    "wins": item["strict_wins"],
                    "games": item["strict_games"],
                    "win_rate": item["overall_win_rate"],
                }
                for item in leaderboard
            ],
            "by_seat": [
                {
                    "seat": seat,
                    **proportion(
                        sum(outcome.get("winner") == seat for outcome in strict), len(strict)
                    ),
                }
                for seat in (0, 1)
            ],
        },
        "engine_completion": {
            **proportion(
                sum(outcome.get("termination") == "game_over" for outcome in outcomes),
                len(outcomes),
            )
        },
    }


def _turn_units(
    rows: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("legal", {}).get("requires_model_reasoning"):
            grouped[(str(row["game_id"]), int(row["turn"]), int(row["player"]))].append(row)
    units = []
    for (game_id, turn, player), unit_rows in sorted(grouped.items()):
        outcome = outcomes[game_id]
        unit_rows.sort(key=lambda row: int(row["decision_index"]))
        event_at = next(
            (index + 1 for index, row in enumerate(unit_rows) if _validation_invalid(row)), None
        )
        units.append(
            {
                "game_id": game_id,
                "turn": turn,
                "player": player,
                "policy": outcome_policy_id(outcome, player),
                "agent": outcome["agents"][player],
                "deck": outcome["decks"][player],
                "seat": player,
                "decision_length": len(unit_rows),
                "mean_legal_branching": sum(
                    int(row["legal"].get("legal_action_count", 0)) for row in unit_rows
                )
                / len(unit_rows),
                "max_chain_depth": max(
                    len(row.get("observation", {}).get("chain", [])) for row in unit_rows
                ),
                "time": event_at or len(unit_rows),
                "event_at": event_at,
                "event_observed": event_at is not None,
                "censor_reason": None if event_at is not None else "turn_ended_without_error",
                "termination": outcome.get("termination"),
            }
        )
    return units


def long_chain_metrics(
    rows: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    outcome_map = {str(outcome["game_id"]): outcome for outcome in outcomes}
    units = _turn_units(rows, outcome_map)
    cutpoints = tuple(
        nearest_rank((unit["decision_length"] for unit in units), q) or 0.0
        for q in (0.25, 0.5, 0.75)
    )

    def quartile(value: int) -> str:
        return (
            "Q1"
            if value <= cutpoints[0]
            else "Q2"
            if value <= cutpoints[1]
            else "Q3"
            if value <= cutpoints[2]
            else "Q4"
        )

    km = []
    for policy in sorted({unit["policy"] for unit in units}):
        selected = [unit for unit in units if unit["policy"] == policy]
        analysis = kaplan_meier(selected, horizon=20)
        analysis["policy"] = policy
        analysis["survival_at_20_cluster_bootstrap"] = (
            cluster_bootstrap(
                selected,
                cluster_key="game_id",
                statistic=lambda sampled: (
                    (kaplan_meier(sampled, horizon=20).get("fixed_window") or {})
                    .get("survival_probability", {})
                    .get("estimate")
                ),
            )
            if include_bootstrap
            else None
        )
        km.append(analysis)

    game_complexity: dict[str, int] = defaultdict(int)
    for unit in units:
        game_complexity[unit["game_id"]] = max(
            game_complexity[unit["game_id"]], int(unit["decision_length"])
        )
    game_decisions: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    depth: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    depth_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("legal", {}).get("requires_model_reasoning"):
            continue
        outcome = outcome_map[str(row["game_id"])]
        policy = outcome_policy_id(outcome, int(row["player"]))
        game_decisions[(str(row["game_id"]), policy)]["decisions"] += 1
        game_decisions[(str(row["game_id"]), policy)]["invalid"] += int(_validation_invalid(row))
        position = int(row["decision_index"])
        label = (
            "1-10"
            if position <= 10
            else "11-20"
            if position <= 20
            else "21-40"
            if position <= 40
            else "41+"
        )
        depth[(policy, label)]["decisions"] += 1
        depth[(policy, label)]["invalid"] += int(_validation_invalid(row))
        depth_rows[(policy, label)].append(
            {
                "game_id": str(row["game_id"]),
                "invalid": int(_validation_invalid(row)),
            }
        )

    complexity: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    complexity_game_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for outcome in outcomes:
        game_id = str(outcome["game_id"])
        if game_id not in game_complexity:
            continue
        bucket = quartile(game_complexity[game_id])
        for seat in (0, 1):
            policy = outcome_policy_id(outcome, seat)
            entry = complexity[(policy, bucket)]
            entry["games_observed"] += 1
            entry["engine_completed"] += int(outcome.get("termination") == "game_over")
            entry["strict_games"] += int(outcome.get("competitive_eligible", False))
            entry["strict_wins"] += int(
                outcome.get("competitive_eligible", False) and outcome.get("winner") == seat
            )
            entry.update(game_decisions[(game_id, policy)])
            game_counts = game_decisions[(game_id, policy)]
            complexity_game_rows[(policy, bucket)].append(
                {
                    "cluster_id": _pair_cluster_id(outcome),
                    "eligible": int(outcome.get("competitive_eligible", False)),
                    "won": int(
                        outcome.get("competitive_eligible", False) and outcome.get("winner") == seat
                    ),
                    "engine_completed": int(outcome.get("termination") == "game_over"),
                    "decisions": int(game_counts["decisions"]),
                    "invalid": int(game_counts["invalid"]),
                }
            )

    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "experiment": 2,
        "status": "COMPLETED" if units else "INSUFFICIENT_DATA",
        "preregistered_survival_depth": 20,
        "trajectory_variables": [
            "turn_decision_length",
            "global_decision_depth",
            "legal_branching_size",
            "chain_depth",
        ],
        "turn_decision_length_quartile_cutpoints": {
            "method": "nearest_rank",
            "q1": cutpoints[0],
            "q2": cutpoints[1],
            "q3": cutpoints[2],
        },
        "event_censor_records": units,
        "kaplan_meier": km,
        "illegal_action_rate_by_decision_depth": [
            {
                "policy": key[0],
                "depth_bin": key[1],
                **counts,
                "illegal_action_rate": rate(counts["invalid"], counts["decisions"]),
                "illegal_action_rate_ci95": proportion(counts["invalid"], counts["decisions"])[
                    "confidence_interval_95"
                ],
                "illegal_action_rate_cluster_bootstrap": (
                    cluster_bootstrap(
                        depth_rows[key],
                        cluster_key="game_id",
                        statistic=lambda sampled: rate(
                            sum(row["invalid"] for row in sampled), len(sampled)
                        ),
                    )
                    if include_bootstrap
                    else None
                ),
            }
            for key, counts in sorted(depth.items())
        ],
        "complexity_stratified": [
            {
                "policy": key[0],
                "quartile": key[1],
                **counts,
                "win_rate": rate(counts["strict_wins"], counts["strict_games"]),
                "win_rate_ci95": proportion(counts["strict_wins"], counts["strict_games"])[
                    "confidence_interval_95"
                ],
                "completion_rate": rate(counts["engine_completed"], counts["games_observed"]),
                "completion_rate_ci95": proportion(
                    counts["engine_completed"], counts["games_observed"]
                )["confidence_interval_95"],
                "illegal_action_rate": rate(counts["invalid"], counts["decisions"]),
                "illegal_action_rate_ci95": proportion(counts["invalid"], counts["decisions"])[
                    "confidence_interval_95"
                ],
                "cluster_bootstrap": (
                    {
                        "win_rate": cluster_bootstrap(
                            complexity_game_rows[key],
                            cluster_key="cluster_id",
                            statistic=lambda sampled: rate(
                                sum(row["won"] for row in sampled),
                                sum(row["eligible"] for row in sampled),
                            ),
                        ),
                        "completion_rate": cluster_bootstrap(
                            complexity_game_rows[key],
                            cluster_key="cluster_id",
                            statistic=lambda sampled: rate(
                                sum(row["engine_completed"] for row in sampled), len(sampled)
                            ),
                        ),
                        "illegal_action_rate": cluster_bootstrap(
                            complexity_game_rows[key],
                            cluster_key="cluster_id",
                            statistic=lambda sampled: rate(
                                sum(row["invalid"] for row in sampled),
                                sum(row["decisions"] for row in sampled),
                            ),
                        ),
                    }
                    if include_bootstrap
                    else None
                ),
            }
            for key, counts in sorted(complexity.items())
        ],
    }


def _primary_error(row: dict[str, Any]) -> str | None:
    validation = row.get("validation", {})
    recorded = validation.get("primary_error_code")
    standard = {
        "wrong_responder",
        "missing_tool_call",
        "multiple_tool_call",
        "invalid_index",
        "stale_action_or_index",
        "invalid_target",
        "invalid_material_selection",
        "invalid_position_or_place",
        "rule_or_timing_violation",
        "engine_or_protocol_parsing_error",
        "unclassified_invalid_action",
    }
    if recorded in standard:
        return str(recorded)
    legacy = {
        "provider_api_failure": "engine_or_protocol_parsing_error",
        "multiple_game_actions": "multiple_tool_call",
        "missing_required_tool": "missing_tool_call",
        "malformed_arguments": "engine_or_protocol_parsing_error",
        "engine_rejection": "rule_or_timing_violation",
        "response_protocol_failure": "engine_or_protocol_parsing_error",
    }
    if recorded in legacy:
        return legacy[str(recorded)]
    trace = row.get("trace", {})
    raw_errors = [str(value) for value in validation.get("protocol_errors", [])]
    raw_errors.extend(
        str(value)
        for value in (validation.get("agent_error"), validation.get("pre_engine_error"))
        if value
    )
    raw = " ".join(raw_errors).lower()
    if trace.get("provider_failure_type") or trace.get("provider_transport_attempts"):
        return "provider_api_failure"
    if "multiple" in raw and ("tool" in raw or "response" in raw):
        return "multiple_tool_call"
    if "missing" in raw and ("tool" in raw or "responder" in raw):
        return "missing_tool_call"
    expected = str(row.get("legal", {}).get("expected_responder", ""))
    attempted = row.get("attempted_action") or {}
    if attempted.get("tool") and attempted.get("tool") != expected:
        return "wrong_responder"
    if validation.get("pre_engine_error"):
        if "argument_validation" in raw:
            return "engine_or_protocol_parsing_error"
        if "stale" in raw:
            return "stale_action_or_index"
        if expected in {"select_place", "select_position"}:
            return "invalid_position_or_place"
        if expected in {"select_sum", "select_tribute"}:
            return "invalid_material_selection"
        if expected in {"select_card", "select_unselect_card", "select_card_codes"}:
            return "invalid_target"
        if any(key in attempted.get("arguments", {}) for key in ("index", "indices")):
            return "invalid_index"
        return "rule_or_timing_violation"
    if validation.get("engine_rejection_error") or validation.get("engine_error"):
        return "rule_or_timing_violation"
    if validation.get("terminal_model_failure"):
        return "unclassified_invalid_action"
    if validation.get("protocol_errors") or validation.get("agent_error"):
        return "engine_or_protocol_parsing_error"
    if _validation_invalid(row):
        return "unclassified_invalid_action"
    return None


def _secondary_errors(row: dict[str, Any]) -> set[str]:
    validation = row.get("validation", {})
    expected = str(row.get("legal", {}).get("expected_responder", ""))
    attempted = row.get("attempted_action") or {}
    arguments = attempted.get("arguments", {}) if isinstance(attempted, dict) else {}
    raw = " ".join(
        str(value or "")
        for value in (
            validation.get("agent_error"),
            validation.get("pre_engine_error"),
            validation.get("engine_rejection_error"),
            validation.get("engine_error"),
            *validation.get("protocol_errors", []),
        )
    ).lower()
    labels = set()
    if not attempted or "missing" in raw and "tool" in raw:
        labels.add("missing_tool_call")
    if attempted.get("tool") and expected and attempted.get("tool") != expected:
        labels.add("wrong_responder")
    if "multiple" in raw:
        labels.add("multiple_tool_call")
    if "index" in raw or (
        "exact_legal_set" in raw and any(key in arguments for key in ("index", "indices"))
    ):
        labels.add("invalid_index")
    if "stale" in raw:
        labels.add("stale_action_or_index")
    if "target" in raw:
        labels.add("invalid_target")
    if validation.get("pre_engine_error") and expected in {
        "select_card",
        "select_unselect_card",
        "select_card_codes",
    }:
        labels.add("invalid_target")
    if validation.get("pre_engine_error") and expected in {"select_sum", "select_tribute"}:
        labels.add("invalid_material_selection")
    if "position" in raw or "place" in raw:
        labels.add("invalid_position_or_place")
    if validation.get("pre_engine_error") and expected in {"select_place", "select_position"}:
        labels.add("invalid_position_or_place")
    if validation.get("engine_rejection_error") or validation.get("engine_error"):
        labels.add("rule_or_timing_violation")
    if validation.get("protocol_errors") or "argument_validation" in raw:
        labels.add("engine_or_protocol_parsing_error")
    if not labels and _validation_invalid(row):
        labels.add("unclassified_invalid_action")
    return labels


def execution_metrics(
    rows: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    outcome_map = {str(outcome["game_id"]): outcome for outcome in outcomes}
    by_policy: dict[str, Counter[str]] = defaultdict(Counter)
    policy_decisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels: dict[str, str] = {}
    primary: dict[str, Counter[str]] = defaultdict(Counter)
    secondary: dict[str, Counter[str]] = defaultdict(Counter)
    error_evidence: list[dict[str, Any]] = []
    by_responder: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    by_complexity: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    responder_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    complexity_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    def complexity(responder: str) -> str:
        if responder in {"select_yesno", "select_option", "rock_paper_scissors"}:
            return "simple_binary_or_option"
        if responder in {"select_card", "select_unselect_card", "select_card_codes"}:
            return "target_selection"
        if responder in {"select_sum", "select_tribute"}:
            return "material_or_subset_selection"
        if responder == "select_chain":
            return "chain_response"
        if responder in {"select_idlecmd", "select_battlecmd", "select_effectyn"}:
            return "timing_sensitive_command"
        if responder in {"select_place", "select_position"}:
            return "position_or_placement"
        return "other"

    for row in rows:
        if not row.get("legal", {}).get("requires_model_reasoning"):
            continue
        outcome = outcome_map[str(row["game_id"])]
        seat = int(row["player"])
        policy = outcome_policy_id(outcome, seat)
        labels[policy] = str(outcome["agents"][seat])
        item = by_policy[policy]
        validation = row.get("validation", {})
        invalid = int(_validation_invalid(row))
        attempts = int(validation.get("model_action_attempts", 1) or 1)
        item["decisions"] += 1
        item["invalid"] += invalid
        item["first_attempt_successes"] += int(not invalid)
        item["action_retries"] += max(0, attempts - 1)
        item["retry_recovered"] += int(validation.get("recovery") == "corrected_retry")
        item["retry_exhausted"] += int(bool(validation.get("terminal_model_failure")))
        item["fallbacks"] += int(bool(row.get("trace", {}).get("fallback")))
        item["deterministic_engine_fallbacks"] += int(
            validation.get("recovery") == "deterministic_fallback_after_engine_error"
        )
        item["exact_legal_rejections"] += int(bool(validation.get("pre_engine_error")))
        item["engine_rejections"] += int(
            bool(validation.get("engine_rejection_error") or validation.get("engine_error"))
        )
        item["tool_calls"] += _tool_calls(row)
        item["model_calls"] += len(row.get("trace", {}).get("turns", []))
        item["inspect_card_calls"] += sum(
            call.get("name") == "inspect_card"
            for turn in row.get("trace", {}).get("turns", [])
            for call in turn.get("tool_calls", [])
        )
        policy_decisions[policy].append({"game_id": str(row["game_id"]), "invalid": invalid})
        code = _primary_error(row)
        if code:
            primary[policy][code] += 1
        secondary_codes = _secondary_errors(row)
        for secondary_code in secondary_codes:
            secondary[policy][secondary_code] += 1
        if invalid:
            error_evidence.append(
                {
                    "game_id": str(row["game_id"]),
                    "decision_id": row.get("decision_id"),
                    "decision_index": row.get("decision_index"),
                    "policy": policy,
                    "expected_responder": row.get("legal", {}).get("expected_responder"),
                    "primary_error": code,
                    "secondary_errors": sorted(secondary_codes),
                    "raw_error_evidence": {
                        key: validation.get(key)
                        for key in (
                            "agent_error",
                            "pre_engine_error",
                            "engine_rejection_error",
                            "engine_error",
                            "protocol_errors",
                        )
                        if validation.get(key)
                    },
                }
            )
        responder = str(row.get("legal", {}).get("expected_responder", "unknown"))
        for group in (
            by_responder[(policy, responder)],
            by_complexity[(policy, complexity(responder))],
        ):
            group["decisions"] += 1
            group["invalid"] += invalid
        responder_rows[(policy, responder)].append(
            {"game_id": str(row["game_id"]), "invalid": invalid}
        )
        complexity_rows[(policy, complexity(responder))].append(
            {"game_id": str(row["game_id"]), "invalid": invalid}
        )

    completion: dict[str, Counter[str]] = defaultdict(Counter)
    terminations: dict[str, Counter[str]] = defaultdict(Counter)
    for outcome in outcomes:
        for seat in (0, 1):
            policy = outcome_policy_id(outcome, seat)
            completion[policy]["games"] += 1
            completion[policy]["engine_completed"] += int(outcome.get("termination") == "game_over")
            terminations[policy][str(outcome.get("termination", "unknown"))] += 1

    enriched = []
    for policy, item in sorted(by_policy.items()):
        decisions = item["decisions"]
        enriched.append(
            {
                "policy": policy,
                "agent": labels[policy],
                **item,
                "first_attempt_success_rate": rate(item["first_attempt_successes"], decisions),
                "illegal_action_rate": rate(item["invalid"], decisions),
                "illegal_action_rate_ci95": proportion(item["invalid"], decisions)[
                    "confidence_interval_95"
                ],
                "illegal_action_rate_cluster_bootstrap": cluster_bootstrap(
                    policy_decisions[policy],
                    cluster_key="game_id",
                    statistic=lambda sampled: rate(
                        sum(row["invalid"] for row in sampled), len(sampled)
                    ),
                )
                if include_bootstrap
                else None,
                "action_retries_per_100_decisions": rate(100 * item["action_retries"], decisions),
                "retry_recovery_rate": rate(item["retry_recovered"], item["invalid"]),
                "retry_exhausted_rate": rate(item["retry_exhausted"], decisions),
                "fallbacks_per_100_decisions": rate(100 * item["fallbacks"], decisions),
                "deterministic_engine_fallbacks_per_100_decisions": rate(
                    100 * item["deterministic_engine_fallbacks"], decisions
                ),
                "exact_legal_rejection_rate": rate(item["exact_legal_rejections"], decisions),
                "engine_rejection_rate": rate(item["engine_rejections"], decisions),
                "tool_calls_per_decision": rate(item["tool_calls"], decisions),
                "model_calls_per_decision": rate(item["model_calls"], decisions),
                "inspect_card_calls_per_decision": rate(item["inspect_card_calls"], decisions),
                "engine_completion_rate": rate(
                    completion[policy]["engine_completed"], completion[policy]["games"]
                ),
                "termination_counts": dict(sorted(terminations[policy].items())),
            }
        )
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "experiment": 4,
        "status": "COMPLETED" if by_policy else "INSUFFICIENT_DATA",
        "denominator_policy": {
            "decision_level": "non-trivial engine decisions",
            "attempt_level": "initial action plus correction submissions",
        },
        "by_agent": enriched,
        "primary_error_taxonomy": [
            {
                "policy": key,
                "agent": labels.get(key),
                "counts": dict(sorted(value.items())),
                "rates_per_decision": {
                    code: rate(count, by_policy[key]["decisions"])
                    for code, count in sorted(value.items())
                },
            }
            for key, value in sorted(primary.items())
        ],
        "error_taxonomy": [
            {
                "policy": key,
                "agent": labels.get(key),
                "counts": dict(sorted(value.items())),
                "rates_per_decision": {
                    code: rate(count, by_policy[key]["decisions"])
                    for code, count in sorted(value.items())
                },
            }
            for key, value in sorted(secondary.items())
        ],
        "error_evidence": error_evidence,
        "termination_reasons": {
            policy: dict(sorted(counts.items())) for policy, counts in sorted(terminations.items())
        },
        "invalid_action_rate_by_responder_type": [
            {
                "policy": key[0],
                "responder": key[1],
                **value,
                "invalid_rate": rate(value["invalid"], value["decisions"]),
                "invalid_rate_ci95": proportion(value["invalid"], value["decisions"])[
                    "confidence_interval_95"
                ],
                "invalid_rate_cluster_bootstrap": (
                    cluster_bootstrap(
                        responder_rows[key],
                        cluster_key="game_id",
                        statistic=lambda sampled: rate(
                            sum(row["invalid"] for row in sampled), len(sampled)
                        ),
                    )
                    if include_bootstrap
                    else None
                ),
            }
            for key, value in sorted(by_responder.items())
        ],
        "invalid_action_rate_by_rule_complexity": [
            {
                "policy": key[0],
                "complexity": key[1],
                **value,
                "invalid_rate": rate(value["invalid"], value["decisions"]),
                "invalid_rate_ci95": proportion(value["invalid"], value["decisions"])[
                    "confidence_interval_95"
                ],
                "invalid_rate_cluster_bootstrap": (
                    cluster_bootstrap(
                        complexity_rows[key],
                        cluster_key="game_id",
                        statistic=lambda sampled: rate(
                            sum(row["invalid"] for row in sampled), len(sampled)
                        ),
                    )
                    if include_bootstrap
                    else None
                ),
            }
            for key, value in sorted(by_complexity.items())
        ],
    }


def _normal(value: Any) -> Any:
    return sorted(value) if isinstance(value, list) else value


def _f1(truth: list[str], prediction: list[str]) -> tuple[int, int, int, float | None]:
    left, right = Counter(truth), Counter(prediction)
    true_positive = sum((left & right).values())
    false_positive = sum((right - left).values())
    false_negative = sum((left - right).values())
    return (
        true_positive,
        false_positive,
        false_negative,
        rate(2 * true_positive, 2 * true_positive + false_positive + false_negative),
    )


def _state_schema_errors(row: dict[str, Any]) -> list[str]:
    truth = row.get("ground_truth")
    prediction = row.get("prediction")
    if not isinstance(truth, dict):
        return ["missing_ground_truth"]
    if not isinstance(prediction, dict):
        return ["missing_or_non_object_prediction"]
    errors = []
    for key, expected in truth.items():
        if key not in prediction:
            errors.append(f"missing_field:{key}")
        elif isinstance(expected, list) and not isinstance(prediction[key], list):
            errors.append(f"wrong_type:{key}:array")
        elif isinstance(expected, int) and (
            not isinstance(prediction[key], int) or isinstance(prediction[key], bool)
        ):
            errors.append(f"wrong_type:{key}:integer")
        elif isinstance(expected, str) and not isinstance(prediction[key], str):
            errors.append(f"wrong_type:{key}:string")
    return errors


def state_metrics(rows: list[dict[str, Any]], *, include_bootstrap: bool = True) -> dict[str, Any]:
    schema_errors = {index: _state_schema_errors(row) for index, row in enumerate(rows)}
    valid = [row for index, row in enumerate(rows) if not schema_errors[index]]
    scalar_correct = scalar_total = true_positive = false_positive = false_negative = 0
    valid_joint = 0
    per_field: dict[str, Counter[str]] = defaultdict(Counter)
    for row in valid:
        truth, prediction = row["ground_truth"], row["prediction"]
        exact = True
        for key, expected in truth.items():
            predicted = prediction[key]
            if isinstance(expected, list):
                tp, fp, fn, _ = _f1(expected, [str(item) for item in predicted])
                true_positive += tp
                false_positive += fp
                false_negative += fn
                per_field[key]["tp"] += tp
                per_field[key]["fp"] += fp
                per_field[key]["fn"] += fn
                exact &= _normal(predicted) == _normal(expected)
            else:
                correct = int(predicted == expected)
                scalar_total += 1
                scalar_correct += correct
                per_field[key]["total"] += 1
                per_field[key]["correct"] += correct
                exact &= bool(correct)
        valid_joint += int(exact)

    end_to_end_joint = valid_joint
    e2e_scalar_correct = e2e_scalar_total = 0
    e2e_tp = e2e_fp = e2e_fn = 0
    e2e_per_field: dict[str, Counter[str]] = defaultdict(Counter)
    progress = defaultdict(Counter)
    complexity = {
        key: defaultdict(Counter)
        for key in (
            "chain_depth",
            "zone_transition_events",
            "historical_action_count",
            "historical_public_activation_count",
        )
    }
    for index, row in enumerate(rows):
        truth = row.get("ground_truth")
        if not isinstance(truth, dict):
            continue
        prediction = row.get("prediction") if not schema_errors[index] else {}
        prediction = prediction if isinstance(prediction, dict) else {}
        for key, expected in truth.items():
            predicted = prediction.get(key)
            if isinstance(expected, list):
                predicted_list = predicted if isinstance(predicted, list) else []
                tp, fp, fn, _ = _f1(expected, [str(item) for item in predicted_list])
                e2e_tp += tp
                e2e_fp += fp
                e2e_fn += fn
                e2e_per_field[key]["tp"] += tp
                e2e_per_field[key]["fp"] += fp
                e2e_per_field[key]["fn"] += fn
            else:
                correct = int(predicted == expected)
                e2e_scalar_total += 1
                e2e_scalar_correct += correct
                e2e_per_field[key]["total"] += 1
                e2e_per_field[key]["correct"] += correct
        exact = not schema_errors[index] and all(
            _normal(prediction.get(key)) == _normal(expected) for key, expected in truth.items()
        )
        progress_value = float(row.get("trajectory_progress", 0))
        progress_bin = (
            "0-25%"
            if progress_value <= 0.25
            else "25-50%"
            if progress_value <= 0.5
            else "50-75%"
            if progress_value <= 0.75
            else "75-100%"
        )
        progress[progress_bin]["samples"] += 1
        progress[progress_bin]["joint_correct"] += int(exact)
        for dimension, groups in complexity.items():
            value = row.get("state_complexity", {}).get(dimension, "unknown")
            if dimension in {"zone_transition_events", "historical_public_activation_count"}:
                label = (
                    "0" if value == 0 else "1-2" if isinstance(value, int) and value <= 2 else "3+"
                )
            elif dimension == "historical_action_count":
                label = (
                    "0-25"
                    if isinstance(value, int) and value <= 25
                    else "26-100"
                    if isinstance(value, int) and value <= 100
                    else "101+"
                )
            else:
                label = str(value)
            groups[label]["samples"] += 1
            groups[label]["joint_correct"] += int(exact)
    field_rows = []
    set_field_f1: list[float] = []
    valid_set_field_f1: list[float] = []
    for key, counts in sorted(e2e_per_field.items()):
        if "total" in counts:
            field_rows.append(
                {
                    "field": key,
                    "type": "scalar",
                    **counts,
                    "accuracy": rate(counts["correct"], counts["total"]),
                    "valid_only_accuracy": rate(per_field[key]["correct"], per_field[key]["total"]),
                }
            )
        else:
            field_f1 = rate(2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"])
            valid_field_f1 = rate(
                2 * per_field[key]["tp"],
                2 * per_field[key]["tp"] + per_field[key]["fp"] + per_field[key]["fn"],
            )
            if field_f1 is not None:
                set_field_f1.append(field_f1)
            if valid_field_f1 is not None:
                valid_set_field_f1.append(valid_field_f1)
            field_rows.append(
                {
                    "field": key,
                    "type": "set",
                    **counts,
                    "f1": field_f1,
                    "valid_only_f1": valid_field_f1,
                }
            )
    error_counts = Counter(error for errors in schema_errors.values() for error in errors)

    def end_to_end_jga(sampled: list[dict[str, Any]]) -> float | None:
        correct = 0
        for row in sampled:
            if _state_schema_errors(row):
                continue
            truth, prediction = row["ground_truth"], row["prediction"]
            correct += int(
                all(_normal(prediction[key]) == _normal(value) for key, value in truth.items())
            )
        return rate(correct, len(sampled))

    bootstrap_rows = [{**row, "game_id": str(row.get("game_id", "unknown"))} for row in rows]
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "experiment": 3,
        "status": "COMPLETED" if rows else "INSUFFICIENT_DATA",
        "samples": len(rows),
        "valid_samples": len(valid),
        "invalid_samples": len(rows) - len(valid),
        "jga": rate(end_to_end_joint, len(rows)),
        "jga_ci95": proportion(end_to_end_joint, len(rows))["confidence_interval_95"],
        "jga_cluster_bootstrap": (
            cluster_bootstrap(bootstrap_rows, cluster_key="game_id", statistic=end_to_end_jga)
            if include_bootstrap
            else None
        ),
        "valid_only_jga": rate(valid_joint, len(valid)),
        "slot_accuracy": rate(e2e_scalar_correct, e2e_scalar_total),
        "valid_only_slot_accuracy": rate(scalar_correct, scalar_total),
        "slot_f1": {
            "micro_f1": rate(2 * e2e_tp, 2 * e2e_tp + e2e_fp + e2e_fn),
            "macro_f1": rate(sum(set_field_f1), len(set_field_f1)),
            "valid_only_micro_f1": rate(
                2 * true_positive, 2 * true_positive + false_positive + false_negative
            ),
            "valid_only_macro_f1": rate(sum(valid_set_field_f1), len(valid_set_field_f1)),
        },
        "per_field": field_rows,
        "schema_failure_rate": rate(len(rows) - len(valid), len(rows)),
        "schema_error_counts": dict(sorted(error_counts.items())),
        "parse_failure_rate": rate(
            sum(not isinstance(row.get("prediction"), dict) for row in rows), len(rows)
        ),
        "jga_by_trajectory_progress": [
            {"bin": key, **value, "jga": rate(value["joint_correct"], value["samples"])}
            for key, value in sorted(progress.items())
        ],
        "jga_by_state_complexity": [
            {"bin": key, **value, "jga": rate(value["joint_correct"], value["samples"])}
            for key, value in sorted(complexity["chain_depth"].items())
        ],
        "jga_by_state_complexity_dimension": {
            dimension: [
                {"bin": key, **value, "jga": rate(value["joint_correct"], value["samples"])}
                for key, value in sorted(groups.items())
            ]
            for dimension, groups in complexity.items()
        },
    }


def forecast_metrics(
    rows: list[dict[str, Any]],
    truth_key: str,
    probability_key: str,
    *,
    include_bootstrap: bool = True,
) -> dict[str, Any]:
    result = binary_probability_metrics(rows, truth_key=truth_key, probability_key=probability_key)
    bootstrap_rows = [{**row, "game_id": str(row.get("game_id", "unknown"))} for row in rows]
    result["cluster_bootstrap"] = (
        {
            metric: cluster_bootstrap(
                bootstrap_rows,
                cluster_key="game_id",
                statistic=lambda sampled, key=metric: binary_probability_metrics(
                    sampled, truth_key=truth_key, probability_key=probability_key
                ).get(key),
            )
            for metric in ("brier_score", "auprc", "auroc", "ece_10_equal_width")
        }
        if include_bootstrap
        else None
    )
    weighted_rows = [row for row in rows if row.get("inclusion_probability")]
    if weighted_rows:
        prepared = [
            {**row, "inverse_probability_weight": 1 / float(row["inclusion_probability"])}
            for row in weighted_rows
        ]
        result["natural_distribution_weighted"] = binary_probability_metrics(
            prepared,
            truth_key=truth_key,
            probability_key=probability_key,
            weight_key="inverse_probability_weight",
        )
    else:
        result["natural_distribution_weighted"] = None
    return result


def forecast_strata(
    rows: list[dict[str, Any]], truth_key: str, probability_key: str
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.get("strata", {}).items():
            if key in {
                "hand_size",
                "opponent_set_card_count",
                "chain_depth",
                "hidden_information_density",
            }:
                value = "0" if int(value) == 0 else "1-2" if int(value) <= 2 else "3+"
            groups[key][str(value)].append(row)
    return {
        key: [
            {
                "stratum": value,
                **forecast_metrics(items, truth_key, probability_key, include_bootstrap=False),
            }
            for value, items in sorted(values.items())
        ]
        for key, values in sorted(groups.items())
    }
