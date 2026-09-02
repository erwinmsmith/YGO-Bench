"""PDF-aligned metrics for full-duel Experiments 1 through 5."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Any

from ygobench.bench.glicko2 import Glicko2, GlickoPlayer
from ygobench.experiments.io import JsonlJournal, atomic_write_json, read_json
from ygobench.experiments.probes import probe_evaluator_id


def _game_dirs(run_dir: Path) -> list[Path]:
    return sorted(path.parent for path in (run_dir / "games").glob("*/manifest.json"))


def _rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _policy_id(agent: str) -> str:
    """Remove execution-seat/profile prefixes without collapsing model identity."""
    parts = agent.split(":")
    return ":".join(parts[-2:]) if len(parts) >= 3 else agent


def _percentile(values: list[int | float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, ceil(fraction * len(ordered)) - 1)])


def _bucket(value: int, cutpoints: tuple[float, float, float]) -> str:
    q1, q2, q3 = cutpoints
    return "Q1" if value <= q1 else "Q2" if value <= q2 else "Q3" if value <= q3 else "Q4"


def _outcomes_and_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes, rows = [], []
    for game_dir in _game_dirs(run_dir):
        outcome = read_json(game_dir / "outcome.json")
        if outcome:
            outcomes.append(outcome)
        rows.extend(JsonlJournal(game_dir / "trajectory.jsonl").recover())
    return outcomes, rows


def _outcome_map(outcomes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(outcome["game_id"]): outcome for outcome in outcomes}


def _tool_calls(row: dict[str, Any]) -> int:
    return sum(len(turn.get("tool_calls", [])) for turn in row.get("trace", {}).get("turns", []))


def _arena_metrics(outcomes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = [outcome for outcome in outcomes if outcome.get("competitive_eligible", False)]
    illegal_by_game: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    tool_by_game: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        player = int(row["player"])
        illegal_by_game[row["game_id"]][player] += int(not row.get("validation", {}).get("valid", False))
        tool_by_game[row["game_id"]][player] += _tool_calls(row)

    seats = []
    for seat in (0, 1):
        games = strict
        wins = sum(outcome.get("winner") == seat for outcome in games)
        seats.append(
            {
                "seat": seat,
                "metric": "side_swapped_win_rate",
                "wins": wins,
                "games": len(games),
                "win_rate": _rate(wins, len(games)),
            }
        )

    by_policy: dict[str, dict[str, Any]] = defaultdict(
        lambda: defaultdict(int, {"decision_seconds": 0.0})
    )
    for outcome in outcomes:
        for seat, agent in enumerate(outcome["agents"]):
            policy = _policy_id(agent)
            row = by_policy[policy]
            row["games_observed"] += 1
            row["engine_completed"] += int(outcome.get("game_over", False))
            row["strict_games"] += int(outcome.get("competitive_eligible", False))
            if outcome.get("competitive_eligible", False):
                row["strict_wins"] += int(outcome.get("winner") == seat)
                row["first_games" if seat == 0 else "second_games"] += 1
                row["first_wins" if seat == 0 else "second_wins"] += int(outcome.get("winner") == seat)
            row["illegal_actions"] += illegal_by_game[outcome["game_id"]][seat]
            row["decisions"] += int(outcome.get("decisions_by_player", [0, 0])[seat])
            row["tool_calls"] += tool_by_game[outcome["game_id"]][seat]
            usage = outcome.get("model_usage_totals", {})
            row["model_calls"] += int(usage.get("model_calls", [0, 0])[seat])
            row["input_tokens"] += int(usage.get("input_tokens", [0, 0])[seat])
            row["output_tokens"] += int(usage.get("output_tokens", [0, 0])[seat])
            row["decision_seconds"] += float(outcome.get("decision_seconds", [0, 0])[seat])

    policies = sorted(by_policy)
    rating: dict[str, GlickoPlayer] = {policy: GlickoPlayer() for policy in policies}
    rating_games = [
        outcome for outcome in strict if _policy_id(outcome["agents"][0]) != _policy_id(outcome["agents"][1])
    ]
    if len(policies) >= 2 and rating_games:
        snapshots = dict(rating)
        for policy in policies:
            results = []
            for outcome in rating_games:
                policy_seats = [_policy_id(agent) for agent in outcome["agents"]]
                if policy not in policy_seats:
                    continue
                seat = policy_seats.index(policy)
                opponent = snapshots[policy_seats[1 - seat]]
                score = 1.0 if outcome.get("winner") == seat else 0.0
                results.append((opponent, score))
            rating[policy] = Glicko2.update(snapshots[policy], results)
        glicko = {"status": "COMPUTED", "rated_games": len(rating_games)}
    else:
        glicko = {
            "status": "INSUFFICIENT_OPPONENT_DIVERSITY",
            "rated_games": 0,
            "reason": "Glicko-2 requires at least two distinct policy identities in strict games.",
        }

    leaderboard = []
    for policy in policies:
        value = dict(by_policy[policy])
        decisions = int(value.get("decisions", 0))
        strict_games = int(value.get("strict_games", 0))
        strict_wins = int(value.get("strict_wins", 0))
        first_games = int(value.get("first_games", 0))
        first_wins = int(value.get("first_wins", 0))
        second_games = int(value.get("second_games", 0))
        second_wins = int(value.get("second_wins", 0))
        engine_completed = int(value.get("engine_completed", 0))
        games_observed = int(value.get("games_observed", 0))
        illegal_actions = int(value.get("illegal_actions", 0))
        tool_calls = int(value.get("tool_calls", 0))
        model_calls = int(value.get("model_calls", 0))
        input_tokens = int(value.get("input_tokens", 0))
        output_tokens = int(value.get("output_tokens", 0))
        decision_seconds = float(value.get("decision_seconds", 0.0))
        item = {
            "policy": policy,
            **value,
            # A policy with no strict-eligible games has no corresponding
            # Counter keys. Materialize zero defaults for downstream summaries.
            "strict_games": strict_games,
            "strict_wins": strict_wins,
            "first_games": first_games,
            "first_wins": first_wins,
            "second_games": second_games,
            "second_wins": second_wins,
            "overall_win_rate": _rate(strict_wins, strict_games),
            "first_player_win_rate": _rate(first_wins, first_games),
            "second_player_win_rate": _rate(second_wins, second_games),
            "engine_completion_rate": _rate(engine_completed, games_observed),
            "illegal_action_rate": _rate(illegal_actions, decisions),
            "tool_calls_per_decision": _rate(tool_calls, decisions),
            "model_calls_per_decision": _rate(model_calls, decisions),
            "input_tokens_per_decision": _rate(input_tokens, decisions),
            "output_tokens_per_decision": _rate(output_tokens, decisions),
            "mean_decision_latency_seconds": _rate(decision_seconds, decisions),
        }
        if glicko["status"] == "COMPUTED":
            item["glicko2"] = {
                "mu": rating[policy].rating,
                "rd_phi": rating[policy].deviation,
                "volatility": rating[policy].volatility,
            }
        else:
            item["glicko2"] = None
        leaderboard.append(item)

    pair_rows = []
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for outcome in strict:
        pairs[tuple(sorted(outcome["decks"]))].append(outcome)
    for decks, games in sorted(pairs.items()):
        directions = Counter(tuple(outcome["decks"]) for outcome in games)
        pair_rows.append(
            {
                "deck_pair": list(decks),
                "strict_games": len(games),
                "directions": {"__vs__".join(key): value for key, value in sorted(directions.items())},
                "reciprocal_pair_complete": len(directions) == 2,
            }
        )
    return {
        "experiment": 1,
        "protocol": "full_duel_arena_pdf_v1",
        "games_observed": len(outcomes),
        "strict_games": len(strict),
        "eligibility_exclusions": dict(Counter(
            (
                "model_retry_exhausted_forfeit"
                if outcome.get("termination") == "model_retry_exhausted_forfeit"
                else "recovered_invalid_action"
                if outcome.get("recovered_invalid_action")
                else outcome.get("termination", "unknown")
            )
            for outcome in outcomes
            if not outcome.get("competitive_eligible", False)
        )),
        "side_swapped_win_rate": {
            "overall_by_policy": [
                {
                    "policy": item["policy"],
                    "wins": item["strict_wins"],
                    "games": item["strict_games"],
                    "win_rate": item["overall_win_rate"],
                }
                for item in sorted(leaderboard, key=lambda item: item["policy"])
            ],
            "by_seat": seats,
            "note": "Seat-conditioned results are not model rankings when both seats share a policy.",
        },
        "glicko2": glicko,
        "engine_completion": {
            "completed_games": sum(outcome.get("termination") == "game_over" for outcome in outcomes),
            "games": len(outcomes),
            "rate": _rate(
                sum(outcome.get("termination") == "game_over" for outcome in outcomes),
                len(outcomes),
            ),
        },
        "deck_pair_coverage": pair_rows,
        "leaderboard": sorted(leaderboard, key=lambda item: item["policy"]),
    }


def _turn_units(rows: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("legal", {}).get("requires_model_reasoning"):
            grouped[(row["game_id"], int(row["turn"]), int(row["player"]))].append(row)
    units = []
    for (game_id, turn, player), unit_rows in grouped.items():
        outcome = outcomes[game_id]
        flags = [not row.get("validation", {}).get("valid", False) for row in unit_rows]
        units.append(
            {
                "game_id": game_id,
                "turn": turn,
                "player": player,
                "agent": outcome["agents"][player],
                "policy": _policy_id(outcome["agents"][player]),
                "deck": outcome["decks"][player],
                "seat": player,
                "decision_length": len(unit_rows),
                "mean_legal_branching": sum(int(row["legal"].get("legal_action_count", 0)) for row in unit_rows) / len(unit_rows),
                "max_chain_depth": max(len(row.get("observation", {}).get("chain", [])) for row in unit_rows),
                "event_at": next((index + 1 for index, flag in enumerate(flags) if flag), None),
                "event": any(flags),
            }
        )
    return units


def _cox_report(units: list[dict[str, Any]]) -> dict[str, Any]:
    policies = sorted({unit["policy"] for unit in units})
    events = sum(unit["event"] for unit in units)
    if len(policies) < 2:
        return {"status": "INSUFFICIENT_DATA", "reason": "fewer than two policy identities", "events": events,
                "formula": "hazard ~ model + deck + seat + branching + chain_depth"}
    if events < 10:
        return {"status": "INSUFFICIENT_DATA", "reason": "fewer than 10 first-error events", "events": events,
                "formula": "hazard ~ model + deck + seat + branching + chain_depth"}
    return {"status": "NOT_FITTED", "reason": "Cox fitting is reserved for a multi-model Arena with adequate events.",
            "events": events, "formula": "hazard ~ model + deck + seat + branching + chain_depth"}


def _long_chain_metrics(rows: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_by_id = _outcome_map(outcomes)
    units = _turn_units(rows, outcome_by_id)
    agents = sorted({unit["agent"] for unit in units})
    km = []
    for agent in agents:
        selected = [unit for unit in units if unit["agent"] == agent]
        survival = 1.0
        for depth in sorted({unit["event_at"] or unit["decision_length"] for unit in selected}):
            at_risk = sum((unit["event_at"] or unit["decision_length"]) >= depth for unit in selected)
            events = sum(unit["event_at"] == depth for unit in selected)
            censored = sum(unit["decision_length"] == depth and not unit["event"] for unit in selected)
            if at_risk:
                survival *= 1 - events / at_risk
            km.append({"agent": agent, "depth": depth, "at_risk": at_risk, "events": events,
                       "censored": censored, "survival": survival})
    cutpoints = tuple(_percentile([unit["decision_length"] for unit in units], q) or 0 for q in (.25, .5, .75))
    game_length = {game_id: max(unit["decision_length"] for unit in units if unit["game_id"] == game_id) for game_id in {unit["game_id"] for unit in units}}
    complexity = defaultdict(lambda: Counter())
    for outcome in outcomes:
        if not outcome.get("competitive_eligible", False) or outcome["game_id"] not in game_length:
            continue
        quartile = _bucket(game_length[outcome["game_id"]], cutpoints)
        for seat, agent in enumerate(outcome["agents"]):
            key = f"{agent}|{quartile}"
            complexity[key]["games"] += 1
            complexity[key]["wins"] += int(outcome.get("winner") == seat)
    depth_bins = defaultdict(lambda: Counter())
    for row in rows:
        if not row.get("legal", {}).get("requires_model_reasoning"):
            continue
        outcome = outcome_by_id[row["game_id"]]
        agent = outcome["agents"][row["player"]]
        depth = int(row["decision_index"])
        label = "1-10" if depth <= 10 else "11-20" if depth <= 20 else "21-40" if depth <= 40 else "41+"
        entry = depth_bins[f"{agent}|{label}"]
        entry["decisions"] += 1
        entry["invalid"] += int(not row.get("validation", {}).get("valid", False))
    return {
        "experiment": 2,
        "trajectory_variables": ["turn_decision_length", "global_decision_depth", "legal_branching_size", "chain_depth"],
        "turn_decision_length_quartile_cutpoints": {"q1": cutpoints[0], "q2": cutpoints[1], "q3": cutpoints[2]},
        "kaplan_meier": km,
        "illegal_action_rate_by_decision_depth": [
            {"group": key, **value, "illegal_action_rate": _rate(value["invalid"], value["decisions"])}
            for key, value in sorted(depth_bins.items())
        ],
        "complexity_stratified_win_rate": [
            {"group": key, **value, "win_rate": _rate(value["wins"], value["games"])}
            for key, value in sorted(complexity.items())
        ],
        "cox_proportional_hazards": _cox_report(units),
    }


def _execution_metrics(rows: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    outcome_by_id = _outcome_map(outcomes)
    by_agent: dict[str, Counter[str]] = defaultdict(Counter)
    by_responder: dict[str, Counter[str]] = defaultdict(Counter)
    by_complexity: dict[str, Counter[str]] = defaultdict(Counter)
    taxonomy: dict[str, Counter[str]] = defaultdict(Counter)
    raw_taxonomy: dict[str, Counter[str]] = defaultdict(Counter)

    def rule_complexity(responder: str) -> str:
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

    def standard_error_labels(row: dict[str, Any]) -> set[str]:
        """Map evidence to the PDF taxonomy without pretending ambiguity is known."""

        validation = row.get("validation", {})
        expected = str(row.get("legal", {}).get("expected_responder", ""))
        attempted = row.get("attempted_action") or {}
        tool = attempted.get("tool") if isinstance(attempted, dict) else None
        arguments = attempted.get("arguments", {}) if isinstance(attempted, dict) else {}
        raw = " ".join(
            str(value or "")
            for value in (
                validation.get("agent_error"), validation.get("pre_engine_error"),
                validation.get("engine_error"), *validation.get("protocol_errors", []),
            )
        ).lower()
        labels: set[str] = set()
        if tool is None or "missing" in raw and "tool" in raw:
            labels.add("missing_tool_call")
        if tool and expected and tool != expected:
            labels.add("wrong_responder")
        if "multiple" in raw and "tool" in raw:
            labels.add("multiple_tool_call")
        if "stale" in raw:
            labels.add("stale_action_or_index")
        elif "index" in raw or ("action_not_in_exact_legal_set" in raw and any(key in arguments for key in ("index", "indices"))):
            labels.add("invalid_index")
        if "target" in raw:
            labels.add("invalid_target")
        if "material" in raw or "tribute" in raw or "select_sum" in raw:
            labels.add("invalid_material_selection")
        if "position" in raw or "place" in raw:
            labels.add("invalid_position_or_place")
        if "timing" in raw or "rule" in raw:
            labels.add("rule_or_timing_violation")
        if "protocol" in raw or "parse" in raw or "provider" in raw:
            labels.add("engine_or_protocol_parsing_error")
        if validation.get("engine_error"):
            labels.add("engine_or_protocol_parsing_error")
        if not labels and not validation.get("valid", False):
            labels.add("unclassified_invalid_action")
        return labels

    def inspect_calls(row: dict[str, Any]) -> int:
        return sum(
            call.get("name") == "inspect_card"
            for turn in row.get("trace", {}).get("turns", [])
            for call in turn.get("tool_calls", [])
        )
    for row in rows:
        if not row.get("legal", {}).get("requires_model_reasoning"):
            continue
        agent = outcome_by_id[row["game_id"]]["agents"][row["player"]]
        validation = row.get("validation", {})
        item = by_agent[agent]
        item["decisions"] += 1
        item["invalid"] += int(not validation.get("valid", False))
        item["fallbacks"] += int(bool(validation.get("fallback")))
        item["model_retry_exhausted_forfeits"] += int(
            bool(validation.get("terminal_model_failure"))
        )
        # ``model_action_attempts`` counts action submissions/corrections;
        # trace length also contains legitimate inspect_card calls.
        action_attempts = int(validation.get("model_action_attempts", 1) or 1)
        item["action_retries"] += max(0, action_attempts - 1)
        item["tool_calls"] += _tool_calls(row)
        item["model_calls"] += len(row.get("trace", {}).get("turns", []))
        item["inspect_card_calls"] += inspect_calls(row)
        responder = str(row["legal"].get("expected_responder", "unknown"))
        entry = by_responder[f"{agent}|{responder}"]
        entry["decisions"] += 1
        entry["invalid"] += int(not validation.get("valid", False))
        complexity = rule_complexity(responder)
        complexity_entry = by_complexity[f"{agent}|{complexity}"]
        complexity_entry["decisions"] += 1
        complexity_entry["invalid"] += int(not validation.get("valid", False))
        for error in standard_error_labels(row):
            taxonomy[agent][error] += 1
        for key in ("protocol_errors",):
            for error in validation.get(key, []): raw_taxonomy[agent][error] += 1
        for label, value in (("agent_error", validation.get("agent_error")), ("pre_engine_error", validation.get("pre_engine_error")),
                             ("engine_error", validation.get("engine_error")), ("recovery", validation.get("recovery"))):
            if value and value != "none": raw_taxonomy[agent][f"{label}:{str(value).split(':', 1)[0]}"] += 1
    completion = defaultdict(lambda: Counter())
    for outcome in outcomes:
        for seat, agent in enumerate(outcome["agents"]):
            completion[agent]["games"] += 1
            completion[agent]["engine_completed"] += int(outcome.get("termination") == "game_over")
    def enrich(key: str, value: Counter[str]) -> dict[str, Any]:
        decisions = value["decisions"]
        return {"agent": key, **value, "illegal_action_rate": _rate(value["invalid"], decisions),
                "fallbacks_per_100_decisions": 100 * value["fallbacks"] / decisions if decisions else None,
                "action_retries_per_100_decisions": 100 * value["action_retries"] / decisions if decisions else None,
                "tool_calls_per_decision": _rate(value["tool_calls"], decisions), "model_calls_per_decision": _rate(value["model_calls"], decisions),
                "inspect_card_calls_per_decision": _rate(value["inspect_card_calls"], decisions),
                "engine_completion_rate": _rate(completion[key]["engine_completed"], completion[key]["games"])}
    return {"experiment": 4, "by_agent": [enrich(key, value) for key, value in sorted(by_agent.items())],
            "invalid_action_rate_by_responder_type": [
                {"group": key, **value, "invalid_rate": _rate(value["invalid"], value["decisions"])} for key, value in sorted(by_responder.items())],
            "invalid_action_rate_by_rule_complexity": [
                {"group": key, **value, "invalid_rate": _rate(value["invalid"], value["decisions"])} for key, value in sorted(by_complexity.items())],
            "error_taxonomy": [{"agent": key, "counts": dict(sorted(value.items()))} for key, value in sorted(taxonomy.items())],
            "raw_error_evidence": [{"agent": key, "counts": dict(sorted(value.items()))} for key, value in sorted(raw_taxonomy.items())]}


def compute_phase2_metrics(
    run_dir: Path,
    *,
    experiments: tuple[str, ...] = ("exp1", "exp2", "exp4"),
) -> dict[str, Any]:
    """Compute selected Phase 2 experiments without touching unrequested ones."""

    available = {
        "exp1": lambda outcomes, rows: _arena_metrics(outcomes, rows),
        "exp2": lambda outcomes, rows: _long_chain_metrics(rows, outcomes),
        "exp4": lambda outcomes, rows: _execution_metrics(rows, outcomes),
    }
    unknown = set(experiments) - set(available)
    if unknown:
        raise ValueError(f"Unknown Phase 2 experiments: {sorted(unknown)}")
    outcomes, rows = _outcomes_and_rows(run_dir)
    combined = {name: available[name](outcomes, rows) for name in experiments}
    for name, value in combined.items(): atomic_write_json(run_dir / "metrics" / name / "metrics.json", value)
    if set(experiments) == set(available):
        atomic_write_json(run_dir / "metrics" / "phase2_metrics.json", combined)
    return combined


def _normal(value: Any) -> Any:
    return sorted(value) if isinstance(value, list) else value


def _f1(truth: list[str], prediction: list[str]) -> tuple[int, int, int, float | None]:
    left, right = Counter(truth), Counter(prediction)
    tp = sum((left & right).values()); fp = sum((right - left).values()); fn = sum((left - right).values())
    return tp, fp, fn, _rate(2 * tp, 2 * tp + fp + fn)


def _state_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if isinstance(row.get("prediction"), dict)]
    list_keys = [key for row in valid for key, value in row["ground_truth"].items() if isinstance(value, list)]
    list_keys = sorted(set(list_keys)); scalar_correct = scalar_total = 0; joint = 0; tp = fp = fn = 0; macro = []
    progress = defaultdict(lambda: Counter())
    complexity: dict[str, defaultdict[str, Counter[str]]] = {
        "chain_depth": defaultdict(Counter),
        "zone_transition_events": defaultdict(Counter),
        "historical_action_count": defaultdict(Counter),
        "historical_public_activation_count": defaultdict(Counter),
    }
    for row in valid:
        truth, prediction = row["ground_truth"], row["prediction"]
        exact = True
        for key, value in truth.items():
            if isinstance(value, list):
                pred = prediction.get(key, [])
                if not isinstance(pred, list): pred = []
                a, b, c, f1 = _f1(value, [str(item) for item in pred]); tp += a; fp += b; fn += c
                if f1 is not None: macro.append(f1)
                exact &= _normal(pred) == _normal(value)
            else:
                scalar_total += 1; scalar_correct += int(prediction.get(key) == value); exact &= prediction.get(key) == value
        joint += int(exact)
        p = float(row.get("trajectory_progress", 0)); pbin = "0-25%" if p <= .25 else "25-50%" if p <= .5 else "50-75%" if p <= .75 else "75-100%"
        progress[pbin]["samples"] += 1; progress[pbin]["joint_correct"] += int(exact)
        state_complexity = row.get("state_complexity", {})
        for dimension, groups in complexity.items():
            value = state_complexity.get(dimension, "unknown")
            if dimension in {"zone_transition_events", "historical_public_activation_count"}:
                label = "0" if value == 0 else "1-2" if isinstance(value, int) and value <= 2 else "3+"
            elif dimension == "historical_action_count":
                label = "0-25" if isinstance(value, int) and value <= 25 else "26-100" if isinstance(value, int) and value <= 100 else "101+"
            else:
                label = str(value)
            groups[label]["samples"] += 1; groups[label]["joint_correct"] += int(exact)
    return {"experiment": 3, "samples": len(rows), "valid_samples": len(valid), "jga": _rate(joint, len(valid)),
            "slot_accuracy": _rate(scalar_correct, scalar_total), "slot_f1": {"micro_f1": _rate(2 * tp, 2 * tp + fp + fn), "macro_f1": sum(macro) / len(macro) if macro else None},
            "parse_failure_rate": _rate(len(rows) - len(valid), len(rows)),
            "jga_by_trajectory_progress": [{"bin": key, **value, "jga": _rate(value["joint_correct"], value["samples"])} for key, value in sorted(progress.items())],
            # Retain the legacy chain-depth list for existing consumers, and
            # expose the PDF-complete multi-axis breakdown alongside it.
            "jga_by_state_complexity": [
                {"bin": key, **value, "jga": _rate(value["joint_correct"], value["samples"])}
                for key, value in sorted(complexity["chain_depth"].items())
            ],
            "jga_by_state_complexity_dimension": {
                dimension: [{"bin": key, **value, "jga": _rate(value["joint_correct"], value["samples"])} for key, value in sorted(groups.items())]
                for dimension, groups in complexity.items()
            }}


def _forecast_metrics(rows: list[dict[str, Any]], truth_key: str, probability_key: str) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    valid = [row for row in rows if row.get(truth_key) in (0, 1) and isinstance(row.get(probability_key), (int, float)) and 0 <= row[probability_key] <= 1]
    y = np.array([row[truth_key] for row in valid], dtype=int); p = np.array([row[probability_key] for row in valid], dtype=float)
    both = len(set(y.tolist())) == 2; ece = 0.0
    for low in np.linspace(0, .9, 10):
        mask = (p >= low) & (p < low + .1 if low < .9 else p <= 1)
        if mask.any(): ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return {"samples": len(rows), "valid_samples": len(valid), "prevalence": float(y.mean()) if len(y) else None,
            "auprc": float(average_precision_score(y, p)) if both else None, "auroc": float(roc_auc_score(y, p)) if both else None,
            "brier_score": float(brier_score_loss(y, p)) if len(y) else None, "ece_10_equal_width": ece if len(y) else None,
            "parse_failure_rate": _rate(len(rows) - len(valid), len(rows))}


def _forecast_strata(rows: list[dict[str, Any]], truth_key: str, probability_key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key, value in row.get("strata", {}).items():
            if key in {"hand_size", "opponent_set_card_count", "chain_depth", "hidden_information_density"}:
                value = "0" if int(value) == 0 else "1-2" if int(value) <= 2 else "3+"
            groups[key][str(value)].append(row)
    return {key: [{"stratum": value, **_forecast_metrics(items, truth_key, probability_key)} for value, items in sorted(values.items())] for key, values in sorted(groups.items())}


def _update_probe_metric_aggregate(
    path: Path, *, experiment: int, evaluator_id: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    """Store one evaluator result without overwriting peer-model metrics."""

    aggregate = read_json(path) or {"experiment": experiment, "evaluators": {}}
    if "evaluators" not in aggregate:
        aggregate = {
            "experiment": experiment,
            "evaluators": {"legacy_unspecified": aggregate},
        }
    aggregate["evaluators"][evaluator_id] = metrics
    atomic_write_json(path, aggregate)
    return aggregate


def compute_probe_metrics(
    run_dir: Path,
    *,
    provider_name: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    evaluator_id = probe_evaluator_id(provider_name, model)
    result_dir = run_dir / "derived" / "probe_results" / evaluator_id
    states = JsonlJournal(result_dir / "state_probe_results.jsonl").recover()
    forecasts = JsonlJournal(result_dir / "forecast_results.jsonl").recover()
    exp3 = _state_metrics(states)
    exp5 = {"experiment": 5, "primary_target": "availability", "availability": _forecast_metrics(forecasts, "availability_ground_truth", "availability_probability"),
            "behavior": _forecast_metrics(forecasts, "behavior_ground_truth", "behavior_probability"),
            "stratified": {"availability": _forecast_strata(forecasts, "availability_ground_truth", "availability_probability"),
                           "behavior": _forecast_strata(forecasts, "behavior_ground_truth", "behavior_probability")}}
    exp3["evaluator_id"] = evaluator_id
    exp5["evaluator_id"] = evaluator_id
    atomic_write_json(run_dir / "metrics" / "exp3" / f"{evaluator_id}.json", exp3)
    atomic_write_json(run_dir / "metrics" / "exp5" / f"{evaluator_id}.json", exp5)
    _update_probe_metric_aggregate(
        run_dir / "metrics" / "exp3" / "metrics.json",
        experiment=3,
        evaluator_id=evaluator_id,
        metrics=exp3,
    )
    _update_probe_metric_aggregate(
        run_dir / "metrics" / "exp5" / "metrics.json",
        experiment=5,
        evaluator_id=evaluator_id,
        metrics=exp5,
    )
    return {"evaluator_id": evaluator_id, "exp3": exp3, "exp5": exp5}
