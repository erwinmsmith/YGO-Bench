"""Metrics-only outputs for Experiments 1, 2, 3, 4, 5, and 6."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ygobench.experiments.io import JsonlJournal, atomic_write_json, read_json


def _game_dirs(run_dir: Path) -> list[Path]:
    return sorted(path.parent for path in (run_dir / "games").glob("*/manifest.json"))


def _metric(name: str, numerator: int, denominator: int, *, direction: str) -> dict[str, Any]:
    return {
        "metric": name,
        "numerator": numerator,
        "denominator": denominator,
        "estimate": numerator / denominator if denominator else None,
        "direction": direction,
    }


def compute_phase2_metrics(run_dir: Path) -> dict[str, Any]:
    outcomes = []
    decisions: list[dict[str, Any]] = []
    for game_dir in _game_dirs(run_dir):
        outcome = read_json(game_dir / "outcome.json")
        if outcome:
            outcomes.append(outcome)
        decisions.extend(JsonlJournal(game_dir / "trajectory.jsonl").recover())

    agents = sorted({agent for outcome in outcomes for agent in outcome.get("agents", [])})
    exp1_rows = []
    for agent in agents:
        seat_games = [
            (outcome, seat)
            for outcome in outcomes
            for seat, seat_agent in enumerate(outcome["agents"])
            if seat_agent == agent
        ]
        wins = sum(outcome.get("winner") == seat for outcome, seat in seat_games)
        exp1_rows.append(
            {
                "agent": agent,
                **_metric(
                    "win_rate", wins, len(seat_games), direction="higher_is_better"
                ),
                "seat_games": len(seat_games),
                "engine_completion_rate": sum(
                    outcome.get("game_over", False) for outcome, _ in seat_games
                )
                / max(1, len(seat_games)),
                "input_tokens": sum(
                    outcome["model_usage_totals"]["input_tokens"][seat]
                    for outcome, seat in seat_games
                ),
                "output_tokens": sum(
                    outcome["model_usage_totals"]["output_tokens"][seat]
                    for outcome, seat in seat_games
                ),
            }
        )
    exp1 = {"experiment": 1, "games": len(outcomes), "leaderboard": exp1_rows}

    taxonomy: dict[str, Counter[str]] = defaultdict(Counter)
    responder: dict[str, Counter[str]] = defaultdict(Counter)
    depth_bins: dict[str, Counter[str]] = defaultdict(Counter)
    turn_units: dict[tuple[str, str, int, int], list[bool]] = defaultdict(list)
    for row in decisions:
        agent = next(
            (
                outcome["agents"][row["player"]]
                for outcome in outcomes
                if outcome["game_id"] == row["game_id"]
            ),
            f"seat-{row['player']}",
        )
        validation = row["validation"]
        nontrivial = bool(row["legal"].get("requires_model_reasoning"))
        invalid = not validation.get("valid", False)
        if nontrivial:
            taxonomy[agent]["decisions"] += 1
            taxonomy[agent]["invalid"] += int(invalid)
            for error in validation.get("protocol_errors", []):
                taxonomy[agent][error] += 1
            agent_error = validation.get("agent_error") or ""
            engine_error = validation.get("engine_error") or ""
            if "wrong_responder" in agent_error:
                taxonomy[agent]["wrong_responder"] += 1
            if engine_error:
                taxonomy[agent]["engine_rejection"] += 1
            if validation.get("retry_count"):
                taxonomy[agent]["retried_decisions"] += 1
            if validation.get("fallback") or agent_error:
                taxonomy[agent]["fallback_decisions"] += 1
            expected = row["legal"].get("expected_responder", "unknown")
            responder[f"{agent}|{expected}"]["decisions"] += 1
            responder[f"{agent}|{expected}"]["invalid"] += int(invalid)
            depth = int(row["decision_index"])
            bucket = (
                "1-10"
                if depth <= 10
                else "11-20"
                if depth <= 20
                else "21-40"
                if depth <= 40
                else "41+"
            )
            depth_bins[f"{agent}|{bucket}"]["decisions"] += 1
            depth_bins[f"{agent}|{bucket}"]["invalid"] += int(invalid)
            turn_units[(agent, row["game_id"], int(row["turn"]), int(row["player"]))].append(
                invalid
            )

    km = []
    for agent in agents:
        durations = []
        for key, flags in turn_units.items():
            if key[0] != agent:
                continue
            event_at = next((index + 1 for index, flag in enumerate(flags) if flag), None)
            durations.append((event_at or len(flags), event_at is not None))
        survival = 1.0
        for depth in sorted({duration for duration, _ in durations}):
            at_risk = sum(duration >= depth for duration, _ in durations)
            events = sum(duration == depth and observed for duration, observed in durations)
            censored = sum(duration == depth and not observed for duration, observed in durations)
            if at_risk:
                survival *= 1 - events / at_risk
            km.append(
                {
                    "agent": agent,
                    "depth": depth,
                    "at_risk": at_risk,
                    "events": events,
                    "censored": censored,
                    "survival": survival,
                }
            )
    exp2 = {
        "experiment": 2,
        "kaplan_meier": km,
        "depth_bins": [
            {
                "group": key,
                **value,
                "illegal_action_rate": value["invalid"] / max(1, value["decisions"]),
            }
            for key, value in sorted(depth_bins.items())
        ],
    }
    exp4 = {
        "experiment": 4,
        "by_agent": [
            {
                "agent": agent,
                **counts,
                "illegal_action_rate": counts["invalid"] / max(1, counts["decisions"]),
            }
            for agent, counts in sorted(taxonomy.items())
        ],
        "by_responder": [
            {
                "group": key,
                **counts,
                "invalid_rate": counts["invalid"] / max(1, counts["decisions"]),
            }
            for key, counts in sorted(responder.items())
        ],
    }
    for name, value in (("exp1", exp1), ("exp2", exp2), ("exp4", exp4)):
        atomic_write_json(run_dir / "metrics" / name / "metrics.json", value)
    combined = {"exp1": exp1, "exp2": exp2, "exp4": exp4}
    atomic_write_json(run_dir / "metrics" / "phase2_metrics.json", combined)
    return combined


def compute_probe_metrics(run_dir: Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    state_rows = JsonlJournal(run_dir / "derived" / "state_probe_results.jsonl").recover()
    forecast_rows = JsonlJournal(run_dir / "derived" / "forecast_results.jsonl").recover()
    valid_state = [row for row in state_rows if isinstance(row.get("prediction"), dict)]
    state_success = [row["prediction"] == row["ground_truth"] for row in valid_state]
    slots_correct = slots_total = 0
    for row in valid_state:
        for key, truth in row["ground_truth"].items():
            slots_total += 1
            slots_correct += row["prediction"].get(key) == truth
    exp3 = {
        "experiment": 3,
        "samples": len(state_rows),
        "valid_samples": len(valid_state),
        "jga": sum(state_success) / len(valid_state) if valid_state else None,
        "slot_accuracy": slots_correct / slots_total if slots_total else None,
        "parse_failure_rate": (len(state_rows) - len(valid_state)) / max(1, len(state_rows)),
    }

    valid_forecast = [
        row
        for row in forecast_rows
        if isinstance(row.get("probability"), (int, float)) and 0 <= row["probability"] <= 1
    ]
    y = np.array([row["ground_truth"] for row in valid_forecast], dtype=int)
    p = np.array([row["probability"] for row in valid_forecast], dtype=float)
    ece = None
    if len(y):
        total = 0.0
        for low in np.linspace(0, 0.9, 10):
            mask = (p >= low) & (p < low + 0.1 if low < 0.9 else p <= 1)
            if mask.any():
                total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
        ece = float(total)
    both_classes = len(set(y.tolist())) == 2
    exp5 = {
        "experiment": 5,
        "samples": len(forecast_rows),
        "valid_samples": len(valid_forecast),
        "prevalence": float(y.mean()) if len(y) else None,
        "auprc": float(average_precision_score(y, p)) if both_classes else None,
        "auroc": float(roc_auc_score(y, p)) if both_classes else None,
        "brier_score": float(brier_score_loss(y, p)) if len(y) else None,
        "ece_10_equal_width": ece,
        "parse_failure_rate": (len(forecast_rows) - len(valid_forecast))
        / max(1, len(forecast_rows)),
    }
    atomic_write_json(run_dir / "metrics" / "exp3" / "metrics.json", exp3)
    atomic_write_json(run_dir / "metrics" / "exp5" / "metrics.json", exp5)
    return {"exp3": exp3, "exp5": exp5}
