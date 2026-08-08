"""Phase-4 legal counterfactual branch selection and execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ygobench.engine.protocol import ActionChoice
from ygobench.experiments.config import ExperimentConfig, stable_id
from ygobench.experiments.io import JsonlJournal, atomic_write_json, read_json
from ygobench.experiments.runner import run_evidence_duel


def find_interruption_candidate(game_dir: Path) -> dict[str, Any] | None:
    rows = JsonlJournal(game_dir / "trajectory.jsonl").recover()
    for index, row in enumerate(rows):
        legal = row.get("legal", {})
        if legal.get("expected_responder") != "select_chain" or not legal.get(
            "enumeration_complete"
        ):
            continue
        alternatives = [
            action
            for action in legal.get("actions", [])
            if action.get("arguments", {}).get("index") is not None
            and action != row.get("executed_action")
        ]
        if alternatives:
            return {
                "decision_index": index,
                "decision_id": row["decision_id"],
                "original_action": row["executed_action"],
                "interruption_action": alternatives[0],
                "interruption_player": int(row["player"]),
                "focal_player": 1 - int(row["player"]),
            }
    return None


def run_interruption_branch(game_dir: Path, *, root: Path) -> dict[str, Any]:
    manifest = read_json(game_dir / "manifest.json")
    parent = manifest["config"]
    rows = JsonlJournal(game_dir / "trajectory.jsonl").recover()
    candidate = find_interruption_candidate(game_dir)
    if candidate is None:
        report = {
            "phase": 4,
            "status": "INELIGIBLE",
            "reason": "no exact legal alternative select_chain interruption in the trajectory",
        }
        atomic_write_json(game_dir / "exp6_branch_report.json", report)
        return report
    branch_run = f"{parent['run_id']}_exp6"
    branch_game = stable_id(
        "branch", [parent["game_id"], candidate["decision_id"], "legal_interruption"]
    )
    config = ExperimentConfig.build(
        run_id=branch_run,
        game_id=branch_game,
        deck1=parent["deck1"],
        deck2=parent["deck2"],
        agent1=parent["agent1"],
        agent2=parent["agent2"],
        seed=int(parent["seed"]),
        max_decisions=int(parent["max_decisions"]),
        checkpoint_interval=int(parent.get("checkpoint_interval", 25)),
    )
    index = int(candidate["decision_index"])
    outcome = run_evidence_duel(
        config,
        root=root,
        replay_prefix=rows[:index],
        interventions={index: ActionChoice(**candidate["interruption_action"])},
    )
    report = {
        "phase": 4,
        "status": "COMPLETED",
        "parent_game_id": parent["game_id"],
        "branch_game_id": branch_game,
        "candidate": candidate,
        "clean_winner": read_json(game_dir / "outcome.json").get("winner"),
        "interrupted_winner": outcome.get("winner"),
        "branch_outcome": outcome,
    }
    atomic_write_json(game_dir / "exp6_branch_report.json", report)
    return report


def write_exp6_metrics(
    run_dir: Path,
    game_dir: Path,
    *,
    reversibility: dict[str, Any],
    branch: dict[str, Any] | None,
) -> dict[str, Any]:
    clean = read_json(game_dir / "outcome.json")
    branch = branch or {
        "status": "SKIPPED",
        "reason": "branch was not run",
    }
    interrupted = branch.get("branch_outcome", {})
    focal_player = branch.get("candidate", {}).get("focal_player")
    clean_win = focal_player is not None and clean.get("winner") == focal_player
    recovered = clean_win and interrupted.get("winner") == focal_player

    def total(outcome: dict[str, Any], key: str) -> int:
        return sum(outcome.get("model_usage_totals", {}).get(key, [0, 0]))

    paired_completed = branch.get("status") == "COMPLETED"
    recovery_costs = (
        {
            "additional_decisions": (
                interrupted.get("decisions", 0) - clean.get("decisions", 0)
            ),
            "additional_model_calls": total(interrupted, "model_calls")
            - total(clean, "model_calls"),
            "additional_input_tokens": total(interrupted, "input_tokens")
            - total(clean, "input_tokens"),
            "additional_output_tokens": total(interrupted, "output_tokens")
            - total(clean, "output_tokens"),
            "additional_decision_latency_seconds": round(
                sum(interrupted.get("decision_seconds", [0, 0]))
                - sum(clean.get("decision_seconds", [0, 0])),
                6,
            ),
        }
        if paired_completed
        else None
    )
    metrics = {
        "experiment": 6,
        "status": branch.get("status"),
        "reversibility_gate": {
            "passed": bool(reversibility.get("passed")),
            "sample_size": int(reversibility.get("sample_size", 0)),
        },
        "paired_branches": int(paired_completed),
        "plan_recovery_rate": {
            "metric": "plan_recovery_rate",
            "numerator": int(recovered),
            "denominator": int(clean_win),
            "estimate": float(recovered) if clean_win else None,
            "direction": "higher_is_better",
        },
        "clean": {
            "game_over": clean.get("game_over"),
            "winner": clean.get("winner"),
            "decisions": clean.get("decisions"),
        },
        "interrupted": {
            "game_over": interrupted.get("game_over"),
            "winner": interrupted.get("winner"),
            "decisions": interrupted.get("decisions"),
        },
        "recovery_costs": recovery_costs,
        "ineligibility_reason": branch.get("reason"),
    }
    atomic_write_json(run_dir / "metrics" / "exp6" / "metrics.json", metrics)
    return metrics
