"""Phase-4 offline legal counterfactual plan-robustness audit."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from ygobench.config import PROJECT_ROOT
from ygobench.engine.protocol import ActionChoice
from ygobench.experiments.config import ExperimentConfig, stable_id
from ygobench.experiments.io import JsonlJournal, atomic_write_json, content_hash, read_json
from ygobench.experiments.legal import exact_legal_action_match
from ygobench.experiments.oracle import build_oracle_state
from ygobench.experiments.runner import run_evidence_duel
from ygobench.experiments.session import DuelSession

_COMMITMENT_COMMANDS = {"activate", "summon", "sp_summon", "attack"}
_NEW_DECISION_BOUNDARIES = {"select_idlecmd", "select_battlecmd", "rock_paper_scissors"}


def _response_window_after(
    rows: list[dict[str, Any]], commitment_index: int
) -> tuple[int, dict[str, Any]] | None:
    """Return the first opposing chain window causally caused by a commitment."""
    commitment = rows[commitment_index]
    actor = int(commitment["player"])
    turn = int(commitment["turn"])
    for response_index in range(commitment_index + 1, len(rows)):
        row = rows[response_index]
        if int(row["turn"]) != turn:
            return None
        responder = str(row.get("legal", {}).get("expected_responder", ""))
        if int(row["player"]) != actor and responder == "select_chain":
            return response_index, row
        command = row.get("executed_action", {}).get("arguments", {}).get("command")
        if responder in _NEW_DECISION_BOUNDARIES or command in _COMMITMENT_COMMANDS:
            return None
    return None


def _action_payload(action: dict[str, Any]) -> dict[str, Any]:
    """Persist actions without model-generated labels or rationale text."""
    return {"tool": action["tool"], "arguments": action["arguments"]}


def find_offline_counterfactual_candidates(game_dir: Path) -> dict[str, Any]:
    """Find strict, natural opponent-interruption alternatives from a clean trace.

    A candidate is eligible only when the clean trace passed a response window
    and the opponent could instead take a non-pass exact-legal chain action.
    This deliberately excludes a player's own optional chains.
    """
    rows = JsonlJournal(game_dir / "trajectory.jsonl").recover()
    candidates: list[dict[str, Any]] = []
    commitment_points = 0
    for index, row in enumerate(rows):
        if not row.get("validation", {}).get("valid", False):
            continue
        action = row.get("executed_action")
        if not isinstance(action, dict):
            continue
        command = action.get("arguments", {}).get("command")
        if command not in _COMMITMENT_COMMANDS:
            continue
        commitment_points += 1
        window = _response_window_after(rows, index)
        if window is None:
            continue
        response_index, response_row = window
        legal = response_row.get("legal", {})
        clean_action = response_row.get("executed_action")
        if (
            not legal.get("enumeration_complete")
            or not isinstance(clean_action, dict)
            or clean_action.get("arguments", {}).get("index") is not None
        ):
            continue
        for alternative in legal.get("actions", []):
            if alternative.get("arguments", {}).get("index") is None:
                continue
            candidates.append(
                {
                    "candidate_id": stable_id(
                        "offline-counterfactual",
                        [
                            row["game_id"],
                            row["decision_id"],
                            response_row["decision_id"],
                            alternative["tool"],
                            alternative["arguments"],
                        ],
                    ),
                    "commitment_index": index,
                    "commitment_decision_id": row["decision_id"],
                    "response_index": response_index,
                    "response_decision_id": response_row["decision_id"],
                    "focal_player": int(row["player"]),
                    "interruption_player": int(response_row["player"]),
                    "commitment_action": _action_payload(action),
                    "clean_response": _action_payload(clean_action),
                    "counterfactual_response": _action_payload(alternative),
                }
            )
    return {
        "commitment_points": commitment_points,
        "eligible_counterfactuals": len(candidates),
        "candidates": candidates,
    }


def _state_summary(oracle: dict[str, Any]) -> dict[str, Any]:
    field = oracle.get("field", {})
    players = field.get("players", [{}, {}])
    tracked = oracle.get("tracked", {})
    return {
        "state_hash": oracle.get("state_hash"),
        "turn": tracked.get("turn_count"),
        "turn_player": tracked.get("turn_player"),
        "game_over": tracked.get("game_over"),
        "players": [
            {
                "lp": player.get("lp"),
                "hand_count": player.get("hand_count_raw"),
                "grave_count": player.get("grave_count_raw"),
                "deck_count": player.get("deck_count"),
                "monster_count": sum(card is not None for card in player.get("mzone_summary", [])),
                "spell_trap_count": sum(
                    card is not None for card in player.get("szone_summary", [])
                ),
            }
            for player in players
        ],
        "pending_message": oracle.get("pending", {}).get("msg_name"),
    }


def _replay_prefix(
    session: DuelSession, rows: list[dict[str, Any]], end_exclusive: int
) -> list[dict[str, Any]]:
    recent_actions: list[dict[str, Any]] = []
    for row in rows[:end_exclusive]:
        view = session.view(recent_actions)
        if view["oracle"]["state_hash"] != row["oracle_before_hash"]:
            raise RuntimeError(f"offline replay before-hash mismatch at {row['decision_id']}")
        if content_hash(view["legal"]) != content_hash(row["legal"]):
            raise RuntimeError(f"offline replay legal-set mismatch at {row['decision_id']}")
        session.execute(ActionChoice(**row["executed_action"]))
        if build_oracle_state(session)["state_hash"] != row["oracle_after_hash"]:
            raise RuntimeError(f"offline replay after-hash mismatch at {row['decision_id']}")
        recent_actions.append(row["action_summary"])
    return recent_actions


def _counterfactual_action_summary(
    row: dict[str, Any], action: ActionChoice
) -> dict[str, Any]:
    return {
        "decision_id": row["decision_id"],
        "action_id": row["action_id"],
        "turn": row["turn"],
        "player": row["player"],
        "tool": action.tool,
        "arguments": action.arguments,
    }


def _try_clean_action(
    session: DuelSession, recent_actions: list[dict[str, Any]], row: dict[str, Any]
) -> str | None:
    """Return a failure reason, or execute one clean continuation action."""
    if session.done:
        return "game_over_before_action"
    view = session.view(recent_actions)
    action = ActionChoice(**row["executed_action"])
    if int(view["player"]) != int(row["player"]):
        return "acting_player_changed"
    if view["legal"].get("expected_responder") != action.tool:
        return "expected_responder_changed"
    if not view["legal"].get("enumeration_complete"):
        return "legal_set_not_enumerated"
    if not exact_legal_action_match(
        action,
        view["actions"],
        signature=session.action_signature,
    ):
        return "action_not_in_counterfactual_exact_legal_set"
    try:
        session.execute(action)
    except Exception as exc:  # noqa: BLE001
        return f"engine_replay_error:{type(exc).__name__}"
    recent_actions.append(row["action_summary"])
    return None


def _audit_candidate(
    config: dict[str, Any], rows: list[dict[str, Any]], candidate: dict[str, Any], horizon: int
) -> dict[str, Any]:
    deck_root = PROJECT_ROOT / "resources" / "decks"
    response_index = int(candidate["response_index"])
    clean_session = DuelSession(
        deck_root / f"{config['deck1']}.ydk",
        deck_root / f"{config['deck2']}.ydk",
        seed=int(config["seed"]),
    )
    counterfactual_session = DuelSession(
        deck_root / f"{config['deck1']}.ydk",
        deck_root / f"{config['deck2']}.ydk",
        seed=int(config["seed"]),
    )
    try:
        clean_recent = _replay_prefix(clean_session, rows, response_index)
        counterfactual_recent = _replay_prefix(counterfactual_session, rows, response_index)
        response_row = rows[response_index]

        clean_view = clean_session.view(clean_recent)
        counterfactual_view = counterfactual_session.view(counterfactual_recent)
        if clean_view["oracle"]["state_hash"] != response_row["oracle_before_hash"]:
            raise RuntimeError("clean response checkpoint does not match logged state")
        if counterfactual_view["oracle"]["state_hash"] != response_row["oracle_before_hash"]:
            raise RuntimeError("counterfactual response checkpoint does not match logged state")

        clean_action = ActionChoice(**candidate["clean_response"])
        counterfactual_action = ActionChoice(**candidate["counterfactual_response"])
        clean_session.execute(clean_action)
        clean_after = build_oracle_state(clean_session)
        if clean_after["state_hash"] != response_row["oracle_after_hash"]:
            raise RuntimeError("clean response transition does not match logged state")

        if not exact_legal_action_match(
            counterfactual_action,
            counterfactual_view["actions"],
            signature=counterfactual_session.action_signature,
        ):
            raise RuntimeError("counterfactual action is not in the exact legal set")
        counterfactual_session.execute(counterfactual_action)
        counterfactual_after = build_oracle_state(counterfactual_session)
        counterfactual_recent.append(
            _counterfactual_action_summary(response_row, counterfactual_action)
        )

        survived_actions = 0
        first_failure: dict[str, Any] | None = None
        for row in rows[response_index + 1 : response_index + 1 + horizon]:
            reason = _try_clean_action(counterfactual_session, counterfactual_recent, row)
            if reason is not None:
                first_failure = {
                    "decision_id": row["decision_id"],
                    "decision_index": row["decision_index"],
                    "reason": reason,
                    "clean_action": _action_payload(row["executed_action"]),
                }
                break
            survived_actions += 1
        checked_all_available_actions = response_index + 1 + survived_actions >= len(rows)
        right_censored = first_failure is None and not checked_all_available_actions
        return {
            "candidate": candidate,
            "status": "COMPLETED",
            "clean_state_after_response": _state_summary(clean_after),
            "counterfactual_state_after_response": _state_summary(counterfactual_after),
            "state_changed": clean_after["state_hash"] != counterfactual_after["state_hash"],
            "action_sequence_survival": {
                "horizon": horizon,
                "survived_actions": survived_actions,
                "first_failure": first_failure,
                "right_censored": right_censored,
            },
        }
    finally:
        clean_session.close()
        counterfactual_session.close()


def run_offline_counterfactual_audit(
    game_dir: Path, *, sample_size: int = 10, horizon: int = 32
) -> dict[str, Any]:
    """Run the redesigned Exp6 without LLM calls or continuation duels."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    manifest = read_json(game_dir / "manifest.json")
    config = manifest["config"]
    rows = JsonlJournal(game_dir / "trajectory.jsonl").recover()
    found = find_offline_counterfactual_candidates(game_dir)
    candidates = found.pop("candidates")
    selected = sorted(
        random.Random(int(config["seed"])).sample(candidates, min(sample_size, len(candidates))),
        key=lambda candidate: (candidate["response_index"], candidate["candidate_id"]),
    )
    results = [_audit_candidate(config, rows, candidate, horizon) for candidate in selected]
    report = {
        "phase": 4,
        "mode": "offline_legal_counterfactual_plan_robustness",
        "status": "COMPLETED" if results else "INELIGIBLE",
        "reason": None if results else "no strict legal opponent-interruption alternatives",
        "selection": {
            "seed": int(config["seed"]),
            "sample_size_requested": sample_size,
            "sample_size_selected": len(selected),
            "action_sequence_horizon": horizon,
        },
        "candidate_inventory": found,
        "results": results,
    }
    atomic_write_json(game_dir / "exp6_offline_report.json", report)
    return report


def write_offline_exp6_metrics(
    run_dir: Path,
    game_dir: Path,
    *,
    reversibility: dict[str, Any],
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write Exp6-R descriptive metrics; no full-duel recovery claims are made."""
    results = audit.get("results", []) if audit else []
    completed = [row for row in results if row.get("status") == "COMPLETED"]
    survivals = [row["action_sequence_survival"]["survived_actions"] for row in completed]
    failures = [
        row["action_sequence_survival"]["first_failure"]
        for row in completed
        if row["action_sequence_survival"]["first_failure"] is not None
    ]
    failure_reasons = Counter(row["reason"] for row in failures)
    metrics = {
        "experiment": 6,
        "mode": "offline_legal_counterfactual_plan_robustness",
        "status": audit.get("status", "SKIPPED") if audit else "SKIPPED",
        "reversibility_gate": {
            "passed": bool(reversibility.get("passed")),
            "sample_size": int(reversibility.get("sample_size", 0)),
        },
        "candidate_inventory": audit.get("candidate_inventory") if audit else None,
        "sampled_counterfactuals": len(completed),
        "state_changed_rate": (
            sum(row["state_changed"] for row in completed) / len(completed)
            if completed
            else None
        ),
        "horizon_plan_invalidation_rate": {
            "numerator": len(failures),
            "denominator": len(completed),
            "estimate": len(failures) / len(completed) if completed else None,
            "direction": "lower_is_better",
        },
        "action_sequence_survival": {
            "median_steps": median(survivals) if survivals else None,
            "mean_steps": sum(survivals) / len(survivals) if survivals else None,
            "right_censored": sum(
                row["action_sequence_survival"]["right_censored"] for row in completed
            ),
            "horizon": audit.get("selection", {}).get("action_sequence_horizon") if audit else None,
        },
        "first_failure_reasons": dict(sorted(failure_reasons.items())),
        "ineligibility_reason": audit.get("reason") if audit else "reversibility gate failed",
        "full_duel_continuations": 0,
        "llm_calls": 0,
    }
    atomic_write_json(run_dir / "metrics" / "exp6" / "metrics.json", metrics)
    return metrics


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
