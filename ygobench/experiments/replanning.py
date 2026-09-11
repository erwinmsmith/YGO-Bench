"""Phase-4 offline legal counterfactual plan-robustness audit."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from ygobench.config import PROJECT_ROOT
from ygobench.engine.protocol import ActionChoice
from ygobench.experiments.config import stable_id
from ygobench.experiments.io import JsonlJournal, atomic_write_json, content_hash, read_json
from ygobench.experiments.legal import exact_legal_action_match
from ygobench.experiments.oracle import build_oracle_state
from ygobench.experiments.provenance import metric_provenance
from ygobench.experiments.session import DuelSession
from ygobench.experiments.statistics import cluster_bootstrap, kaplan_meier

_COMMITMENT_COMMANDS = {"activate", "summon", "sp_summon", "attack"}
_NEW_DECISION_BOUNDARIES = {"select_idlecmd", "select_battlecmd", "rock_paper_scissors"}


def _survival_observation(result: dict[str, Any], *, default_horizon: int | None) -> dict[str, Any]:
    """Normalize current and legacy Exp6 records into event/censor rows."""
    survival = result["action_sequence_survival"]
    survived = int(survival.get("survived_actions", 0))
    failure = survival.get("first_failure")
    if failure is not None:
        return {
            "time": int(survival.get("event_time", survived + 1)),
            "event_observed": True,
            "censor_reason": None,
        }
    horizon = survival.get("horizon", default_horizon)
    censor_reason = survival.get("censor_reason")
    if censor_reason is None:
        censor_reason = (
            "horizon_reached"
            if horizon is not None and survived >= int(horizon)
            else "trajectory_exhausted"
        )
    return {
        "time": int(survival.get("censor_time", survived)),
        "event_observed": False,
        "censor_reason": censor_reason,
    }


def kaplan_meier_action_survival(
    results: list[dict[str, Any]], *, horizon: int | None
) -> dict[str, Any]:
    observations = [_survival_observation(result, default_horizon=horizon) for result in results]
    analysis = kaplan_meier(observations, horizon=horizon)
    censor_reasons = Counter(
        row["censor_reason"] for row in observations if not row["event_observed"]
    )
    analysis["time_unit"] = "subsequent_logged_actions"
    analysis["censor_reasons"] = dict(sorted(censor_reasons.items()))
    return analysis


def _survival_analyses(completed: list[dict[str, Any]], *, horizon: int | None) -> dict[str, Any]:
    changed = [row for row in completed if bool(row.get("state_changed"))]
    return {
        "method": "Kaplan-Meier with Greenwood pointwise confidence intervals",
        "primary_population": "completed candidates with state_changed=true",
        "primary": kaplan_meier_action_survival(changed, horizon=horizon),
        "sensitivity_all_completed": kaplan_meier_action_survival(completed, horizon=horizon),
    }


def _failure_reason_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reasons = Counter(
        row["action_sequence_survival"]["first_failure"]["reason"]
        for row in results
        if row.get("action_sequence_survival", {}).get("first_failure")
    )
    events = sum(reasons.values())
    return [
        {
            "reason": reason,
            "count": count,
            "proportion_of_completed_candidates": count / len(results) if results else None,
            "proportion_of_events": count / events if events else None,
        }
        for reason, count in sorted(reasons.items())
    ]


def _survival_cluster_bootstrap(
    results: list[dict[str, Any]], *, horizon: int | None
) -> dict[str, Any]:
    changed = [row for row in results if row.get("state_changed") and row.get("game_id")]

    def survival_at_horizon(sampled: list[dict[str, Any]]) -> float | None:
        analysis = kaplan_meier_action_survival(sampled, horizon=horizon)
        fixed = analysis.get("fixed_window") or {}
        return fixed.get("survival_probability", {}).get("estimate")

    def rmst_at_horizon(sampled: list[dict[str, Any]]) -> float | None:
        analysis = kaplan_meier_action_survival(sampled, horizon=horizon)
        return (analysis.get("rmst") or {}).get("estimate")

    return {
        "cluster": "duel",
        "survival_at_horizon": cluster_bootstrap(
            changed, cluster_key="game_id", statistic=survival_at_horizon
        ),
        "rmst_at_horizon": cluster_bootstrap(
            changed, cluster_key="game_id", statistic=rmst_at_horizon
        ),
    }


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


def _counterfactual_action_summary(row: dict[str, Any], action: ActionChoice) -> dict[str, Any]:
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
        available_follow_up = max(0, len(rows) - response_index - 1)
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
        event_observed = first_failure is not None
        censor_reason = None
        if not event_observed:
            censor_reason = (
                "trajectory_exhausted" if available_follow_up <= horizon else "horizon_reached"
            )
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
                "event_observed": event_observed,
                "event_time": survived_actions + 1 if event_observed else None,
                "right_censored": not event_observed,
                "censor_time": survived_actions if not event_observed else None,
                "censor_reason": censor_reason,
            },
        }
    finally:
        clean_session.close()
        counterfactual_session.close()


def run_offline_counterfactual_audit(
    game_dir: Path, *, sample_size: int = 10, horizon: int = 32
) -> dict[str, Any]:
    """Run Exp6 without LLM calls, resuming after each completed candidate."""
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
    audit_id = content_hash(
        {
            "protocol": "offline_legal_counterfactual_plan_robustness",
            "game_id": config["game_id"],
            "seed": int(config["seed"]),
            "horizon": horizon,
            "candidate_ids": [candidate["candidate_id"] for candidate in selected],
        }
    )
    journal = JsonlJournal(game_dir / "exp6_offline_results.jsonl")
    completed_by_candidate = {
        str(record["candidate_id"]): record["result"]
        for record in journal.recover()
        if record.get("audit_id") == audit_id
        and isinstance(record.get("candidate_id"), str)
        and isinstance(record.get("result"), dict)
        and record["result"].get("status") == "COMPLETED"
    }
    resumed_candidates = len(completed_by_candidate)
    for candidate in selected:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in completed_by_candidate:
            continue
        result = _audit_candidate(config, rows, candidate, horizon)
        journal.append(
            {
                "schema_version": "2.0.0",
                "audit_id": audit_id,
                "candidate_id": candidate_id,
                "result": result,
            }
        )
        completed_by_candidate[candidate_id] = result
    results = [completed_by_candidate[str(candidate["candidate_id"])] for candidate in selected]
    report = {
        "schema_version": "2.0.0",
        "phase": 4,
        "mode": "offline_legal_counterfactual_plan_robustness",
        "status": "COMPLETED" if results else "INELIGIBLE",
        "reason": None if results else "no strict legal opponent-interruption alternatives",
        "selection": {
            "audit_id": audit_id,
            "seed": int(config["seed"]),
            "sample_size_requested": sample_size,
            "sample_size_selected": len(selected),
            "action_sequence_horizon": horizon,
        },
        "resume": {
            "journal": "exp6_offline_results.jsonl",
            "candidates_reused": resumed_candidates,
            "candidates_computed": len(results) - resumed_candidates,
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
    """Write censor-aware Exp6-R survival metrics; no LLM calls are made."""
    results = audit.get("results", []) if audit else []
    completed = [row for row in results if row.get("status") == "COMPLETED"]
    failures = [
        row["action_sequence_survival"]["first_failure"]
        for row in completed
        if row["action_sequence_survival"]["first_failure"] is not None
    ]
    failure_reasons = Counter(row["reason"] for row in failures)
    horizon = audit.get("selection", {}).get("action_sequence_horizon") if audit else None
    metrics = {
        "schema_version": "2.0.0",
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
            sum(row["state_changed"] for row in completed) / len(completed) if completed else None
        ),
        "action_sequence_survival": _survival_analyses(completed, horizon=horizon),
        "first_failure_reasons": dict(sorted(failure_reasons.items())),
        "first_failure_reason_summary": _failure_reason_summary(completed),
        "ineligibility_reason": audit.get("reason") if audit else "reversibility gate failed",
        "full_duel_continuations": 0,
        "llm_calls": 0,
    }
    atomic_write_json(run_dir / "metrics" / "exp6" / "metrics.json", metrics)
    return metrics


def aggregate_offline_exp6_metrics(run_dir: Path) -> dict[str, Any]:
    """Aggregate the offline Exp6 reports for every game in one run.

    Per-game phase-4 reports remain the authoritative audit artifacts.  This
    function creates the run-level metric without rerunning a duel or making
    model calls, so it is safe to invoke after any subset of games has finished.
    """
    per_game: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    completed_by_model: dict[str, list[dict[str, Any]]] = {}
    failure_reasons: Counter[str] = Counter()
    commitment_points = eligible_counterfactuals = 0
    missing_reports: list[str] = []

    for manifest_path in sorted((run_dir / "games").glob("*/manifest.json")):
        game_dir = manifest_path.parent
        game_id = game_dir.name
        reversibility = read_json(game_dir / "reversibility_report.json")
        audit = read_json(game_dir / "exp6_offline_report.json")
        if not audit:
            missing_reports.append(game_id)
            per_game.append(
                {
                    "game_id": game_id,
                    "status": "MISSING",
                    "reversibility_passed": bool(reversibility.get("passed")),
                    "sampled_counterfactuals": 0,
                }
            )
            continue

        inventory = audit.get("candidate_inventory", {})
        manifest = read_json(game_dir / "manifest.json", {})
        policy_identities = manifest.get("policy_identities", [])
        commitment_points += int(inventory.get("commitment_points", 0))
        eligible_counterfactuals += int(inventory.get("eligible_counterfactuals", 0))
        game_completed = [
            result for result in audit.get("results", []) if result.get("status") == "COMPLETED"
        ]
        for result in game_completed:
            focal_player = int(result.get("candidate", {}).get("focal_player", 0))
            identity = (
                policy_identities[focal_player] if focal_player < len(policy_identities) else {}
            )
            enriched = {
                **result,
                "game_id": game_id,
                "policy_id": identity.get("policy_id"),
                "model_configuration_id": identity.get(
                    "model_configuration_id", identity.get("policy_id", "legacy_unknown")
                ),
            }
            completed.append(enriched)
            completed_by_model.setdefault(enriched["model_configuration_id"], []).append(enriched)
        failures = [
            result["action_sequence_survival"]["first_failure"]
            for result in game_completed
            if result["action_sequence_survival"].get("first_failure") is not None
        ]
        failure_reasons.update(failure["reason"] for failure in failures)
        per_game.append(
            {
                "game_id": game_id,
                "status": audit.get("status", "SKIPPED"),
                "reversibility_passed": bool(reversibility.get("passed")),
                "reversibility_sample_size": int(reversibility.get("sample_size", 0)),
                "candidate_inventory": inventory,
                "sampled_counterfactuals": len(game_completed),
                "ineligibility_reason": audit.get("reason"),
            }
        )

    horizons = {
        int(result["action_sequence_survival"]["horizon"])
        for result in completed
        if result["action_sequence_survival"].get("horizon") is not None
    }
    horizon = min(horizons) if horizons else None
    metrics = {
        "schema_version": "2.0.0",
        "experiment": 6,
        "mode": "offline_legal_counterfactual_plan_robustness",
        "status": "COMPLETED" if not missing_reports else "PARTIAL",
        "games_observed": len(per_game),
        "games_with_missing_report": missing_reports,
        "reversibility_gate": {
            "passed_games": sum(row["reversibility_passed"] for row in per_game),
            "observed_games": len(per_game),
        },
        "candidate_inventory": {
            "commitment_points": commitment_points,
            "eligible_counterfactuals": eligible_counterfactuals,
        },
        "sampled_counterfactuals": len(completed),
        "state_changed_rate": (
            sum(result["state_changed"] for result in completed) / len(completed)
            if completed
            else None
        ),
        "action_sequence_survival": _survival_analyses(completed, horizon=horizon),
        "cluster_bootstrap": _survival_cluster_bootstrap(completed, horizon=horizon),
        "by_model_configuration": [
            {
                "model_configuration_id": model_configuration,
                "sampled_counterfactuals": len(model_results),
                "state_changed": sum(bool(row.get("state_changed")) for row in model_results),
                "action_sequence_survival": _survival_analyses(model_results, horizon=horizon),
                "first_failure_reasons": dict(
                    sorted(
                        Counter(
                            row["action_sequence_survival"]["first_failure"]["reason"]
                            for row in model_results
                            if row["action_sequence_survival"].get("first_failure")
                        ).items()
                    )
                ),
                "first_failure_reason_summary": _failure_reason_summary(model_results),
                "cluster_bootstrap": _survival_cluster_bootstrap(model_results, horizon=horizon),
            }
            for model_configuration, model_results in sorted(completed_by_model.items())
        ],
        "first_failure_reasons": dict(sorted(failure_reasons.items())),
        "first_failure_reason_summary": _failure_reason_summary(completed),
        "full_duel_continuations": 0,
        "llm_calls": 0,
        "per_game": per_game,
        "provenance": metric_provenance(run_dir),
    }
    atomic_write_json(run_dir / "metrics" / "exp6" / "metrics.json", metrics)
    return metrics
