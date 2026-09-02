"""Resumable full-duel runner that writes public and oracle evidence separately."""

from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ygobench.agents.factory import create_agent
from ygobench.config import PROJECT_ROOT
from ygobench.engine.full_duel import _derive_deck_shuffle_seeds, _passive_fallback
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.experiments.config import ExperimentConfig, stable_id
from ygobench.experiments.io import (
    JsonlJournal,
    atomic_write_json,
    content_hash,
    read_json,
)
from ygobench.experiments.legal import exact_legal_action_match
from ygobench.experiments.oracle import build_oracle_state
from ygobench.experiments.registry import TaskRegistry
from ygobench.experiments.session import DuelSession

MAX_MODEL_ACTION_ATTEMPTS = 3


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or None


def _record_totals(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    totals: dict[str, list[Any]] = {
        "decisions": [0, 0],
        "illegal": [0, 0],
        "seconds": [0.0, 0.0],
        "model_calls": [0, 0],
        "input_tokens": [0, 0],
        "output_tokens": [0, 0],
    }
    for row in rows:
        player = int(row["player"])
        totals["decisions"][player] += 1
        totals["illegal"][player] += int(not row.get("validation", {}).get("valid", False))
        totals["seconds"][player] += float(row.get("elapsed_seconds", 0.0))
        for turn in row.get("trace", {}).get("turns", []):
            totals["model_calls"][player] += 1
            usage = turn.get("usage", {})
            for key in ("input_tokens", "output_tokens"):
                value = usage.get(key, 0)
                if isinstance(value, (int, float)):
                    totals[key][player] += int(value)
    return totals


def _trace_diagnostics(trace: dict[str, Any], expected: str) -> dict[str, Any]:
    turns = trace.get("turns", []) if isinstance(trace, dict) else []
    response_counts = [
        sum(call.get("name") == expected for call in turn.get("tool_calls", [])) for turn in turns
    ]
    errors = []
    if trace.get("fallback"):
        errors.append("missing_tool_call")
    if any(count > 1 for count in response_counts):
        errors.append("multiple_response_calls")
    for diagnostic in (
        trace.get("provider_protocol_diagnostics"),
        trace.get("error"),
    ):
        if isinstance(diagnostic, dict):
            errors.extend(str(value) for value in diagnostic.get("protocol_errors", []))
    return {
        "automatic": bool(trace.get("automatic")),
        "retry_count": max(0, len(turns) - 1),
        "fallback": bool(trace.get("fallback")),
        "protocol_errors": errors,
    }


def _reconcile(public: JsonlJournal, oracle: JsonlJournal) -> tuple[list[dict], list[dict]]:
    public_rows = public.recover()
    oracle_rows = oracle.recover()
    common = min(len(public_rows), len(oracle_rows))
    while common and public_rows[common - 1].get("decision_id") != oracle_rows[common - 1].get(
        "decision_id"
    ):
        common -= 1
    if len(public_rows) != common:
        public.rewrite(public_rows[:common])
    if len(oracle_rows) != common:
        oracle.rewrite(oracle_rows[:common])
    return public_rows[:common], oracle_rows[:common]


def _manifest(config: ExperimentConfig, game_dir: Path) -> dict[str, Any]:
    deck_root = PROJECT_ROOT / "resources" / "decks"
    prompt_root = PROJECT_ROOT / "ygobench" / "agents" / "prompts"
    upstream = PROJECT_ROOT / "vendor" / "yugi-bench"
    return {
        "schema_version": config.schema_version,
        "created_at": datetime.now(UTC).isoformat(),
        "config": config.to_dict(),
        "config_hash": config.hash(),
        "paths": {"game_dir": str(game_dir)},
        "provenance": {
            "ygobench_commit": _git_commit(PROJECT_ROOT),
            "yugi_bench_commit": _git_commit(upstream),
            "ocgcore_commit": _git_commit(upstream / "vendor" / "ygopro-core"),
            "cardscripts_commit": _git_commit(upstream / "vendor" / "distribution" / "script"),
            "puzzles_commit": _git_commit(upstream / "vendor" / "puzzles"),
            "deck_hashes": {
                config.deck1: _file_hash(deck_root / f"{config.deck1}.ydk"),
                config.deck2: _file_hash(deck_root / f"{config.deck2}.ydk"),
            },
            "prompt_hashes": {
                path.name: _file_hash(path) for path in sorted(prompt_root.glob("*.md"))
            },
        },
        "checkpoint_mode": "deterministic_action_prefix_replay",
        "native_engine_serialization": False,
        "randomization": {
            "engine_seed": config.seed,
            "deck_shuffle_seeds": list(_derive_deck_shuffle_seeds(config.seed)),
            "deck_shuffle_algorithm": "python_random_fisher_yates_v1",
            "llm_sampling": "provider_non_deterministic_unless_provider_seed_is_supported",
        },
    }


def run_evidence_duel(
    config: ExperimentConfig,
    *,
    root: Path | None = None,
    interventions: dict[int, ActionChoice] | None = None,
    stop_after_prefix: int | None = None,
    replay_prefix: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = root or PROJECT_ROOT / "bench_data" / "experiments"
    run_dir = root / config.run_id
    game_dir = config.game_dir(root)
    game_dir.mkdir(parents=True, exist_ok=True)
    public = JsonlJournal(game_dir / "trajectory.jsonl")
    oracle = JsonlJournal(game_dir / "oracle_trajectory.jsonl")
    public_rows, oracle_rows = _reconcile(public, oracle)
    outcome_path = game_dir / "outcome.json"
    existing_outcome = read_json(outcome_path)
    if existing_outcome and existing_outcome.get("game_over"):
        return existing_outcome

    registry = TaskRegistry(run_dir / "task_state.sqlite")
    registry.add(config.game_id, "full_duel", config.to_dict())
    task = registry.row(config.game_id)
    if task and task["status"] == "COMPLETED" and existing_outcome:
        return existing_outcome
    if not registry.claim(config.game_id):
        raise RuntimeError(f"task {config.game_id} is already running or completed")

    manifest_path = game_dir / "manifest.json"
    if not manifest_path.exists():
        atomic_write_json(manifest_path, _manifest(config, game_dir))
    atomic_write_json(
        game_dir / "status.json",
        {"status": "RUNNING", "committed_decisions": len(public_rows)},
    )

    deck_root = PROJECT_ROOT / "resources" / "decks"
    agents = (
        create_agent(config.agent1, seed=config.seed * 2),
        create_agent(config.agent2, seed=config.seed * 2 + 1),
    )
    for agent in agents:
        agent.reset()
    session: DuelSession | None = None
    recent_actions: list[dict[str, Any]] = []
    termination = "decision_budget_exhausted"
    forfeit_winner: int | None = None
    started = time.perf_counter()
    interventions = interventions or {}
    replay_rows = (
        [*replay_prefix, *public_rows] if replay_prefix is not None else public_rows
    )
    external_prefix_totals = (
        _record_totals(replay_rows) if replay_prefix is not None else _record_totals([])
    )
    previous_commit = public_rows[-1].get("commit_hash", "") if public_rows else ""
    try:
        session = DuelSession(
            deck_root / f"{config.deck1}.ydk",
            deck_root / f"{config.deck2}.ydk",
            seed=config.seed,
        )
        decision_index = 0
        while not session.done and (
            config.max_decisions <= 0 or decision_index < config.max_decisions
        ):
            view = session.view(recent_actions)
            player = view["player"]
            before = view["oracle"]
            turn_before = int(session.duel.state.turn_count)

            # Deterministically replay already committed decisions without API calls.
            if decision_index < len(replay_rows):
                recorded = replay_rows[decision_index]
                if before["state_hash"] != recorded["oracle_before_hash"]:
                    raise RuntimeError(
                        f"resume hash mismatch at decision {decision_index + 1}: "
                        f"{before['state_hash']} != {recorded['oracle_before_hash']}"
                    )
                action = ActionChoice(**recorded["executed_action"])
                session.execute(action)
                after = build_oracle_state(session)
                if after["state_hash"] != recorded["oracle_after_hash"]:
                    raise RuntimeError(
                        f"resume transition mismatch at decision {decision_index + 1}"
                    )
                recent_actions.append(recorded["action_summary"])
                decision_index += 1
                continue

            if stop_after_prefix is not None and decision_index >= stop_after_prefix:
                termination = "prefix_verified"
                break

            expected = view["legal"]["expected_responder"]
            request = DecisionRequest(
                player=player,
                observation=view["observation"],
                legal_actions=view["actions"],
                decision_type=expected,
                legal_actions_complete=bool(view["legal"]["enumeration_complete"]),
            )
            agent = agents[player]
            attempted: ActionChoice | None = None
            executed: ActionChoice | None = None
            agent_error: str | None = None
            terminal_model_failure = False
            model_action_attempts = 0
            call_started = time.perf_counter()
            try:
                model_action_attempts = 1
                attempted = interventions.get(decision_index) or agent.predict(request)
            except Exception as exc:  # noqa: BLE001
                agent_error = f"{type(exc).__name__}: {exc}"
                error_type = type(exc).__name__
                previous_trace = getattr(agent, "last_trace", {})
                if isinstance(previous_trace, dict):
                    provider_diagnostics = getattr(exc, "diagnostics", None)
                    model_action_attempts = max(
                        model_action_attempts,
                        int(
                            provider_diagnostics.get("model_action_attempts", 0)
                            if isinstance(provider_diagnostics, dict)
                            else 0
                        ),
                        len(previous_trace.get("transport_attempts", [])),
                    )
                trace = {
                    **(previous_trace if isinstance(previous_trace, dict) else {}),
                    "fallback": False,
                    "exception": agent_error,
                    "provider_failure_type": error_type,
                    "terminal_model_failure": True,
                }
                if error_type == "ProviderProtocolError":
                    trace["provider_protocol_diagnostics"] = getattr(exc, "diagnostics", None)
                if getattr(exc, "provider_failure_type", None) == "provider_call":
                    trace["provider_transport_attempts"] = getattr(exc, "attempts", [])
                terminal_model_failure = True
                forfeit_winner = 1 - player
                termination = "model_retry_exhausted_forfeit"
            else:
                executed = attempted
                if attempted.tool != expected:
                    agent_error = f"wrong_responder: expected {expected}, got {attempted.tool}"
                trace = (
                    getattr(agent, "last_trace", {})
                    if decision_index not in interventions
                    else {"intervention": True}
                )
            attempted_invalid = agent_error is not None
            recovery = "model_retry_exhausted_forfeit" if terminal_model_failure else "none"
            elapsed = time.perf_counter() - call_started
            diagnostics = _trace_diagnostics(trace, expected)

            exact_legal_check_performed = bool(view["legal"]["enumeration_complete"])
            validation_scope = "exact" if exact_legal_check_performed else "engine_only"
            pre_engine_error: str | None = None
            engine_error: str | None = None
            correction_trace: dict[str, Any] | None = None
            exact_legal_match = True
            if not terminal_model_failure and exact_legal_check_performed and executed is not None:
                try:
                    exact_legal_match = exact_legal_action_match(
                        executed,
                        view["actions"],
                        signature=session.action_signature,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    exact_legal_match = False
                    pre_engine_error = (
                        "action_argument_validation_failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            needs_model_correction = (
                not terminal_model_failure
                and (
                    agent_error is not None
                    or (exact_legal_check_performed and not exact_legal_match)
                )
            )
            if needs_model_correction:
                if pre_engine_error is None and exact_legal_check_performed:
                    pre_engine_error = (
                        "action_not_in_exact_legal_set: "
                        f"{executed.tool} {executed.arguments!r}"
                    )
                attempted_invalid = True
                correction_trace = {"attempts": []}
                for correction_number in range(2, MAX_MODEL_ACTION_ATTEMPTS + 1):
                    model_action_attempts = correction_number
                    corrected: ActionChoice | None = None
                    attempt_trace: dict[str, Any]
                    if decision_index in interventions or not hasattr(agent, "correct_invalid_action"):
                        attempt_trace = {"error": "no_model_correction_available"}
                    else:
                        try:
                            corrected, attempt_trace = agent.correct_invalid_action(request)
                        except Exception as exc:  # noqa: BLE001
                            attempt_trace = {"exception": f"{type(exc).__name__}: {exc}"}
                    correction_trace["attempts"].append(
                        {
                            "attempt": correction_number,
                            "action": asdict(corrected) if corrected is not None else None,
                            "trace": attempt_trace,
                        }
                    )
                    correction_is_legal = (
                        corrected is not None
                        and corrected.tool == expected
                        and (
                            not exact_legal_check_performed
                            or exact_legal_action_match(
                                corrected,
                                view["actions"],
                                signature=session.action_signature,
                            )
                        )
                    )
                    if correction_is_legal:
                        executed = corrected
                        recovery = "corrected_retry"
                        break
                else:
                    executed = None
                    terminal_model_failure = True
                    recovery = "model_retry_exhausted_forfeit"
                    forfeit_winner = 1 - player
                    termination = "model_retry_exhausted_forfeit"

            if terminal_model_failure:
                after = before
                step = None
            else:
                assert executed is not None
                try:
                    step = session.execute(executed)
                except Exception as exc:  # noqa: BLE001
                    engine_error = f"{type(exc).__name__}: {exc}"
                    attempted_invalid = True
                    executed = _passive_fallback(session.duel.pending, session.replay_module)
                    recovery = "deterministic_fallback_after_engine_error"
                    try:
                        step = session.execute(executed)
                        engine_error = None
                    except Exception as fallback_exc:  # noqa: BLE001
                        engine_error = f"fallback_failed: {type(fallback_exc).__name__}: {fallback_exc}"
                        termination = "engine_rejection_abort"
                        step = None
                after = build_oracle_state(session)
            decision_id = f"d{decision_index + 1:06d}"
            action_id = stable_id("action", [config.game_id, decision_id])
            action_summary = {
                "decision_id": decision_id,
                "action_id": action_id,
                "turn": turn_before,
                "player": player,
                "tool": executed.tool if executed is not None else None,
                "arguments": executed.arguments if executed is not None else None,
            }
            record = {
                "type": "decision",
                "schema_version": config.schema_version,
                "run_id": config.run_id,
                "game_id": config.game_id,
                "decision_id": decision_id,
                "action_id": action_id,
                "decision_index": decision_index + 1,
                "turn": turn_before,
                "phase": view["observation"].get("phase"),
                "player": player,
                "observation": view["observation"],
                "legal": view["legal"],
                "attempted_action": asdict(attempted) if attempted else None,
                "executed_action": asdict(executed) if executed is not None else None,
                "action_summary": action_summary,
                "validation": {
                    "valid": (
                        agent_error is None
                        and not attempted_invalid
                        and engine_error is None
                        and not terminal_model_failure
                        and not diagnostics["protocol_errors"]
                    ),
                    "agent_error": agent_error,
                    "pre_engine_error": pre_engine_error,
                    "attempted_invalid": attempted_invalid,
                    "recovery": recovery,
                    "model_action_attempts": model_action_attempts,
                    "terminal_model_failure": terminal_model_failure,
                    "validation_scope": validation_scope,
                    "exact_legal_check_performed": exact_legal_check_performed,
                    "engine_submission_attempted": (
                        pre_engine_error is None and executed is not None and not terminal_model_failure
                    ),
                    "recovery_engine_submission_attempted": step is not None,
                    "engine_error": engine_error,
                    **diagnostics,
                },
                "trace": trace,
                "correction_trace": correction_trace,
                "engine_events": step.events if step is not None else [],
                "elapsed_seconds": round(elapsed, 6),
                "oracle_before_hash": before["state_hash"],
                "oracle_after_hash": after["state_hash"],
                "previous_commit_hash": previous_commit,
            }
            record["commit_hash"] = content_hash(record)
            oracle_record = {
                "type": "oracle_decision",
                "run_id": config.run_id,
                "game_id": config.game_id,
                "decision_id": decision_id,
                "before": before,
                "after": after,
                "public_commit_hash": record["commit_hash"],
            }
            oracle.append(oracle_record)
            public.append(record)
            previous_commit = record["commit_hash"]
            if executed is not None:
                recent_actions.append(action_summary)
            decision_index += 1
            if config.checkpoint_interval > 0 and decision_index % config.checkpoint_interval == 0:
                atomic_write_json(
                    game_dir / "checkpoints" / f"checkpoint_{decision_index:06d}.json",
                    {
                        "mode": "deterministic_action_prefix_replay",
                        "decision_count": decision_index,
                        "decision_id": decision_id,
                        "oracle_after_hash": after["state_hash"],
                        "prefix_hash": content_hash(
                            [row["executed_action"] for row in public.recover()]
                        ),
                    },
                )
            atomic_write_json(
                game_dir / "status.json",
                {"status": "RUNNING", "committed_decisions": decision_index},
            )
            registry.renew(config.game_id)
            if terminal_model_failure or engine_error:
                break

        if forfeit_winner is None:
            if session.duel.state.game_over:
                termination = "game_over"
            elif session.duel.pending is None:
                termination = "no_pending_decision"
        winner = forfeit_winner if forfeit_winner is not None else session.duel.state.winner
        game_over = bool(session.duel.state.game_over or forfeit_winner is not None)
        final_rows = public.recover()
        recovered_invalid_action = any(
            bool(row.get("validation", {}).get("attempted_invalid")) for row in final_rows
        )
        final_totals = _record_totals(final_rows)
        full_decisions_by_player = [
            int(final_totals["decisions"][seat])
            + int(external_prefix_totals["decisions"][seat])
            for seat in (0, 1)
        ]
        full_illegal = [
            int(final_totals["illegal"][seat])
            + int(external_prefix_totals["illegal"][seat])
            for seat in (0, 1)
        ]
        outcome = {
            "schema_version": config.schema_version,
            "run_id": config.run_id,
            "game_id": config.game_id,
            "termination": termination,
            "game_over": game_over,
            "competitive_eligible": bool(
                termination == "game_over" and not recovered_invalid_action
            ),
            "recovered_invalid_action": recovered_invalid_action,
            "winner": winner,
            "agents": [config.agent1, config.agent2],
            "decks": [config.deck1, config.deck2],
            "seed": config.seed,
            "randomization": {
                "engine_seeds": list(session.engine_seeds),
                "deck_shuffle_seeds": list(session.deck_shuffle_seeds),
                "deck_order_hashes": list(session.deck_order_hashes),
            },
            "turn_count": session.duel.state.turn_count,
            "decisions": decision_index,
            "decisions_by_player": full_decisions_by_player,
            "illegal_actions": full_illegal,
            "decision_seconds": [round(value, 6) for value in final_totals["seconds"]],
            "decision_seconds_scope": "executed_suffix_excludes_replayed_prefix",
            "replayed_prefix_decisions": sum(external_prefix_totals["decisions"]),
            "measured_suffix_decisions": len(final_rows),
            "model_usage_totals": {
                "model_calls": final_totals["model_calls"],
                "input_tokens": final_totals["input_tokens"],
                "output_tokens": final_totals["output_tokens"],
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        atomic_write_json(outcome_path, outcome)
        atomic_write_json(
            game_dir / "status.json",
            {"status": "COMPLETED", "committed_decisions": decision_index},
        )
        registry.finish(config.game_id)
        return outcome
    except Exception as exc:
        registry.finish(config.game_id, error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(
            game_dir / "status.json",
            {
                "status": "FAILED_RETRYABLE",
                "committed_decisions": len(public.recover()),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    finally:
        if session is not None:
            session.close()
