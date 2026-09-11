"""Experiment 7 model-capability matrix and internal association analysis."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any

from ygobench.experiments.calculations import (
    arena_metrics,
    execution_metrics,
    forecast_metrics,
    long_chain_metrics,
    state_metrics,
)
from ygobench.experiments.identity import outcome_policy_id
from ygobench.experiments.io import JsonlJournal, atomic_write_json, read_json
from ygobench.experiments.provenance import metric_provenance
from ygobench.experiments.replanning import kaplan_meier_action_survival
from ygobench.experiments.statistics import percentile, spearman

CAPABILITIES = {
    "glicko2_mu": "higher_is_better",
    "long_chain_survival_at_20": "higher_is_better",
    "state_jga": "higher_is_better",
    "illegal_action_rate": "lower_is_better",
    "availability_brier": "lower_is_better",
    "plan_survival_at_horizon": "higher_is_better",
}

AUXILIARY_CAPABILITIES = {
    "side_swapped_win_rate": "higher_is_better",
    "glicko2_rd": "lower_is_better",
    "long_chain_km_median": "higher_is_better",
    "state_slot_accuracy": "higher_is_better",
    "state_micro_f1": "higher_is_better",
    "state_macro_f1": "higher_is_better",
    "first_attempt_success_rate": "higher_is_better",
    "retry_exhausted_rate": "lower_is_better",
    "availability_auprc": "higher_is_better",
    "availability_auroc": "higher_is_better",
    "availability_ece": "lower_is_better",
    "behavior_brier": "lower_is_better",
    "behavior_auprc": "higher_is_better",
    "plan_failure_at_horizon": "lower_is_better",
    "plan_km_median": "higher_is_better",
    "plan_rmst": "higher_is_better",
}


def _fixed_survival(analysis: dict[str, Any] | None) -> float | None:
    fixed = (analysis or {}).get("fixed_window")
    if not fixed:
        return None
    return fixed.get("survival_probability", {}).get("estimate")


def _metric_evaluators(path: Path) -> list[dict[str, Any]]:
    value = read_json(path, {})
    return [item for _, item in sorted((value.get("evaluators") or {}).items())]


def build_capability_matrix(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exp1 = read_json(run_dir / "metrics" / "exp1" / "metrics.json", {})
    exp2 = read_json(run_dir / "metrics" / "exp2" / "metrics.json", {})
    exp4 = read_json(run_dir / "metrics" / "exp4" / "metrics.json", {})
    exp6 = read_json(run_dir / "metrics" / "exp6" / "metrics.json", {})
    policies = {
        row["policy"]: {
            "policy_id": row["policy"],
            "model_configuration_id": row.get("model_configuration_id"),
            "agent": row.get("agent"),
            "strict_duels": row.get("strict_games", 0),
            "glicko2_mu": (row.get("glicko2") or {}).get("mu"),
            "side_swapped_win_rate": row.get("overall_win_rate"),
            "glicko2_rd": (row.get("glicko2") or {}).get("rd_phi"),
        }
        for row in exp1.get("leaderboard", [])
    }
    missing: list[dict[str, Any]] = []
    for row in exp2.get("kaplan_meier", []):
        policy = row.get("policy")
        if policy in policies:
            policies[policy]["long_chain_survival_at_20"] = _fixed_survival(row)
            policies[policy]["long_chain_km_median"] = (row.get("km_median") or {}).get("estimate")
    for row in exp2.get("illegal_action_rate_by_decision_depth", []):
        policy = row.get("policy")
        if policy in policies:
            label = str(row.get("depth_bin", "unknown")).replace("+", "plus")
            policies[policy][f"depth_{label}_illegal_rate"] = row.get("illegal_action_rate")
    for row in exp2.get("complexity_stratified", []):
        policy = row.get("policy")
        if policy in policies:
            label = str(row.get("quartile", "unknown")).lower()
            policies[policy][f"{label}_win_rate"] = row.get("win_rate")
            policies[policy][f"{label}_completion_rate"] = row.get("completion_rate")
            policies[policy][f"{label}_illegal_rate"] = row.get("illegal_action_rate")
    for row in exp4.get("by_agent", []):
        policy = row.get("policy")
        if policy in policies:
            policies[policy]["illegal_action_rate"] = row.get("illegal_action_rate")
            policies[policy]["first_attempt_success_rate"] = row.get("first_attempt_success_rate")
            policies[policy]["retry_exhausted_rate"] = row.get("retry_exhausted_rate")

    by_model_config: dict[str, list[str]] = {}
    for policy, row in policies.items():
        by_model_config.setdefault(str(row.get("model_configuration_id")), []).append(policy)

    for experiment, field, metric_path in (
        (3, "state_jga", run_dir / "metrics" / "exp3" / "metrics.json"),
        (5, "availability_brier", run_dir / "metrics" / "exp5" / "metrics.json"),
    ):
        for evaluator in _metric_evaluators(metric_path):
            identity = evaluator.get("evaluator_identity") or {}
            model_config = str(identity.get("model_configuration_id"))
            targets = by_model_config.get(model_config, [])
            if len(targets) != 1:
                missing.append(
                    {
                        "experiment": experiment,
                        "model_configuration_id": model_config,
                        "reason": "evaluator_policy_mismatch_or_ambiguous_target",
                    }
                )
                continue
            value = (
                evaluator.get("jga")
                if experiment == 3
                else evaluator.get("availability", {}).get("brier_score")
            )
            policies[targets[0]][field] = value
            if experiment == 3:
                policies[targets[0]]["state_slot_accuracy"] = evaluator.get("slot_accuracy")
                policies[targets[0]]["state_micro_f1"] = evaluator.get("slot_f1", {}).get(
                    "micro_f1"
                )
                policies[targets[0]]["state_macro_f1"] = evaluator.get("slot_f1", {}).get(
                    "macro_f1"
                )
            else:
                availability = evaluator.get("availability", {})
                behavior = evaluator.get("behavior_given_availability", {})
                policies[targets[0]]["availability_auprc"] = availability.get("auprc")
                policies[targets[0]]["availability_auroc"] = availability.get("auroc")
                policies[targets[0]]["availability_ece"] = availability.get("ece_10_equal_width")
                policies[targets[0]]["behavior_brier"] = behavior.get("brier_score")
                policies[targets[0]]["behavior_auprc"] = behavior.get("auprc")

    for row in exp6.get("by_model_configuration", []):
        model_config = str(row.get("model_configuration_id"))
        targets = by_model_config.get(model_config, [])
        if len(targets) != 1:
            continue
        analysis = row.get("action_sequence_survival", {}).get("primary", {})
        policies[targets[0]]["plan_survival_at_horizon"] = _fixed_survival(analysis)
        policies[targets[0]]["plan_rmst"] = (analysis.get("rmst") or {}).get("estimate")
        fixed = analysis.get("fixed_window") or {}
        policies[targets[0]]["plan_failure_at_horizon"] = fixed.get(
            "cumulative_failure_probability", {}
        ).get("estimate")
        policies[targets[0]]["plan_km_median"] = (analysis.get("km_median") or {}).get("estimate")
        policies[targets[0]]["plan_failure_reasons"] = row.get("first_failure_reasons", {})

    for row in policies.values():
        for capability in CAPABILITIES:
            if row.get(capability) is None:
                missing.append(
                    {
                        "policy_id": row["policy_id"],
                        "metric": capability,
                        "reason": "metric_missing_or_not_estimable",
                    }
                )
    return [policies[key] for key in sorted(policies)], missing


def _correlation(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    overlap = [row for row in rows if row.get(left) is not None and row.get(right) is not None]
    estimate = spearman(
        [float(row[left]) for row in overlap], [float(row[right]) for row in overlap]
    )
    return {
        "left": left,
        "right": right,
        "n_models": len(overlap),
        "models": [row["policy_id"] for row in overlap],
        "strict_duels_by_model": {row["policy_id"]: row.get("strict_duels", 0) for row in overlap},
        "spearman_rho": estimate,
        "status": "COMPLETED"
        if estimate is not None and len(overlap) >= 3
        else "INSUFFICIENT_DATA",
        "confidence_interval_95": None,
        "cluster_bootstrap": {
            "status": "NOT_ESTIMABLE_FROM_MODEL_LEVEL_AGGREGATES",
            "required_cluster": "seed x seat-swapped pair x deck matchup",
            "reason": (
                "raw-cluster recomputation is required; model rows are not independent clusters"
            ),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _cluster_id(outcome: dict[str, Any]) -> str:
    identities = outcome.get("policy_identities") or []
    models = []
    for seat in (0, 1):
        identity = identities[seat] if seat < len(identities) else {}
        model = identity.get("model_configuration_id") or outcome_policy_id(outcome, seat)
        models.append(str(model))
    decks = sorted(str(deck) for deck in outcome["decks"])
    return (
        f"seed={outcome.get('seed')}|models={'__vs__'.join(sorted(models))}"
        f"|decks={'__vs__'.join(decks)}"
    )


def _bootstrap_inputs(
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    games = []
    for manifest_path in sorted((run_dir / "games").glob("*/manifest.json")):
        game_dir = manifest_path.parent
        manifest = read_json(manifest_path, {})
        outcome = read_json(game_dir / "outcome.json")
        if not outcome:
            continue
        if not outcome.get("policy_identities"):
            outcome["policy_identities"] = manifest.get("policy_identities", [])
        games.append(
            {
                "game_id": game_dir.name,
                "cluster_id": _cluster_id(outcome),
                "outcome": outcome,
                "rows": JsonlJournal(game_dir / "trajectory.jsonl").recover(),
                "exp6": read_json(game_dir / "exp6_offline_report.json", {}),
                "policy_identities": manifest.get("policy_identities", []),
            }
        )
    state_by_evaluator: dict[str, list[dict[str, Any]]] = {}
    forecast_by_evaluator: dict[str, list[dict[str, Any]]] = {}
    for evaluator_dir in sorted((run_dir / "derived" / "probe_results").glob("*")):
        if not evaluator_dir.is_dir():
            continue
        state_by_evaluator[evaluator_dir.name] = JsonlJournal(
            evaluator_dir / "state_probe_results.jsonl"
        ).recover()
        forecast_by_evaluator[evaluator_dir.name] = JsonlJournal(
            evaluator_dir / "forecast_results.jsonl"
        ).recover()
    return games, state_by_evaluator, forecast_by_evaluator


def _resampled_capability_matrix(
    games: list[dict[str, Any]],
    state_by_evaluator: dict[str, list[dict[str, Any]]],
    forecast_by_evaluator: dict[str, list[dict[str, Any]]],
    sampled_clusters: list[str],
) -> list[dict[str, Any]]:
    games_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for game in games:
        games_by_cluster[game["cluster_id"]].append(game)
    selected_games: list[tuple[dict[str, Any], str]] = []
    for replicate_index, cluster in enumerate(sampled_clusters):
        for game in games_by_cluster[cluster]:
            selected_games.append((game, f"boot{replicate_index}:{game['game_id']}"))

    outcomes, decision_rows = [], []
    multiplicity: dict[str, list[str]] = defaultdict(list)
    for game, cloned_id in selected_games:
        outcome = deepcopy(game["outcome"])
        outcome["game_id"] = cloned_id
        outcomes.append(outcome)
        multiplicity[game["game_id"]].append(cloned_id)
        for source in game["rows"]:
            row = deepcopy(source)
            row["game_id"] = cloned_id
            decision_rows.append(row)

    exp1 = arena_metrics(outcomes, decision_rows, include_bootstrap=False)
    exp2 = long_chain_metrics(decision_rows, outcomes, include_bootstrap=False)
    exp4 = execution_metrics(decision_rows, outcomes, include_bootstrap=False)
    policies = {
        row["policy"]: {
            "policy_id": row["policy"],
            "model_configuration_id": row.get("model_configuration_id"),
            "glicko2_mu": (row.get("glicko2") or {}).get("mu"),
            "illegal_action_rate": None,
        }
        for row in exp1.get("leaderboard", [])
    }
    for row in exp2.get("kaplan_meier", []):
        if row.get("policy") in policies:
            policies[row["policy"]]["long_chain_survival_at_20"] = _fixed_survival(row)
    for row in exp4.get("by_agent", []):
        if row.get("policy") in policies:
            policies[row["policy"]]["illegal_action_rate"] = row.get("illegal_action_rate")
    by_model: dict[str, list[str]] = defaultdict(list)
    for policy, row in policies.items():
        by_model[str(row.get("model_configuration_id"))].append(policy)

    for _evaluator, source_rows in state_by_evaluator.items():
        sampled_rows = []
        for source in source_rows:
            for cloned_id in multiplicity.get(str(source.get("game_id")), []):
                sampled_rows.append({**source, "game_id": cloned_id})
        if not sampled_rows:
            continue
        identity = sampled_rows[0].get("evaluator_identity") or {}
        targets = by_model.get(str(identity.get("model_configuration_id")), [])
        if len(targets) == 1:
            policies[targets[0]]["state_jga"] = state_metrics(
                sampled_rows, include_bootstrap=False
            ).get("jga")

    for _evaluator, source_rows in forecast_by_evaluator.items():
        sampled_rows = []
        for source in source_rows:
            for cloned_id in multiplicity.get(str(source.get("game_id")), []):
                sampled_rows.append({**source, "game_id": cloned_id})
        if not sampled_rows:
            continue
        identity = sampled_rows[0].get("evaluator_identity") or {}
        targets = by_model.get(str(identity.get("model_configuration_id")), [])
        if len(targets) == 1:
            policies[targets[0]]["availability_brier"] = forecast_metrics(
                sampled_rows,
                "availability_ground_truth",
                "availability_probability",
                include_bootstrap=False,
            ).get("brier_score")

    exp6_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    horizons: list[int] = []
    for game, _ in selected_games:
        identities = game["policy_identities"]
        for result in game["exp6"].get("results", []):
            if result.get("status") != "COMPLETED":
                continue
            focal = int(result.get("candidate", {}).get("focal_player", 0))
            if focal >= len(identities):
                continue
            model = identities[focal].get("model_configuration_id")
            exp6_by_model[str(model)].append(result)
            value = result.get("action_sequence_survival", {}).get("horizon")
            if value is not None:
                horizons.append(int(value))
    horizon = min(horizons) if horizons else None
    for model, results in exp6_by_model.items():
        targets = by_model.get(model, [])
        if len(targets) != 1:
            continue
        changed = [result for result in results if result.get("state_changed")]
        analysis = kaplan_meier_action_survival(changed, horizon=horizon)
        policies[targets[0]]["plan_survival_at_horizon"] = _fixed_survival(analysis)
        policies[targets[0]]["plan_rmst"] = (analysis.get("rmst") or {}).get("estimate")
    return [policies[key] for key in sorted(policies)]


def _cluster_bootstrap_correlations(
    run_dir: Path,
    pairs: list[tuple[str, str]],
    *,
    seed: int,
    replicates: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    games, states, forecasts = _bootstrap_inputs(run_dir)
    clusters = sorted({game["cluster_id"] for game in games})
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    if len(clusters) < 2:
        return {
            pair: {
                "status": "INSUFFICIENT_CLUSTERS",
                "clusters": len(clusters),
                "replicates_requested": replicates,
                "replicates_valid": 0,
                "confidence_interval_95": None,
            }
            for pair in pairs
        }
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled = [rng.choice(clusters) for _ in clusters]
        matrix = _resampled_capability_matrix(games, states, forecasts, sampled)
        for pair in pairs:
            overlap = [
                row
                for row in matrix
                if row.get(pair[0]) is not None and row.get(pair[1]) is not None
            ]
            estimate = spearman(
                [float(row[pair[0]]) for row in overlap],
                [float(row[pair[1]]) for row in overlap],
            )
            if estimate is not None:
                values[pair].append(estimate)
    return {
        pair: {
            "status": "COMPLETED" if values[pair] else "INSUFFICIENT_VALID_REPLICATES",
            "clusters": len(clusters),
            "replicates_requested": replicates,
            "replicates_valid": len(values[pair]),
            "confidence_interval_95": (
                {
                    "lower": percentile(values[pair], 0.025),
                    "upper": percentile(values[pair], 0.975),
                }
                if values[pair]
                else None
            ),
        }
        for pair in pairs
    }


def compute_exp7_metrics(
    run_dir: Path, *, bootstrap_seed: int = 0, bootstrap_replicates: int = 1000
) -> dict[str, Any]:
    matrix, missing = build_capability_matrix(run_dir)
    child_metrics = [key for key in CAPABILITIES if key != "glicko2_mu"]
    versus_strength = [_correlation(matrix, key, "glicko2_mu") for key in child_metrics]
    pairwise = [_correlation(matrix, left, right) for left, right in combinations(CAPABILITIES, 2)]
    pairs = sorted({(item["left"], item["right"]) for item in [*versus_strength, *pairwise]})
    bootstrap = _cluster_bootstrap_correlations(
        run_dir,
        pairs,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    for item in [*versus_strength, *pairwise]:
        uncertainty = bootstrap[(item["left"], item["right"])]
        item["cluster_bootstrap"] = uncertainty
        item["confidence_interval_95"] = uncertainty["confidence_interval_95"]
    result = {
        "schema_version": "2.0.0",
        "experiment": 7,
        "status": "COMPLETED" if matrix else "INSUFFICIENT_DATA",
        "analysis_unit": "exact policy configuration",
        "capability_directions": CAPABILITIES,
        "auxiliary_metric_directions": AUXILIARY_CAPABILITIES,
        "model_capability_matrix": matrix,
        "capability_profiles": [
            {
                "policy_id": row["policy_id"],
                "agent": row.get("agent"),
                "primary_metrics": {key: row.get(key) for key in CAPABILITIES},
                "auxiliary_metrics": {
                    key: row.get(key) for key in AUXILIARY_CAPABILITIES if key in row
                },
                "missing_metrics": [key for key in CAPABILITIES if row.get(key) is None],
            }
            for row in matrix
        ],
        "correlation_with_glicko2": versus_strength,
        "pairwise_capability_correlations": pairwise,
        "missingness": missing,
        "mixed_effects_logistic_regression": {
            "status": "NOT_RUN_OPTIONAL_ANALYSIS",
            "reason": "requires a preregistered adequately powered multi-model duel sample",
        },
        "bootstrap_protocol": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "cluster": "seed x seat-swapped pair x deck matchup",
        },
        "provenance": metric_provenance(run_dir, statistics_seed=bootstrap_seed),
    }
    output = run_dir / "metrics" / "exp7"
    atomic_write_json(output / "metrics.json", result)
    _write_csv(output / "capability_matrix.csv", matrix)
    _write_csv(output / "correlation_matrix.csv", pairwise)
    atomic_write_json(output / "exclusions.json", missing)
    return result
