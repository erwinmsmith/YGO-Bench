"""Public metric entry points for full-duel Experiments 1 through 5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ygobench.experiments.calculations import (
    arena_metrics as _arena_metrics,
)
from ygobench.experiments.calculations import (
    execution_metrics as _execution_metrics,
)
from ygobench.experiments.calculations import (
    forecast_metrics as _forecast_metrics,
)
from ygobench.experiments.calculations import (
    forecast_strata as _forecast_strata,
)
from ygobench.experiments.calculations import (
    long_chain_metrics as _long_chain_metrics,
)
from ygobench.experiments.calculations import (
    state_metrics as _state_metrics,
)
from ygobench.experiments.io import JsonlJournal, atomic_write_json, read_json
from ygobench.experiments.probes import probe_evaluator_id
from ygobench.experiments.provenance import metric_provenance
from ygobench.experiments.statistics import rate


def _game_dirs(run_dir: Path) -> list[Path]:
    return sorted(path.parent for path in (run_dir / "games").glob("*/manifest.json"))


def _outcomes_and_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outcomes, rows = [], []
    for game_dir in _game_dirs(run_dir):
        outcome = read_json(game_dir / "outcome.json")
        if outcome:
            manifest = read_json(game_dir / "manifest.json", {})
            if not outcome.get("policy_identities") and manifest.get("policy_identities"):
                outcome["policy_identities"] = manifest["policy_identities"]
            outcomes.append(outcome)
        rows.extend(JsonlJournal(game_dir / "trajectory.jsonl").recover())
    return outcomes, rows


def compute_phase2_metrics(
    run_dir: Path,
    *,
    experiments: tuple[str, ...] = ("exp1", "exp2", "exp4"),
) -> dict[str, Any]:
    """Compute passive metrics without touching unrequested experiments."""
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
    provenance = metric_provenance(run_dir)
    for name, value in combined.items():
        value["provenance"] = provenance
        atomic_write_json(run_dir / "metrics" / name / "metrics.json", value)
    if set(experiments) == set(available):
        atomic_write_json(run_dir / "metrics" / "phase2_metrics.json", combined)
    return combined


def _update_probe_metric_aggregate(
    path: Path, *, experiment: int, evaluator_id: str, metrics: dict[str, Any]
) -> dict[str, Any]:
    aggregate = read_json(path) or {
        "schema_version": "2.0.0",
        "experiment": experiment,
        "evaluators": {},
    }
    if "evaluators" not in aggregate:
        aggregate = {
            "schema_version": "2.0.0",
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
    experiments: tuple[str, ...] = ("exp3", "exp5"),
) -> dict[str, Any]:
    """Compute strict Exp3/Exp5 metrics for one post-hoc evaluator."""
    unknown = set(experiments) - {"exp3", "exp5"}
    if unknown:
        raise ValueError(f"Unknown probe experiments: {sorted(unknown)}")
    evaluator_id = probe_evaluator_id(provider_name, model)
    result_dir = run_dir / "derived" / "probe_results" / evaluator_id
    states = JsonlJournal(result_dir / "state_probe_results.jsonl").recover()
    forecasts = JsonlJournal(result_dir / "forecast_results.jsonl").recover()
    behavior = [row for row in forecasts if row.get("availability_ground_truth") == 1]
    impossible = [
        row
        for row in forecasts
        if row.get("availability_ground_truth") == 0 and row.get("behavior_ground_truth") == 1
    ]
    pool_counts: dict[str, int] = {}
    seen_games: set[str] = set()
    for row in forecasts:
        game_id = str(row.get("game_id", "unknown"))
        if game_id in seen_games:
            continue
        seen_games.add(game_id)
        for key, value in (row.get("candidate_joint_stratum_counts") or {}).items():
            pool_counts[key] = pool_counts.get(key, 0) + int(value)
    pool_total = sum(pool_counts.values())
    pool_availability_positive = pool_counts.get("A1B0", 0) + pool_counts.get("A1B1", 0)
    pool_behavior_positive = pool_counts.get("A1B1", 0)
    evaluator_identity = next(
        (
            row.get("evaluator_identity")
            for row in [*states, *forecasts]
            if isinstance(row.get("evaluator_identity"), dict)
        ),
        None,
    )
    exp3 = _state_metrics(states)
    exp5 = {
        "schema_version": "2.0.0",
        "experiment": 5,
        "status": (
            "DATA_QUALITY_ERROR"
            if impossible
            else "COMPLETED"
            if forecasts
            else "INSUFFICIENT_DATA"
        ),
        "primary_targets": ["availability", "behavior_given_availability"],
        "availability": _forecast_metrics(
            forecasts, "availability_ground_truth", "availability_probability"
        ),
        "behavior_given_availability": _forecast_metrics(
            behavior, "behavior_ground_truth", "behavior_probability"
        ),
        "behavior_unconditional_diagnostic": _forecast_metrics(
            forecasts, "behavior_ground_truth", "behavior_probability"
        ),
        "impossible_a0_b1_samples": [row.get("sample_id") for row in impossible],
        "candidate_pool": {
            "joint_stratum_counts": dict(sorted(pool_counts.items())),
            "total": pool_total,
            "availability_prevalence": rate(pool_availability_positive, pool_total),
            "behavior_prevalence_given_availability": rate(
                pool_behavior_positive, pool_availability_positive
            ),
        },
        "stratified": {
            "availability": _forecast_strata(
                forecasts, "availability_ground_truth", "availability_probability"
            ),
            "behavior_given_availability": _forecast_strata(
                behavior, "behavior_ground_truth", "behavior_probability"
            ),
        },
    }
    provenance = metric_provenance(run_dir)
    for value in (exp3, exp5):
        value["evaluator_id"] = evaluator_id
        value["evaluator_identity"] = evaluator_identity
        value["provenance"] = provenance
    requested = {3: exp3, 5: exp5}
    for experiment, value in requested.items():
        if f"exp{experiment}" not in experiments:
            continue
        atomic_write_json(run_dir / "metrics" / f"exp{experiment}" / f"{evaluator_id}.json", value)
        _update_probe_metric_aggregate(
            run_dir / "metrics" / f"exp{experiment}" / "metrics.json",
            experiment=experiment,
            evaluator_id=evaluator_id,
            metrics=value,
        )
    return {
        "evaluator_id": evaluator_id,
        **{name: requested[int(name[3:])] for name in experiments},
    }
