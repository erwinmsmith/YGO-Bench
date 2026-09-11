"""Command line interface for the staged experiment pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ygobench.config import PROJECT_ROOT
from ygobench.experiments.capabilities import inspect_capabilities
from ygobench.experiments.config import ExperimentConfig
from ygobench.experiments.internal_validity import compute_exp7_metrics
from ygobench.experiments.io import atomic_write_json
from ygobench.experiments.metrics import compute_phase2_metrics, compute_probe_metrics
from ygobench.experiments.probes import extract_probe_samples, run_probes
from ygobench.experiments.replanning import (
    aggregate_offline_exp6_metrics,
    run_offline_counterfactual_audit,
    write_offline_exp6_metrics,
)
from ygobench.experiments.replay import verify_reversible_decisions
from ygobench.experiments.runner import run_evidence_duel


def _root() -> Path:
    return PROJECT_ROOT / "bench_data" / "experiments"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ygo-experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    p0 = sub.add_parser("phase0")
    p0.add_argument("--run-id", default="phase0")
    duel = sub.add_parser("duel")
    duel.add_argument("--run-id", required=True)
    duel.add_argument("--deck1", default="BlueEyes")
    duel.add_argument("--deck2", default="BlueEyes")
    duel.add_argument("--agent1", default="react-fast:deepseek:deepseek-v4-flash")
    duel.add_argument("--agent2", default="react-fast:deepseek:deepseek-v4-flash")
    duel.add_argument("--seed", type=int, default=1)
    duel.add_argument("--max-decisions", type=int, default=12)
    metrics = sub.add_parser("phase2")
    metrics.add_argument("--run-id", required=True)
    metrics.add_argument(
        "--experiments",
        nargs="+",
        choices=("exp1", "exp2", "exp4"),
        default=("exp1", "exp2", "exp4"),
        help="Phase 2 experiments to compute (default: all).",
    )
    probes = sub.add_parser("phase3")
    probes.add_argument("--run-id", required=True)
    probes.add_argument("--extract-only", action="store_true")
    probes.add_argument("--max-state-samples", type=int, default=4)
    probes.add_argument("--max-forecast-samples", type=int, default=4)
    probes.add_argument(
        "--experiments",
        nargs="+",
        choices=("exp3", "exp5"),
        default=("exp3", "exp5"),
        help="Post-hoc experiments to run (default: both).",
    )
    probes.add_argument(
        "--provider",
        default="deepseek",
        help="Provider used for Exp3/Exp5 post-hoc probes.",
    )
    probes.add_argument(
        "--model",
        default=None,
        help="Model used for Exp3/Exp5 post-hoc probes.",
    )
    probes.add_argument(
        "--forecast-sampling",
        choices=("chronological", "stratified"),
        default="stratified",
        help=("Deterministic Exp5 sampling policy (default: joint availability/behavior strata)."),
    )
    verify = sub.add_parser("phase4")
    verify.add_argument("--run-id", required=True)
    verify_group = verify.add_mutually_exclusive_group(required=True)
    verify_group.add_argument("--game-id")
    verify_group.add_argument("--all-games", action="store_true")
    verify.add_argument("--offline-sample-size", type=int, default=10)
    verify.add_argument("--offline-horizon", type=int, default=32)
    internal = sub.add_parser("phase5")
    internal.add_argument("--run-id", required=True)
    internal.add_argument("--bootstrap-seed", type=int, default=0)
    internal.add_argument("--bootstrap-replicates", type=int, default=1000)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument(
        "--experiments",
        nargs="+",
        choices=("exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"),
        default=("exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7"),
    )
    evaluate.add_argument("--provider", default="deepseek")
    evaluate.add_argument("--model", default=None)
    evaluate.add_argument("--max-state-samples", type=int, default=4)
    evaluate.add_argument("--max-forecast-samples", type=int, default=4)
    evaluate.add_argument("--offline-sample-size", type=int, default=10)
    evaluate.add_argument("--offline-horizon", type=int, default=32)
    evaluate.add_argument("--bootstrap-seed", type=int, default=0)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser


def _run_phase4_all(run_dir: Path, *, sample_size: int, horizon: int) -> dict[str, Any]:
    game_dirs = sorted(path.parent for path in (run_dir / "games").glob("*/manifest.json"))
    game_results = []
    for game_dir in game_dirs:
        reversible = verify_reversible_decisions(
            game_dir, output=game_dir / "reversibility_report.json"
        )
        audit = None
        if reversible["passed"]:
            audit = run_offline_counterfactual_audit(
                game_dir, sample_size=sample_size, horizon=horizon
            )
        game_results.append(
            {
                "game_id": game_dir.name,
                "reversibility": reversible,
                "offline_audit": audit,
                "metrics": write_offline_exp6_metrics(
                    run_dir, game_dir, reversibility=reversible, audit=audit
                ),
            }
        )
    return {"games": game_results, "metrics": aggregate_offline_exp6_metrics(run_dir)}


def main() -> int:
    args = build_parser().parse_args()
    root = _root()
    if args.command == "phase0":
        output = root / args.run_id / "phase0_capabilities.json"
        result = inspect_capabilities(output)
    elif args.command == "duel":
        config = ExperimentConfig.build(
            run_id=args.run_id,
            deck1=args.deck1,
            deck2=args.deck2,
            agent1=args.agent1,
            agent2=args.agent2,
            seed=args.seed,
            max_decisions=args.max_decisions,
        )
        result = run_evidence_duel(config, root=root)
    elif args.command == "phase2":
        result = compute_phase2_metrics(
            root / args.run_id,
            experiments=tuple(args.experiments),
        )
    elif args.command == "phase3":
        run_dir = root / args.run_id
        extracted = extract_probe_samples(
            run_dir,
            forecast_sampling=args.forecast_sampling,
        )
        result = {"extracted": extracted}
        if not args.extract_only:
            result["probes"] = run_probes(
                run_dir,
                provider_name=args.provider,
                model=args.model,
                max_state_samples=args.max_state_samples,
                max_forecast_samples=args.max_forecast_samples,
                experiments=tuple(args.experiments),
            )
            result["metrics"] = compute_probe_metrics(
                run_dir,
                provider_name=args.provider,
                model=args.model,
                experiments=tuple(args.experiments),
            )
    elif args.command == "phase4":
        run_dir = root / args.run_id
        game_dirs = (
            sorted(path.parent for path in (run_dir / "games").glob("*/manifest.json"))
            if args.all_games
            else [run_dir / "games" / args.game_id]
        )
        game_results = []
        for game_dir in game_dirs:
            reversible = verify_reversible_decisions(
                game_dir, output=game_dir / "reversibility_report.json"
            )
            audit = None
            if reversible["passed"]:
                audit = run_offline_counterfactual_audit(
                    game_dir,
                    sample_size=args.offline_sample_size,
                    horizon=args.offline_horizon,
                )
            game_results.append(
                {
                    "game_id": game_dir.name,
                    "reversibility": reversible,
                    "offline_audit": audit,
                    "metrics": write_offline_exp6_metrics(
                        run_dir,
                        game_dir,
                        reversibility=reversible,
                        audit=audit,
                    ),
                }
            )
        result = (
            {
                "games": game_results,
                "metrics": aggregate_offline_exp6_metrics(run_dir),
            }
            if args.all_games
            else game_results[0]
        )
    elif args.command == "phase5":
        result = compute_exp7_metrics(
            root / args.run_id,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    elif args.command == "evaluate":
        run_dir = root / args.run_id
        requested = tuple(dict.fromkeys(args.experiments))
        result = {"run_id": args.run_id, "requested": list(requested), "completed": {}}
        passive = tuple(name for name in requested if name in {"exp1", "exp2", "exp4"})
        if passive:
            result["completed"].update(compute_phase2_metrics(run_dir, experiments=passive))
            atomic_write_json(run_dir / "evaluation_status.json", result)
        probes_requested = tuple(name for name in requested if name in {"exp3", "exp5"})
        if probes_requested:
            result["probe_extraction"] = extract_probe_samples(run_dir)
            result["probe_execution"] = run_probes(
                run_dir,
                provider_name=args.provider,
                model=args.model,
                max_state_samples=args.max_state_samples,
                max_forecast_samples=args.max_forecast_samples,
                experiments=probes_requested,
            )
            result["completed"].update(
                compute_probe_metrics(
                    run_dir,
                    provider_name=args.provider,
                    model=args.model,
                    experiments=probes_requested,
                )
            )
            atomic_write_json(run_dir / "evaluation_status.json", result)
        if "exp6" in requested:
            result["completed"]["exp6"] = _run_phase4_all(
                run_dir,
                sample_size=args.offline_sample_size,
                horizon=args.offline_horizon,
            )
            atomic_write_json(run_dir / "evaluation_status.json", result)
        if "exp7" in requested:
            result["completed"]["exp7"] = compute_exp7_metrics(
                run_dir,
                bootstrap_seed=args.bootstrap_seed,
                bootstrap_replicates=args.bootstrap_replicates,
            )
        result["status"] = "COMPLETED"
        atomic_write_json(run_dir / "evaluation_status.json", result)
    else:  # pragma: no cover - argparse enforces the command set
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
