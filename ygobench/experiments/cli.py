"""Command line interface for the staged experiment pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ygobench.config import PROJECT_ROOT
from ygobench.experiments.capabilities import inspect_capabilities
from ygobench.experiments.config import ExperimentConfig
from ygobench.experiments.metrics import compute_phase2_metrics, compute_probe_metrics
from ygobench.experiments.probes import extract_probe_samples, run_probes
from ygobench.experiments.replanning import run_interruption_branch, write_exp6_metrics
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
    probes = sub.add_parser("phase3")
    probes.add_argument("--run-id", required=True)
    probes.add_argument("--extract-only", action="store_true")
    probes.add_argument("--max-state-samples", type=int, default=4)
    probes.add_argument("--max-forecast-samples", type=int, default=4)
    verify = sub.add_parser("phase4")
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--game-id", required=True)
    verify.add_argument("--skip-branch", action="store_true")
    return parser


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
        result = compute_phase2_metrics(root / args.run_id)
    elif args.command == "phase3":
        run_dir = root / args.run_id
        extracted = extract_probe_samples(run_dir)
        result = {"extracted": extracted}
        if not args.extract_only:
            result["probes"] = run_probes(
                run_dir,
                max_state_samples=args.max_state_samples,
                max_forecast_samples=args.max_forecast_samples,
            )
            result["metrics"] = compute_probe_metrics(run_dir)
    else:
        game_dir = root / args.run_id / "games" / args.game_id
        reversible = verify_reversible_decisions(
            game_dir, output=game_dir / "reversibility_report.json"
        )
        result = {"reversibility": reversible}
        if reversible["passed"] and not args.skip_branch:
            result["branch"] = run_interruption_branch(game_dir, root=root)
        result["metrics"] = write_exp6_metrics(
            root / args.run_id,
            game_dir,
            reversibility=reversible,
            branch=result.get("branch"),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
