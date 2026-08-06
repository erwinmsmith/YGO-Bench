"""Command-line interface for setup, evaluation and reporting."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ygobench.bench.eval_pipeline import EvalConfig, run_evaluation
from ygobench.bench.metrics import load_and_summarize, save_metrics
from ygobench.config import PROJECT_ROOT, default_model_config, missing_api_key
from ygobench.engine.upstream import UpstreamLayout


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "vllm", "deepseek", "claude-cli"],
        default=None,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ygo-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Build pinned ocgcore resources and verify offline")
    sub.add_parser("doctor", help="Check source, build tools, engine, and dataset")

    duel = sub.add_parser("duel", help="Run a complete headless duel between two agents")
    duel.add_argument("--deck1", default="BlueEyes")
    duel.add_argument("--deck2", default="BlueEyes")
    duel.add_argument("--agent1", default="passive")
    duel.add_argument("--agent2", default="passive")
    duel.add_argument("--seed", type=int, default=0)
    duel.add_argument(
        "--max-decisions",
        type=int,
        default=2000,
        help="Decision cap; use 0 to run until the engine reaches a terminal result",
    )

    arena = sub.add_parser("arena", help="Run a seat-swapped full-duel tournament")
    arena.add_argument("--agents", nargs="+", default=["passive", "random"])
    arena.add_argument("--decks", nargs="+", default=["BlueEyes", "CyberDragon"])
    arena.add_argument("--seeds", nargs="+", type=int, default=[0])
    arena.add_argument("--max-decisions", type=int, default=2000)
    arena.add_argument("--run-name")

    evaluate = sub.add_parser("eval", help="Run puzzle benchmark evaluation")
    _add_model_args(evaluate)
    evaluate.add_argument("--run-name")
    evaluate.add_argument("--attempts", type=int, default=None)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--offset", type=int, default=0)
    evaluate.add_argument("--only", action="append", default=[])
    evaluate.add_argument("--concurrency", type=int, default=1)
    evaluate.add_argument("--max-tool-calls", type=int, default=500)
    evaluate.add_argument("--perspective", type=int, choices=[0, 1], default=0)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.add_argument("--forage", action="store_true")
    evaluate.add_argument("--show-solution", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")

    report = sub.add_parser("report", help="Normalize an upstream _summary.json")
    report.add_argument("summary", type=Path)
    report.add_argument("--perspective", type=int, choices=[0, 1], default=0)
    report.add_argument("--output", type=Path)
    return parser


def _setup(layout: UpstreamLayout) -> int:
    errors = layout.source_errors()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print("Run: git submodule update --init --recursive", file=sys.stderr)
        return 2
    local_premake = PROJECT_ROOT / ".tools" / "premake5"
    if shutil.which("premake5") is None and not local_premake.is_file():
        bootstrap = PROJECT_ROOT / "scripts" / "bootstrap_premake.py"
        bootstrap_result = subprocess.run([sys.executable, str(bootstrap)], check=False)
        if bootstrap_result.returncode != 0:
            return bootstrap_result.returncode
    env = os.environ.copy()
    env["PATH"] = f"{local_premake.parent}{os.pathsep}{env.get('PATH', '')}"
    result = subprocess.run(
        ["bash", str(layout.setup_script)], cwd=layout.root, env=env, check=False
    )
    return result.returncode


def _doctor(layout: UpstreamLayout) -> int:
    local_premake = PROJECT_ROOT / ".tools" / "premake5"
    checks = {
        "git": shutil.which("git") is not None,
        "c++": shutil.which("g++") is not None or shutil.which("clang++") is not None,
        "make": shutil.which("make") is not None,
        "premake5": shutil.which("premake5") is not None or local_premake.is_file(),
        "submodule": not layout.source_errors(),
        "ocgcore": layout.engine_library.is_file(),
        "dataset": layout.dataset.is_file(),
    }
    for name, ok in checks.items():
        print(f"{'OK' if ok else 'MISSING':<7} {name}")
    if not all(checks.values()):
        print("\nRun `ygo-bench setup` after installing missing build tools.")
        return 1
    return 0


def _eval(args: argparse.Namespace) -> int:
    model = default_model_config(args.provider, args.model)
    if args.base_url:
        model = type(model)(provider=model.provider, model=model.model, base_url=args.base_url)
    missing = missing_api_key(model.provider)
    if missing and not args.dry_run:
        env_path = PROJECT_ROOT / ".env"
        print(f"Missing {missing}. Add it to {env_path} or the shell.", file=sys.stderr)
        return 2
    if args.attempts is not None and args.attempts < 1:
        print("--attempts must be at least 1", file=sys.stderr)
        return 2
    config = EvalConfig(
        model=model,
        run_name=args.run_name,
        attempts=args.attempts,
        limit=args.limit,
        offset=args.offset,
        only=tuple(args.only),
        concurrency=args.concurrency,
        max_tool_calls=args.max_tool_calls,
        perspective=args.perspective,
        overwrite=args.overwrite,
        forage=args.forage,
        show_solution=args.show_solution,
    )
    try:
        return_code, run_dir = run_evaluation(config, dry_run=args.dry_run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Run artifacts: {run_dir}")
    return return_code


def _report(args: argparse.Namespace) -> int:
    metrics = load_and_summarize(args.summary, perspective=args.perspective)
    output = args.output or args.summary.with_name("metrics.json")
    save_metrics(metrics, output)
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    print(f"Wrote {output}")
    return 0


def _duel(args: argparse.Namespace) -> int:
    from ygobench.agents.factory import create_agent
    from ygobench.engine.full_duel import run_duel

    deck_root = PROJECT_ROOT / "resources" / "decks"
    deck1 = deck_root / f"{args.deck1}.ydk"
    deck2 = deck_root / f"{args.deck2}.ydk"
    missing = [str(path) for path in (deck1, deck2) if not path.is_file()]
    if missing:
        print(f"Missing deck file(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        result = run_duel(
            deck1,
            deck2,
            agent1=create_agent(args.agent1, seed=args.seed * 2),
            agent2=create_agent(args.agent2, seed=args.seed * 2 + 1),
            seed=args.seed,
            max_decisions=args.max_decisions,
        )
    except Exception as exc:
        print(f"Full duel failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.game_over else 1


def _arena(args: argparse.Namespace) -> int:
    from ygobench.bench.arena import ArenaConfig, run_arena

    try:
        run_dir = run_arena(
            ArenaConfig(
                agents=tuple(args.agents),
                decks=tuple(args.decks),
                seeds=tuple(args.seeds),
                max_decisions=args.max_decisions,
                run_name=args.run_name,
            )
        )
    except Exception as exc:
        print(f"Arena failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Arena artifacts: {run_dir}")
    print((run_dir / "metrics.json").read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    layout = UpstreamLayout()
    if args.command == "setup":
        return _setup(layout)
    if args.command == "doctor":
        return _doctor(layout)
    if args.command == "duel":
        return _duel(args)
    if args.command == "arena":
        return _arena(args)
    if args.command == "eval":
        return _eval(args)
    if args.command == "report":
        return _report(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
