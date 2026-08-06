"""Run the pinned puzzle benchmark and write normalized metrics."""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ygobench.bench.metrics import load_and_summarize, save_metrics
from ygobench.config import PROJECT_ROOT, ModelConfig
from ygobench.engine.upstream import UpstreamLayout


@dataclass(frozen=True)
class EvalConfig:
    model: ModelConfig
    run_name: str | None = None
    attempts: int | None = None
    limit: int | None = None
    offset: int = 0
    only: tuple[str, ...] = ()
    concurrency: int = 1
    max_tool_calls: int = 500
    perspective: int = 0
    overwrite: bool = False
    forage: bool = False
    show_solution: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        mode = f"attempts-{self.attempts}" if self.attempts is not None else "interactive"
        model = self.model.model.replace("/", "_").replace(" ", "_")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}_{mode}_{self.model.provider}_{model}"


def build_command(
    config: EvalConfig,
    *,
    layout: UpstreamLayout | None = None,
    results_root: Path | None = None,
) -> tuple[list[str], Path]:
    layout = layout or UpstreamLayout()
    layout.require_runtime()
    results_root = results_root or PROJECT_ROOT / "bench_data" / "runs"
    run_name = config.resolved_run_name()

    command = [
        sys.executable,
        str(layout.runner),
        "--dataset",
        str(layout.dataset),
        "--provider",
        config.model.provider,
        "--model",
        config.model.model,
        "--run-name",
        run_name,
        "--results-root",
        str(results_root),
        "--perspective",
        str(config.perspective),
        "--concurrency",
        str(config.concurrency),
        "--max-tool-calls",
        str(config.max_tool_calls),
    ]
    if config.model.base_url:
        command.extend(["--base-url", config.model.base_url])
    if config.attempts is None:
        command.append("--interactive")
    else:
        command.extend(["--attempts", str(config.attempts)])
    if config.limit is not None:
        command.extend(["--limit", str(config.limit)])
    if config.offset:
        command.extend(["--offset", str(config.offset)])
    for puzzle_id in config.only:
        command.extend(["--only", puzzle_id])
    if config.overwrite:
        command.append("--overwrite")
    if config.forage:
        command.append("--forage")
    if config.show_solution:
        command.append("--show-solution")
    command.extend(config.extra_args)
    return command, results_root / run_name


def run_evaluation(config: EvalConfig, *, dry_run: bool = False) -> tuple[int, Path]:
    command, run_dir = build_command(config)
    print(shlex.join(command))
    if dry_run:
        return 0, run_dir

    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    summary_path = run_dir / "_summary.json"
    if summary_path.exists():
        metrics = load_and_summarize(summary_path, perspective=config.perspective)
        save_metrics(metrics, run_dir / "metrics.json")
    return completed.returncode, run_dir
