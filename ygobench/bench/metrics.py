"""Normalize upstream puzzle outcomes into stable benchmark metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkMetrics:
    total: int
    solved: int
    solve_rate: float
    engine_completed: int
    engine_completion_rate: float
    avg_tool_calls: float
    avg_model_calls: float
    avg_elapsed_seconds: float
    input_tokens: int
    output_tokens: int
    termination_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def summarize_upstream(summary: dict[str, Any], *, perspective: int = 0) -> BenchmarkMetrics:
    rows = list((summary.get("per_instance") or {}).values())
    total = len(rows)
    solved = sum(1 for row in rows if row.get("winner") == perspective)
    completed = sum(1 for row in rows if row.get("game_over") is True)

    termination_counts: dict[str, int] = {}
    tool_calls = model_calls = elapsed = 0.0
    input_tokens = output_tokens = 0
    for row in rows:
        termination = str(row.get("termination", "unknown"))
        termination_counts[termination] = termination_counts.get(termination, 0) + 1
        tool_calls += _number(row.get("tool_calls_used"))
        elapsed += _number(row.get("elapsed"))
        usage = row.get("model_usage_totals") or {}
        model_calls += _number(usage.get("model_calls"))
        input_tokens += int(_number(usage.get("input_tokens")))
        output_tokens += int(_number(usage.get("output_tokens")))

    divisor = total or 1
    return BenchmarkMetrics(
        total=total,
        solved=solved,
        solve_rate=solved / divisor if total else 0.0,
        engine_completed=completed,
        engine_completion_rate=completed / divisor if total else 0.0,
        avg_tool_calls=tool_calls / divisor if total else 0.0,
        avg_model_calls=model_calls / divisor if total else 0.0,
        avg_elapsed_seconds=elapsed / divisor if total else 0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        termination_counts=termination_counts,
    )


def load_and_summarize(path: Path, *, perspective: int = 0) -> BenchmarkMetrics:
    return summarize_upstream(json.loads(path.read_text()), perspective=perspective)


def save_metrics(metrics: BenchmarkMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n")

