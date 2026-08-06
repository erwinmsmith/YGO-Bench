import pytest

from ygobench.bench.metrics import summarize_upstream


def test_summarize_upstream() -> None:
    summary = {
        "per_instance": {
            "a": {
                "termination": "game_over",
                "game_over": True,
                "winner": 0,
                "tool_calls_used": 8,
                "elapsed": 10,
                "model_usage_totals": {
                    "model_calls": 4,
                    "input_tokens": 100,
                    "output_tokens": 20,
                },
            },
            "b": {
                "termination": "tool_budget_exhausted",
                "game_over": False,
                "winner": None,
                "tool_calls_used": 12,
                "elapsed": 20,
                "model_usage_totals": {
                    "model_calls": 6,
                    "input_tokens": 200,
                    "output_tokens": 30,
                },
            },
        }
    }
    metrics = summarize_upstream(summary)
    assert metrics.total == 2
    assert metrics.solved == 1
    assert metrics.solve_rate == pytest.approx(0.5)
    assert metrics.engine_completion_rate == pytest.approx(0.5)
    assert metrics.avg_tool_calls == pytest.approx(10)
    assert metrics.avg_model_calls == pytest.approx(5)
    assert metrics.avg_elapsed_seconds == pytest.approx(15)
    assert metrics.input_tokens == 300
    assert metrics.output_tokens == 50


def test_empty_summary() -> None:
    metrics = summarize_upstream({})
    assert metrics.total == 0
    assert metrics.solve_rate == 0

