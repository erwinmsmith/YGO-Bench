import pytest

import ygobench.experiments.replanning as replanning
from ygobench.experiments.identity import (
    model_configuration_id,
    policy_descriptor,
    policy_id,
)
from ygobench.experiments.internal_validity import compute_exp7_metrics
from ygobench.experiments.io import atomic_write_json
from ygobench.experiments.metrics import (
    _execution_metrics,
    _long_chain_metrics,
    _state_metrics,
)
from ygobench.experiments.probes import (
    _select_across_games,
    _select_forecast_candidates,
)
from ygobench.experiments.replanning import kaplan_meier_action_survival
from ygobench.experiments.statistics import (
    binary_probability_metrics,
    nearest_rank,
    spearman,
)


def _survival_result(survived: int, *, failed: bool, censor_reason: str | None = None) -> dict:
    return {
        "state_changed": True,
        "action_sequence_survival": {
            "horizon": 8,
            "survived_actions": survived,
            "first_failure": {"reason": "not_legal"} if failed else None,
            "event_time": survived + 1 if failed else None,
            "censor_time": survived if not failed else None,
            "censor_reason": censor_reason,
        },
    }


def test_exp6_km_example_and_no_legacy_averages() -> None:
    results = [
        _survival_result(0, failed=True),
        _survival_result(2, failed=True),
        _survival_result(5, failed=True),
        _survival_result(7, failed=False, censor_reason="trajectory_exhausted"),
        _survival_result(8, failed=False, censor_reason="horizon_reached"),
    ]
    analysis = kaplan_meier_action_survival(results, horizon=8)
    assert analysis["fixed_window"]["survival_probability"]["estimate"] == pytest.approx(0.4)
    assert analysis["fixed_window"]["cumulative_failure_probability"]["estimate"] == pytest.approx(
        0.6
    )
    assert analysis["km_median"] == {"estimate": 6, "status": "ESTIMATED"}
    assert analysis["rmst"]["estimate"] == pytest.approx(5.2)
    assert "mean_steps" not in analysis
    assert "median_steps" not in analysis
    assert "horizon_plan_invalidation_rate" not in analysis


def test_exp6_legacy_trajectory_end_is_right_censored() -> None:
    legacy = {
        "state_changed": True,
        "action_sequence_survival": {
            "horizon": 8,
            "survived_actions": 4,
            "first_failure": None,
            "right_censored": False,
        },
    }
    analysis = kaplan_meier_action_survival([legacy], horizon=4)
    assert analysis["events"] == 0
    assert analysis["right_censored"] == 1
    assert analysis["censor_reasons"] == {"trajectory_exhausted": 1}


def test_policy_identity_separates_prompt_but_model_configuration_does_not() -> None:
    left = policy_descriptor("react-fast:dashscope:qwen3.7-flash", prompt_hashes={"system": "a"})
    right = policy_descriptor("react-fast:dashscope:qwen3.7-flash", prompt_hashes={"system": "b"})
    assert policy_id(left) != policy_id(right)
    assert model_configuration_id(left) == model_configuration_id(right)

    hotter = policy_descriptor(
        "react-fast:dashscope:qwen3.7-flash",
        prompt_hashes={"system": "a"},
        runtime={"temperature": 0.7, "max_tokens": None, "thinking_enabled": False},
    )
    assert model_configuration_id(left) != model_configuration_id(hotter)


def test_nearest_rank_and_spearman_ties() -> None:
    assert nearest_rank([4, 7, 21, 6, 8], 0.75) == 8
    assert spearman([1, 1, 3], [1, 2, 3]) == pytest.approx(0.8660254)


def test_exp2_has_no_cox_output() -> None:
    outcome = {
        "game_id": "g1",
        "agents": ["alpha", "beta"],
        "decks": ["A", "B"],
        "competitive_eligible": True,
        "termination": "game_over",
        "winner": 0,
    }
    row = {
        "game_id": "g1",
        "turn": 1,
        "player": 0,
        "decision_index": 1,
        "legal": {"requires_model_reasoning": True, "legal_action_count": 2},
        "observation": {"chain": []},
        "validation": {"valid": True},
    }
    metrics = _long_chain_metrics([row], [outcome])
    assert "cox_proportional_hazards" not in metrics
    assert metrics["turn_decision_length_quartile_cutpoints"]["method"] == "nearest_rank"


def test_exp3_schema_failure_stays_in_primary_denominator() -> None:
    rows = [
        {"game_id": "g1", "ground_truth": {"turn": 1}, "prediction": {"turn": 1}},
        {"game_id": "g2", "ground_truth": {"turn": 1}, "prediction": None},
    ]
    metrics = _state_metrics(rows)
    assert metrics["valid_samples"] == 1
    assert metrics["jga"] == pytest.approx(0.5)
    assert metrics["valid_only_jga"] == pytest.approx(1.0)
    assert metrics["slot_accuracy"] == pytest.approx(0.5)
    assert metrics["valid_only_slot_accuracy"] == pytest.approx(1.0)


def test_exp3_macro_f1_averages_field_level_f1() -> None:
    rows = [
        {
            "game_id": "g1",
            "ground_truth": {"cards": ["A", "B"], "events": ["X"]},
            "prediction": {"cards": ["A"], "events": ["X"]},
        },
        {
            "game_id": "g2",
            "ground_truth": {"cards": ["C"], "events": ["Y"]},
            "prediction": {"cards": ["C"], "events": []},
        },
    ]
    metrics = _state_metrics(rows)
    # cards F1=4/5 and events F1=2/3; macro is their unweighted mean.
    assert metrics["slot_f1"]["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)


def test_exp4_normalizes_legacy_error_code_and_keeps_raw_evidence() -> None:
    outcome = {
        "game_id": "g1",
        "agents": ["alpha", "beta"],
        "decks": ["A", "B"],
        "termination": "game_over",
    }
    row = {
        "game_id": "g1",
        "decision_id": "d1",
        "decision_index": 1,
        "player": 0,
        "legal": {"requires_model_reasoning": True, "expected_responder": "select_place"},
        "attempted_action": {"tool": "select_place", "arguments": {"place": 9}},
        "trace": {"turns": [], "fallback": False},
        "validation": {
            "valid": False,
            "primary_error_code": "exact_legal_rejection",
            "pre_engine_error": "action_not_in_exact_legal_set",
            "model_action_attempts": 2,
        },
    }
    metrics = _execution_metrics([row], [outcome], include_bootstrap=False)
    taxonomy = metrics["primary_error_taxonomy"][0]
    assert taxonomy["counts"] == {"invalid_position_or_place": 1}
    assert metrics["error_evidence"][0]["raw_error_evidence"] == {
        "pre_engine_error": "action_not_in_exact_legal_set"
    }


def test_exp7_writes_matrix_without_creating_a_composite_score(tmp_path) -> None:
    run_dir = tmp_path / "run"
    atomic_write_json(
        run_dir / "metrics" / "exp1" / "metrics.json",
        {
            "leaderboard": [
                {
                    "policy": "p1",
                    "model_configuration_id": "m1",
                    "agent": "agent-1",
                    "strict_games": 2,
                    "overall_win_rate": 0.5,
                    "glicko2": {"mu": 1500.0, "rd_phi": 200.0},
                }
            ]
        },
    )
    atomic_write_json(run_dir / "metrics" / "exp2" / "metrics.json", {"kaplan_meier": []})
    atomic_write_json(run_dir / "metrics" / "exp4" / "metrics.json", {"by_agent": []})
    atomic_write_json(
        run_dir / "metrics" / "exp6" / "metrics.json",
        {"by_model_configuration": []},
    )

    result = compute_exp7_metrics(run_dir, bootstrap_replicates=5)

    assert "composite_score" not in result
    assert result["model_capability_matrix"][0]["policy_id"] == "p1"
    assert result["correlation_with_glicko2"][0]["status"] == "INSUFFICIENT_DATA"
    assert (run_dir / "metrics" / "exp7" / "capability_matrix.csv").is_file()


def test_probability_metrics_support_inverse_probability_weights() -> None:
    rows = [
        {"truth": 1, "probability": 0.9, "weight": 4.0},
        {"truth": 0, "probability": 0.1, "weight": 1.0},
    ]
    metrics = binary_probability_metrics(
        rows, truth_key="truth", probability_key="probability", weight_key="weight"
    )
    assert metrics["prevalence"] == pytest.approx(0.8)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.01)


def test_exp5_joint_stratification_covers_availability_and_behavior() -> None:
    candidates = [
        {
            "sample_id": f"s{index}",
            "trajectory_index": index,
            "availability_ground_truth": availability,
            "behavior_ground_truth": behavior,
        }
        for index, (availability, behavior) in enumerate(
            [(0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 1)]
        )
    ]
    selected = _select_forecast_candidates(
        candidates, max_samples=3, sampling="stratified", seed=11
    )
    assert {
        (row["availability_ground_truth"], row["behavior_ground_truth"]) for row in selected
    } == {(0, 0), (1, 0), (1, 1)}
    assert selected == _select_forecast_candidates(
        candidates, max_samples=3, sampling="stratified", seed=11
    )


def test_probe_limit_is_distributed_across_games() -> None:
    samples = [
        {"game_id": game, "sample_id": f"{game}-{index}"}
        for game in ("g1", "g2", "g3")
        for index in range(3)
    ]
    selected = _select_across_games(samples, 3)
    assert {row["game_id"] for row in selected} == {"g1", "g2", "g3"}


def test_exp6_candidate_journal_resumes_after_crash(tmp_path, monkeypatch) -> None:
    game_dir = tmp_path / "game"
    atomic_write_json(
        game_dir / "manifest.json",
        {
            "config": {
                "game_id": "g1",
                "deck1": "A",
                "deck2": "B",
                "seed": 7,
            }
        },
    )
    candidates = [
        {"candidate_id": "c1", "response_index": 1},
        {"candidate_id": "c2", "response_index": 2},
    ]
    monkeypatch.setattr(
        replanning,
        "find_offline_counterfactual_candidates",
        lambda _game_dir: {
            "commitment_points": 2,
            "eligible_counterfactuals": 2,
            "candidates": candidates,
        },
    )
    calls: list[str] = []

    def crash_on_second(_config, _rows, candidate, _horizon):
        calls.append(candidate["candidate_id"])
        if candidate["candidate_id"] == "c2":
            raise RuntimeError("injected crash")
        return {"status": "COMPLETED", "candidate": candidate}

    monkeypatch.setattr(replanning, "_audit_candidate", crash_on_second)
    with pytest.raises(RuntimeError, match="injected crash"):
        replanning.run_offline_counterfactual_audit(game_dir, sample_size=2, horizon=8)
    assert calls == ["c1", "c2"]

    resumed_calls: list[str] = []

    def complete(_config, _rows, candidate, _horizon):
        resumed_calls.append(candidate["candidate_id"])
        return {"status": "COMPLETED", "candidate": candidate}

    monkeypatch.setattr(replanning, "_audit_candidate", complete)
    report = replanning.run_offline_counterfactual_audit(game_dir, sample_size=2, horizon=8)
    assert resumed_calls == ["c2"]
    assert report["resume"] == {
        "journal": "exp6_offline_results.jsonl",
        "candidates_reused": 1,
        "candidates_computed": 1,
    }
