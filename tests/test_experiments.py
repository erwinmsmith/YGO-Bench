import json
import sys
from types import SimpleNamespace

from ygobench.agents.llm_agent import _tool_protocol_diagnostics, exact_legal_actions_packet
from ygobench.agents.provider_limits import (
    force_deepseek_thinking_mode,
    force_provider_thinking_disabled,
    force_single_tool_call,
    omit_deepseek_token_limit,
    omit_reasoning_model_token_limit,
    scrub_reasoning_content,
)
from ygobench.engine.protocol import ActionChoice
from ygobench.experiments import probes
from ygobench.experiments.capabilities import inspect_capabilities
from ygobench.experiments.config import ExperimentConfig, stable_id
from ygobench.experiments.io import JsonlJournal
from ygobench.experiments.legal import (
    _combination_count,
    exact_legal_action_match,
)
from ygobench.experiments.metrics import _arena_metrics, _execution_metrics, _state_metrics
from ygobench.experiments.probes import _response_window_after, _state_truth
from ygobench.experiments.registry import TaskRegistry
from ygobench.experiments.replay import verify_reversible_decisions
from ygobench.experiments.runner import (
    _decision_token_usage_breakdown,
    _record_totals,
    _trace_diagnostics,
    run_evidence_duel,
)


def test_stable_id_is_deterministic() -> None:
    assert stable_id("game", {"seed": 1}) == stable_id("game", {"seed": 1})
    assert stable_id("game", {"seed": 1}) != stable_id("game", {"seed": 2})


def test_exp1_handles_runs_with_no_strict_eligible_games() -> None:
    metrics = _arena_metrics(
        [
            {
                "game_id": "g1",
                "agents": ["alpha", "beta"],
                "decks": ["Labrynth", "Labrynth"],
                "winner": 0,
                "game_over": True,
                "competitive_eligible": False,
                "termination": "game_over",
                "decisions_by_player": [1, 1],
                "model_usage_totals": {},
                "decision_seconds": [0.0, 0.0],
            }
        ],
        [],
    )
    assert metrics["strict_games"] == 0
    assert all(row["strict_wins"] == 0 for row in metrics["leaderboard"])


def test_exp3_truth_includes_public_resources_and_history() -> None:
    observation = {
        "perspective_player": 0,
        "phase": "main1",
        "you": {"banished_count": 2, "extra_deck_count": 12, "monster_zone": [{"name": "Blue-Eyes"}]},
        "opponent": {"banished_count": 1, "extra_deck_count": 14, "graveyard": [{"name": "Ash Blossom"}]},
    }
    oracle = {"field": {"players": [{"lp": 8000, "hand_count_raw": 5, "grave_count_raw": 0}, {"lp": 7900, "hand_count_raw": 4, "grave_count_raw": 1}], "chain": []}, "tracked": {"turn_count": 2, "turn_player": 1}}
    prefix = [{"player": 0, "executed_action": {"arguments": {"command": "activate"}}, "engine_events": [{"msg_name": "MSG_CHAINING"}]}]
    truth = _state_truth(observation, oracle, prefix)
    assert truth["player0_banished_count"] == 2
    assert truth["player1_extra_deck_count"] == 14
    assert truth["player0_public_activation_count"] == 1
    assert truth["recent_public_engine_event_types"] == ["MSG_CHAINING"]


def test_exp3_reports_each_state_complexity_dimension() -> None:
    metrics = _state_metrics([{
        "ground_truth": {"turn": 1}, "prediction": {"turn": 1}, "trajectory_progress": 0.5,
        "state_complexity": {"chain_depth": 2, "zone_transition_events": 3, "historical_action_count": 30, "historical_public_activation_count": 1},
    }])
    assert set(metrics["jga_by_state_complexity_dimension"]) == {
        "chain_depth", "zone_transition_events", "historical_action_count", "historical_public_activation_count"
    }


def test_exp4_separates_action_retries_from_card_inspections() -> None:
    rows = [{
        "game_id": "g1", "player": 0,
        "legal": {"requires_model_reasoning": True, "expected_responder": "select_card"},
        "attempted_action": {"tool": "select_idlecmd", "arguments": {"index": 4}},
        "validation": {"valid": False, "model_action_attempts": 3, "protocol_errors": [], "pre_engine_error": "action_not_in_exact_legal_set"},
        "trace": {"turns": [{"tool_calls": [{"name": "inspect_card"}]}, {"tool_calls": [{"name": "select_idlecmd"}]}]},
    }]
    outcomes = [{"game_id": "g1", "agents": ["alpha", "beta"], "termination": "game_over"}]
    metrics = _execution_metrics(rows, outcomes)
    alpha = metrics["by_agent"][0]
    assert alpha["action_retries"] == 2
    assert alpha["inspect_card_calls"] == 1
    assert metrics["error_taxonomy"][0]["counts"]["wrong_responder"] == 1
    assert metrics["error_taxonomy"][0]["counts"]["invalid_index"] == 1


def test_combination_count_detects_enumeration_cap() -> None:
    assert _combination_count(5, 1, 1) == 5
    assert _combination_count(20, 1, 2) > 128


def test_exact_legal_action_match_uses_tool_and_arguments_only() -> None:
    legal = (
        ActionChoice(
            "select_battlecmd",
            {"command": "to_main_phase_2"},
            "legal label",
        ),
        ActionChoice("select_battlecmd", {"command": "to_end_phase"}),
    )
    assert exact_legal_action_match(
        ActionChoice("select_battlecmd", {"command": "to_main_phase_2"}, "model text"),
        legal,
    )
    assert not exact_legal_action_match(
        ActionChoice("select_battlecmd", {"command": "attack", "index": 0}),
        legal,
    )


def test_exact_legal_action_match_canonicalizes_unordered_card_indices() -> None:
    legal = (ActionChoice("select_card", {"indices": [0, 3], "cancel": False}),)
    assert exact_legal_action_match(
        ActionChoice("select_card", {"indices": [3, 0], "cancel": False}), legal
    )


def test_exact_legal_actions_packet_is_authoritative_and_copyable() -> None:
    from ygobench.engine.protocol import DecisionRequest

    packet = exact_legal_actions_packet(
        DecisionRequest(
            player=0,
            observation={},
            legal_actions=(
                ActionChoice("select_card", {"indices": [0, 3], "cancel": False}),
            ),
            decision_type="select_card",
            legal_actions_complete=True,
        )
    )
    assert "complete for this engine decision" in packet
    assert '"indices": [' in packet
    assert '"cancel": false' in packet


def test_tool_protocol_diagnostics_detects_missing_result() -> None:
    diagnostics = _tool_protocol_diagnostics(
        [
            {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
            {"role": "user", "content": "next"},
        ]
    )
    assert diagnostics["unresolved_tool_call_ids"] == 1
    assert "non_tool_message_before_all_tool_results" in diagnostics["protocol_errors"]


def test_tool_protocol_diagnostics_accepts_one_call_one_result() -> None:
    diagnostics = _tool_protocol_diagnostics(
        [
            {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
        ]
    )
    assert diagnostics["unresolved_tool_call_ids"] == 0
    assert diagnostics["protocol_errors"] == []


def test_forecast_response_window_skips_same_player_subdecisions() -> None:
    public = [
        {
            "player": 0,
            "turn": 1,
            "legal": {"expected_responder": "select_idlecmd"},
            "executed_action": {"arguments": {"command": "activate", "index": 0}},
        },
        {
            "player": 0,
            "turn": 1,
            "legal": {"expected_responder": "select_place"},
            "executed_action": {"arguments": {"places": []}},
        },
        {
            "player": 0,
            "turn": 1,
            "legal": {"expected_responder": "select_card"},
            "executed_action": {"arguments": {"indices": [0]}},
        },
        {
            "player": 1,
            "turn": 1,
            "legal": {"expected_responder": "select_chain"},
            "executed_action": {"arguments": {"index": None}},
        },
    ]
    response = _response_window_after(public, 0)
    assert response is not None
    assert response[0] == 3


def test_forecast_response_window_does_not_cross_new_decision() -> None:
    public = [
        {
            "player": 0,
            "turn": 1,
            "legal": {"expected_responder": "select_idlecmd"},
            "executed_action": {"arguments": {"command": "activate", "index": 0}},
        },
        {
            "player": 0,
            "turn": 1,
            "legal": {"expected_responder": "select_idlecmd"},
            "executed_action": {"arguments": {"command": "to_end_phase"}},
        },
        {
            "player": 1,
            "turn": 1,
            "legal": {"expected_responder": "select_chain"},
            "executed_action": {"arguments": {"index": 0}},
        },
    ]
    assert _response_window_after(public, 0) is None


def test_exact_legal_gate_forfeits_after_model_retries_are_exhausted(tmp_path) -> None:
    config = ExperimentConfig.build(
        run_id="pre-engine-gate",
        deck1="BlueEyes",
        deck2="BlueEyes",
        agent1="passive",
        agent2="passive",
        seed=23,
        max_decisions=1,
    )
    outcome = run_evidence_duel(
        config,
        root=tmp_path,
        interventions={
            0: ActionChoice("select_idlecmd", {"command": "attack", "index": 0})
        },
    )
    row = JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()[0]
    assert outcome["termination"] == "model_retry_exhausted_forfeit"
    assert outcome["winner"] == 1
    assert row["validation"]["pre_engine_error"].startswith(
        "action_not_in_exact_legal_set"
    )
    assert row["validation"]["exact_legal_check_performed"] is True
    assert row["validation"]["engine_submission_attempted"] is False
    assert row["validation"]["recovery"] == "model_retry_exhausted_forfeit"
    assert row["validation"]["terminal_model_failure"] is True
    assert row["validation"]["model_action_attempts"] == 3
    assert row["validation"]["attempted_invalid"] is True
    assert row["validation"]["engine_error"] is None
    assert row["engine_events"] == []
    assert row["oracle_before_hash"] == row["oracle_after_hash"]


def test_exact_legal_gate_accepts_exact_initial_chain_response(tmp_path) -> None:
    config = ExperimentConfig.build(
        run_id="pre-engine-gate-defaults",
        deck1="BlueEyes",
        deck2="BlueEyes",
        agent1="passive",
        agent2="passive",
        seed=29,
        max_decisions=1,
    )
    run_evidence_duel(
        config,
        root=tmp_path,
        interventions={
            0: ActionChoice("select_chain", {"index": None})
        },
    )
    row = JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()[0]
    assert row["validation"]["valid"] is True
    assert row["validation"]["pre_engine_error"] is None
    assert row["validation"]["engine_submission_attempted"] is True


def test_deepseek_requests_omit_token_limit() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return "ok"

    completions = SimpleNamespace(create=create)
    provider = SimpleNamespace(
        name="deepseek",
        max_tokens=64000,
        _client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    omit_deepseek_token_limit(provider)
    assert provider._client.chat.completions.create(model="deepseek", max_tokens=2048) == "ok"
    assert "max_tokens" not in captured
    assert provider.max_tokens is None


def test_dashscope_requests_omit_token_limit() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return "ok"

    completions = SimpleNamespace(create=create)
    provider = SimpleNamespace(
        name="dashscope",
        max_tokens=4096,
        _client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    omit_reasoning_model_token_limit(provider)
    assert provider._client.chat.completions.create(model="qwen3.8-max", max_tokens=2048) == "ok"
    assert "max_tokens" not in captured
    assert provider.max_tokens is None


def test_provider_requests_disable_parallel_tool_calls() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return "ok"

    provider = SimpleNamespace(
        _client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
    )
    force_single_tool_call(provider)
    assert provider._client.chat.completions.create(model="test") == "ok"
    assert captured["parallel_tool_calls"] is False


def test_deepseek_requests_explicitly_disable_thinking() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return "ok"

    provider = SimpleNamespace(
        name="deepseek",
        thinking_enabled=True,
        reasoning_effort="high",
        _client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    force_deepseek_thinking_mode(provider, enabled=False)
    result = provider._client.chat.completions.create(
        model="deepseek-v4-flash",
        extra_body={"existing": True},
    )

    assert result == "ok"
    assert captured["extra_body"] == {
        "existing": True,
        "thinking": {"type": "disabled"},
    }
    assert provider.thinking_enabled is False
    assert provider.reasoning_effort is None


def test_probe_provider_explicitly_disables_deepseek_thinking(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return "ok"

    provider = SimpleNamespace(
        name="deepseek",
        max_tokens=64000,
        thinking_enabled=True,
        reasoning_effort="high",
        _client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    monkeypatch.setattr(probes, "UpstreamLayout", lambda: SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(
        probes,
        "default_model_config",
        lambda **_kwargs: SimpleNamespace(
            backend="deepseek",
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url=None,
            api_key=None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "providers",
        SimpleNamespace(get_provider=lambda *_args, **_kwargs: provider),
    )

    configured = probes._provider("deepseek", "deepseek-v4-pro")
    assert configured._client.chat.completions.create(model="test", max_tokens=999) == "ok"
    assert "max_tokens" not in captured
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_dashscope_probe_requests_explicitly_disable_thinking() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return "ok"

    provider = SimpleNamespace(
        name="dashscope",
        thinking_enabled=True,
        _client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    force_provider_thinking_disabled(provider, provider_name="bailian")

    assert provider._client.chat.completions.create(model="qwen3.8-max") == "ok"
    assert captured["extra_body"] == {"enable_thinking": False}
    assert provider.thinking_enabled is False


def test_reasoning_content_is_removed_recursively() -> None:
    value = {
        "provider_data": {"reasoning_content": "private", "request_id": "r1"},
        "messages": [{"provider_data": {"reasoning_content": "also private"}}],
    }
    assert scrub_reasoning_content(value) == {
        "provider_data": {"request_id": "r1"},
        "messages": [{"provider_data": {}}],
    }


def test_jsonl_journal_recovers_partial_record(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_bytes(b'{"decision_id":"d1"}\n{"broken"')
    journal = JsonlJournal(path)
    assert journal.recover() == [{"decision_id": "d1"}]
    assert path.read_text() == '{"decision_id":"d1"}\n'


def test_registry_claim_and_finish(tmp_path) -> None:
    registry = TaskRegistry(tmp_path / "tasks.sqlite")
    registry.add("one", "duel", {"seed": 1})
    assert registry.claim("one")
    assert not registry.claim("one")
    registry.finish("one")
    assert registry.row("one")["status"] == "COMPLETED"


def test_trace_diagnostics_preserves_fallback_and_multiple_calls() -> None:
    trace = {
        "fallback": True,
        "turns": [
            {
                "tool_calls": [
                    {"name": "select_yesno"},
                    {"name": "select_yesno"},
                ]
            }
        ],
    }
    result = _trace_diagnostics(trace, "select_yesno")
    assert result["fallback"] is True
    assert result["protocol_errors"] == ["missing_tool_call", "multiple_response_calls"]


def test_token_usage_is_split_by_initial_inspection_and_illegal_correction() -> None:
    trace = {
        "turns": [
            {
                "request_stage": "initial_decision",
                "usage": {"input_tokens": 100, "output_tokens": 5},
            },
            {
                "request_stage": "card_inspection",
                "usage": {"input_tokens": 200, "output_tokens": 7},
            },
        ]
    }
    correction = {
        "attempts": [
            {
                "trace": {
                    "request_stage": "illegal_action_correction",
                    "usage": {"input_tokens": 30, "output_tokens": 3},
                }
            }
        ]
    }
    breakdown = _decision_token_usage_breakdown(trace, correction)
    assert breakdown["initial_decision_input_tokens"] == 100
    assert breakdown["card_inspection_input_tokens"] == 200
    assert breakdown["illegal_action_correction_input_tokens"] == 30
    assert breakdown["input_tokens_total"] == 330

    totals = _record_totals(
        [
            {
                "player": 0,
                "validation": {"valid": False},
                "elapsed_seconds": 1.0,
                "trace": trace,
                "correction_trace": correction,
            }
        ]
    )
    assert totals["input_tokens"] == [330, 0]
    assert totals["output_tokens"] == [15, 0]
    assert totals["model_calls"] == [3, 0]
    assert totals["input_tokens_by_stage"]["card_inspection"] == [200, 0]
    assert totals["input_tokens_by_stage"]["illegal_action_correction"] == [30, 0]


def test_phase0_reports_native_checkpoint_limitation() -> None:
    report = inspect_capabilities()
    assert report["capabilities"]["oracle_zone_query"]["supported"]
    assert not report["capabilities"]["checkpoint"]["native_serialization"]
    assert report["capabilities"]["checkpoint"]["deterministic_action_prefix_replay"]


def test_passive_evidence_runner_and_replay(tmp_path) -> None:
    config = ExperimentConfig.build(
        run_id="test-run",
        deck1="BlueEyes",
        deck2="BlueEyes",
        agent1="passive",
        agent2="passive",
        seed=17,
        max_decisions=3,
        checkpoint_interval=2,
    )
    outcome = run_evidence_duel(config, root=tmp_path)
    assert outcome["decisions"] == 3
    game_dir = config.game_dir(tmp_path)
    public = JsonlJournal(game_dir / "trajectory.jsonl").recover()
    oracle = JsonlJournal(game_dir / "oracle_trajectory.jsonl").recover()
    assert len(public) == len(oracle) == 3
    assert "before" not in public[0]
    assert oracle[0]["before"]["players"]
    assert public[0]["attempted_action"] == public[0]["executed_action"]
    report = verify_reversible_decisions(game_dir, sample_size=3)
    assert report["passed"], json.dumps(report, indent=2)
    assert all(row["legal_match"] for row in report["results"])
