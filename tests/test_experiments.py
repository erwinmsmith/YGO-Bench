import json
from types import SimpleNamespace

from ygobench.agents.provider_limits import (
    force_single_tool_call,
    omit_deepseek_token_limit,
    scrub_reasoning_content,
)
from ygobench.agents.llm_agent import (
    ProviderProtocolError,
    _tool_protocol_diagnostics,
    exact_legal_actions_packet,
)
from ygobench.engine.protocol import ActionChoice
from ygobench.experiments.capabilities import inspect_capabilities
from ygobench.experiments.config import ExperimentConfig, stable_id
from ygobench.experiments.io import JsonlJournal
from ygobench.experiments.legal import (
    _combination_count,
    exact_legal_action_match,
)
from ygobench.experiments.probes import _response_window_after
from ygobench.experiments.registry import TaskRegistry
from ygobench.experiments.replay import verify_reversible_decisions
from ygobench.experiments.runner import _trace_diagnostics, run_evidence_duel


def test_stable_id_is_deterministic() -> None:
    assert stable_id("game", {"seed": 1}) == stable_id("game", {"seed": 1})
    assert stable_id("game", {"seed": 1}) != stable_id("game", {"seed": 2})


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


def test_exact_legal_gate_recovers_with_fallback_before_engine_submission(tmp_path) -> None:
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
    assert outcome["termination"] != "illegal_action_forfeit"
    assert row["validation"]["pre_engine_error"].startswith(
        "action_not_in_exact_legal_set"
    )
    assert row["validation"]["exact_legal_check_performed"] is True
    assert row["validation"]["engine_submission_attempted"] is False
    assert row["validation"]["recovery"] == "deterministic_fallback"
    assert row["validation"]["attempted_invalid"] is True
    assert row["validation"]["engine_error"] is None
    assert row["engine_events"]
    assert row["oracle_before_hash"] != row["oracle_after_hash"]


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
