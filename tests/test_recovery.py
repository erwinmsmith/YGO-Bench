import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ygobench.agents.llm_agent import LLMFullDuelAgent
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.experiments.io import content_hash
from ygobench.experiments.recovery import RecoveryStore

REASONING = "  synthetic reasoning fixture\n\u4e2d\u6587\n"


def _turn(provider_data, *, inspection=False):
    return SimpleNamespace(
        text="",
        tool_calls=[SimpleNamespace(
            id="inspect-1" if inspection else "action-1",
            name="inspect_cards" if inspection else "select_yesno",
            arguments={"card_ids": [1]} if inspection else {"accept": True},
        )],
        stop_reason="tool_calls",
        usage={"input_tokens": 10, "output_tokens": 2},
        wallclock_seconds=0.1,
        response_headers={"x-request-id": "r1", "authorization": "secret-header"},
        provider_data=provider_data,
        raw_dict={"secret": "raw-response-must-not-be-saved"},
    )


def _agent(provider, path):
    agent = LLMFullDuelAgent.__new__(LLMFullDuelAgent)
    agent._provider = provider
    agent.provider_config = {"thinking_enabled": True, "reasoning_effort": "low"}
    agent._tool_defs = {
        "inspect_cards": {"name": "inspect_cards"},
        "select_yesno": {"name": "select_yesno"},
    }
    agent._max_inspection_rounds = 1
    agent._max_forced_retries = 2
    agent._max_provider_connection_attempts = 1
    agent._system_prompt = "system"
    agent._observation_template = "{{STATE_JSON}} {{REQUIRED_RESPONDER}}"
    agent.usage = {}
    agent.model_calls = 0
    agent.invalid_outputs = 0
    agent.last_trace = {}
    agent.set_recovery_scope(RecoveryStore(path), "game:decision:0")
    return agent


@pytest.mark.parametrize("backend", ["openai", "deepseek"])
def test_thinking_tool_call_resumes_without_repeating_saved_response(
    tmp_path, monkeypatch, backend
):
    source = Path(__file__).resolve().parents[1] / "vendor" / "yugi-bench" / "src"
    monkeypatch.syspath_prepend(str(source))
    module = pytest.importorskip(f"providers.{backend}")
    translate = getattr(module, f"_to_{backend}_messages")

    class SimulatedCrash(BaseException):
        pass

    calls = []

    def interrupted(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _turn({"reasoning_content": REASONING}, inspection=True)
        raise SimulatedCrash

    path = tmp_path / "recovery.sqlite"
    decision = DecisionRequest(
        player=0,
        observation={
            "decision": {"responder": "select_yesno"},
            "you": {"hand": [{"code": 1, "name": "Fixture card"}]},
        },
        legal_actions=(
            ActionChoice("select_yesno", {"accept": False}),
            ActionChoice("select_yesno", {"accept": True}),
        ),
        decision_type="select_yesno",
    )
    initial = _agent(SimpleNamespace(respond=interrupted), path)
    with pytest.raises(SimulatedCrash):
        initial.predict(decision)
    assert len(calls) == 2

    resumed_calls = []

    def continued(**kwargs):
        wire = translate(kwargs["system"], kwargs["messages"])
        resumed_calls.append(wire)
        assistant = next(message for message in wire if message["role"] == "assistant")
        assert assistant.get("reasoning_content") == REASONING
        assert assistant["tool_calls"][0]["id"] == "inspect-1"
        tool = next(message for message in wire if message["role"] == "tool")
        assert tool["tool_call_id"] == "inspect-1"
        return _turn({"reasoning_content": "synthetic final fixture"})

    resumed = _agent(SimpleNamespace(respond=continued), path)
    action = resumed.predict(decision)
    assert action.arguments == {"accept": True}
    assert len(resumed_calls) == 1
    assert resumed.last_trace["transport_attempts"][0]["outcome"] == "saved_response_reused"
    assert "reasoning_content" not in json.dumps(resumed.last_trace)
    assert REASONING not in json.dumps(resumed.last_trace, ensure_ascii=False)


@pytest.mark.parametrize("provider_data", [
    {},
    {"reasoning_content": ""},
    {"reasoning_content": None},
    {"reasoning_content": REASONING, "request_id": "r1"},
])
def test_recovery_preserves_response_fields_without_storing_raw_response(tmp_path, provider_data):
    path = tmp_path / "recovery.sqlite"
    request = {"model": "fixture"}
    original = _turn(provider_data)
    saved = RecoveryStore(path).call(
        scope_id="s", call_index=0, request=request, invoke=lambda: original
    )
    assert saved.reused is False

    reopened = RecoveryStore(path)
    restored = reopened.call(
        scope_id="s", call_index=0, request=request,
        invoke=lambda: pytest.fail("A saved response must not invoke the provider again"),
    )
    assert restored.reused is True
    assert restored.attempt == 1
    assert restored.turn.provider_data == provider_data
    assert restored.turn.usage == original.usage
    assert restored.turn.tool_calls[0].arguments == {"accept": True}
    assert restored.turn.response_headers == {"x-request-id": "r1"}

    with reopened.connect() as db:
        payload = db.execute("SELECT response_json FROM api_calls").fetchone()[0]
    snapshot = json.loads(payload)
    assert snapshot["snapshot_version"] == 2
    assert "raw_dict" not in snapshot
    assert "secret-header" not in payload
    assert "raw-response-must-not-be-saved" not in payload


def test_legacy_snapshot_without_reasoning_remains_readable(tmp_path):
    path = tmp_path / "recovery.sqlite"
    request = {"model": "legacy"}
    snapshot = {
        "text": "",
        "tool_calls": [{"id": "old", "name": "select_yesno", "arguments": {"accept": True}}],
        "stop_reason": "tool_calls",
        "usage": {},
        "wallclock_seconds": 0.1,
        "provider_data": {},
    }
    store = RecoveryStore(path)
    with store.connect() as db:
        db.execute(
            "INSERT INTO api_calls "
            "(scope_id,call_index,request_hash,status,attempt_count,response_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("legacy", 0, content_hash(request), "RESPONSE_SAVED", 1, json.dumps(snapshot)),
        )
    reopened = RecoveryStore(path)
    restored = reopened.call(
        scope_id="legacy", call_index=0, request=request,
        invoke=lambda: pytest.fail("Reading a legacy snapshot must not cause a new API call"),
    )
    assert restored.reused is True
    assert restored.turn.provider_data == {}
    assert restored.turn.tool_calls[0].id == "old"
    with reopened.connect() as db:
        payload = db.execute("SELECT response_json FROM api_calls").fetchone()[0]
    assert json.loads(payload) == snapshot
