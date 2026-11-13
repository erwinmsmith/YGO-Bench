"""Offline CJRouter SSE checks through the real SDK and vendored provider."""

import importlib
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

import ygobench.agents.llm_agent as llm_module
from ygobench.agents.llm_agent import LLMFullDuelAgent, ProviderCallError
from ygobench.agents.provider_streaming import (
    IncompleteStreamError,
    cjrouter_streaming_configuration,
    configure_cjrouter_streaming,
)
from ygobench.config import ModelConfig
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.experiments import probes
from ygobench.experiments.identity import policy_descriptor, policy_id
from ygobench.experiments.recovery import RecoveryStore

MODEL = "doubao-seed-2-1-lite-260915"
BASE = "https://cjrouter.cn/v1"
REASONING = "  synthetic reasoning\n中文\n"
USAGE = {
    "prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
    "completion_tokens_details": {"reasoning_tokens": 10},
    "prompt_tokens_details": {"cached_tokens": 3},
}


def _chunk(delta=None, *, finish=None, usage=None):
    return {
        "choices": [] if delta is None and finish is None else [
            {"index": 0, "delta": delta or {}, "finish_reason": finish}
        ],
        "usage": usage,
    }


def _tool_turn(name="select_yesno", arguments='{"accept":true}', *, call_id="call_1"):
    midpoint = len(arguments) // 2
    return [
        _chunk({"role": "assistant", "reasoning_content": REASONING[:10]}),
        _chunk({"reasoning_content": REASONING[10:], "content": "已"}),
        _chunk({"content": "选择", "tool_calls": [{
            "index": 0, "id": call_id[:5], "type": "function",
            "function": {"name": name[:4], "arguments": arguments[:midpoint]},
        }]}),
        _chunk({"tool_calls": [{
            "index": 0, "id": call_id[5:],
            "function": {"name": name[4:], "arguments": arguments[midpoint:]},
        }]}),
        _chunk({}, finish="tool_calls"),
        _chunk(usage=USAGE),
    ]


class EventBody(httpx.SyncByteStream):
    def __init__(self, events, model):
        self.events = events
        self.model = model
        self.closed = False

    def __iter__(self):
        yield b": heartbeat\n\n"
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            data = {
                "id": "chat-test", "created": 1, "model": self.model,
                "object": "chat.completion.chunk", **event,
            }
            wire = ("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode()
            # Exercise SSE decoding with HTTP byte boundaries inside UTF-8/JSON.
            for index in range(0, len(wire), 7):
                yield wire[index:index + 7]
        yield b"data: [DONE]\n\n"

    def close(self):
        self.closed = True


@pytest.fixture
def make_agent(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "vendor/yugi-bench/src"
    monkeypatch.syspath_prepend(str(source))
    provider_type = importlib.import_module("providers.openai").OpenAIToolProvider
    clients = []

    def factory(responses, *, model=MODEL, entry="agent"):
        pending = deque(responses)
        requests = []
        bodies = []

        def handle(request):
            payload = json.loads(request.content)
            requests.append(payload)
            assert request.url == BASE + "/chat/completions"
            body = EventBody(pending.popleft(), payload["model"])
            bodies.append(body)
            return httpx.Response(
                200, stream=body,
                headers={
                    "content-type": "text/event-stream",
                    "x-request-id": "test-request", "authorization": "not-for-logs",
                },
            )

        provider = provider_type(model=model, api_key="sk-test", base_url=BASE)
        provider._client.close()
        provider._client = OpenAI(
            api_key="sk-test", base_url=BASE, max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        clients.append(provider._client)
        monkeypatch.setattr(llm_module, "_provider_runtime", lambda: (
            SimpleNamespace(TOOLS=[{
                "name": "select_yesno", "description": "Choose yes or no",
                "input_schema": {
                    "type": "object", "properties": {"accept": {"type": "boolean"}},
                    "required": ["accept"],
                },
            }]),
            lambda *_args, **_kwargs: provider,
        ))
        config = ModelConfig(
            provider="external", model=model, base_url=BASE,
            api_key="sk-test", backend="openai",
        )
        if entry == "probe":
            provider_module = importlib.import_module("providers")
            monkeypatch.setattr(provider_module, "get_provider", lambda *_a, **_k: provider)
            monkeypatch.setattr(probes, "default_model_config", lambda **_k: config)
            return probes._provider("external", model), requests, bodies
        agent = LLMFullDuelAgent(config, profile="react-fast")
        agent._max_provider_connection_attempts = 1
        return agent, requests, bodies

    yield factory
    for client in clients:
        client.close()


@pytest.mark.parametrize(("model", "toggle"), [
    ("qwen3.8-flash", "enable_thinking"),
    ("DeepSeek-V4.1-Flash-1017", "thinking"),
    ("MiniMax-M3.1-Flash", None),
    ("glm-5.3-flash", "thinking"),
    (MODEL, "thinking"),
    ("gemini-3.8-flash", None),
    ("gpt-6-luna", None),
])
def test_selected_models_stream_and_continue_tools_without_changing_policy(
    make_agent, model, toggle
):
    agent, requests, bodies = make_agent([
        _tool_turn("inspect_cards", '{"card_ids":[1]}'),
        _tool_turn(),
    ], model=model)
    provider = agent._provider
    first = provider.respond(system="system", messages=[{"role": "user", "content": "check"}],
                             tools=[agent._tool_defs["inspect_cards"]])
    assert first.text == "已选择"
    assert first.provider_data == {"reasoning_content": REASONING}
    assert first.tool_calls[0].id == "call_1"
    assert first.tool_calls[0].name == "inspect_cards"
    assert first.tool_calls[0].arguments == {"card_ids": [1]}
    assert first.usage["input_tokens"] == 120
    assert first.usage["output_tokens"] == 30
    assert first.usage["total_tokens"] == 150
    assert first.usage["reasoning_tokens"] == 10
    assert first.usage["cached_tokens"] == 3
    assert first.response_headers == {"x-request-id": "test-request"}
    assert first.raw_dict["model"] == model

    second = provider.respond(system="system", messages=[
        {"role": "user", "content": "check"},
        {"role": "assistant", "text": first.text, "tool_calls": [{
            "id": first.tool_calls[0].id, "name": first.tool_calls[0].name,
            "arguments": first.tool_calls[0].arguments,
        }], "provider_data": first.provider_data},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"card":1}'},
    ], tools=[agent._tool_defs["select_yesno"]])
    assert second.tool_calls[0].arguments == {"accept": True}
    assert requests[1]["messages"][2]["reasoning_content"] == REASONING
    assert requests[1]["messages"][3]["tool_call_id"] == "call_1"
    assert all(body.closed for body in bodies)
    for request in requests:
        assert request["stream"] is True
        assert request["stream_options"] == {"include_usage": True}
        assert request["reasoning_effort"] == "low"
        assert request["model"] == model
        assert request["temperature"] == 0.0
        assert request["seed"] == 0
        assert request["parallel_tool_calls"] is False
        assert "max_tokens" not in request
        if toggle:
            assert request[toggle] == (True if toggle == "enable_thinking"
                                       else {"type": "enabled"})
    descriptor = policy_descriptor(agent.name, runtime=agent.provider_config)
    assert descriptor["runtime_stream"] is True
    assert descriptor["runtime_stream_options"] == {"include_usage": True}
    assert descriptor["runtime_response_transport"] == "cjrouter-sse-v1"
    transport_keys = {"runtime_stream", "runtime_stream_options", "runtime_response_transport"}
    previous = {k: v for k, v in descriptor.items() if k not in transport_keys}
    assert policy_id(previous) != policy_id(descriptor)


def test_text_only_stream_and_conflicting_transport_overrides(make_agent):
    agent, requests, bodies = make_agent([[
        _chunk({"role": "assistant", "content": "O"}),
        _chunk({"content": "K"}, finish="stop", usage=USAGE),
    ]])
    response = agent._provider._client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "check"}], tools=[],
        stream=False, stream_options={"include_usage": False},
        max_tokens=10, reasoning_effort="high",
        extra_body={"stream": False, "stream_options": {"include_usage": False}, "custom": 1},
    )
    assert response.choices[0].message.content == "OK"
    assert response.choices[0].message.tool_calls is None
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"]["include_usage"] is True
    assert requests[0]["reasoning_effort"] == "low"
    assert requests[0]["custom"] == 1
    assert "max_tokens" not in requests[0]
    assert bodies[0].closed


def test_interleaved_tool_fragments_remain_separate(make_agent):
    agent, _, _ = make_agent([[
        _chunk({"tool_calls": [
            {"index": 1, "id": "b", "type": "function",
             "function": {"name": "inspect_", "arguments": '{"card_ids":['}},
            {"index": 0, "id": "a", "type": "function",
             "function": {"name": "inspect_cards", "arguments": '{"card_ids":[1'}},
        ]}),
        _chunk({"tool_calls": [
            {"index": 0, "function": {"arguments": "]}"}},
            {"index": 1, "function": {"name": "cards", "arguments": "2]}"}},
        ]}, finish="tool_calls"),
        _chunk(usage=USAGE),
    ]])
    turn = agent._provider.respond(system="s", messages=[], tools=[])
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [
        ("a", "inspect_cards", {"card_ids": [1]}),
        ("b", "inspect_cards", {"card_ids": [2]}),
    ]


@pytest.mark.parametrize("case", [
    "empty", "no_finish", "no_usage", "pre_finish_usage",
    "length", "bad_json", "array_arguments", "no_id",
    "read_error", "error_after_finish", "sse_error",
])
def test_failed_stream_is_closed_and_never_saved_as_a_success(make_agent, tmp_path, case):
    events = _tool_turn()
    if case == "empty":
        events = []
    elif case == "no_finish":
        events = events[:4] + events[5:]
    elif case == "no_usage":
        events = events[:-1]
    elif case == "pre_finish_usage":
        events = events[:-2] + [events[-1], events[-2]]
    elif case == "length":
        events[-2]["choices"][0]["finish_reason"] = "length"
    elif case == "bad_json":
        events = _tool_turn(arguments='{"accept":')
    elif case == "array_arguments":
        events = _tool_turn(arguments="[]")
    elif case == "no_id":
        events = _tool_turn(call_id="")
    elif case == "read_error":
        events = events[:3] + [httpx.ReadError("synthetic interrupted stream")]
    elif case == "error_after_finish":
        events += [httpx.ReadError("synthetic interrupted stream")]
    elif case == "sse_error":
        events = events[:3] + [{"error": {"message": "synthetic SSE failure"}}]

    agent, requests, bodies = make_agent([events])
    store = RecoveryStore(tmp_path / "recovery.sqlite")
    with pytest.raises(Exception) as error:
        store.call(scope_id="test", call_index=0, request={"fixture": 1},
                   invoke=lambda: agent._provider.respond(system="s", messages=[], tools=[]))
    if case not in {"read_error", "error_after_finish", "sse_error"}:
        assert isinstance(error.value, IncompleteStreamError)
    assert REASONING not in str(error.value)
    assert bodies[0].closed
    assert len(requests) == 1
    with store.connect() as db:
        row = db.execute("SELECT status,response_json FROM api_calls").fetchone()
    assert tuple(row) == ("REQUEST_ERROR", None)


def _decision():
    return DecisionRequest(
        player=0,
        observation={
            "decision": {"responder": "select_yesno"},
            "you": {"hand": [{"code": 1, "name": "Fixture card", "description": "test"}]},
        },
        legal_actions=(
            ActionChoice("select_yesno", {"accept": False}),
            ActionChoice("select_yesno", {"accept": True}),
        ),
        decision_type="select_yesno",
    )


def test_streamed_inspection_survives_restart_and_partial_followup_is_retried(
    make_agent, tmp_path
):
    path = tmp_path / "recovery.sqlite"
    initial, first_requests, bodies = make_agent([
        _tool_turn("inspect_cards", '{"card_ids":[1]}'),
        _tool_turn()[:3] + [httpx.ReadError("synthetic interruption")],
    ])
    initial.set_recovery_scope(RecoveryStore(path), "duel:g:d000104:p1")
    with pytest.raises(ProviderCallError):
        initial.predict(_decision())
    assert len(first_requests) == 2
    assert all(body.closed for body in bodies)

    with RecoveryStore(path).connect() as db:
        rows = db.execute(
            "SELECT call_index,status,response_json FROM api_calls ORDER BY call_index"
        ).fetchall()
    assert rows[0]["status"] == "RESPONSE_SAVED"
    snapshot = json.loads(rows[0]["response_json"])
    assert snapshot["provider_data"]["reasoning_content"] == REASONING
    assert snapshot["usage"]["reasoning_tokens"] == 10
    assert snapshot["response_headers"] == {"x-request-id": "test-request"}
    assert rows[1]["status"] == "REQUEST_ERROR"
    assert rows[1]["response_json"] is None

    resumed, next_requests, _ = make_agent([_tool_turn()])
    resumed.set_recovery_scope(RecoveryStore(path), "duel:g:d000104:p1")
    action = resumed.predict(_decision())
    assert action.arguments == {"accept": True}
    assert len(next_requests) == 1
    assistant = next(m for m in next_requests[0]["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == REASONING
    assert assistant["tool_calls"][0]["function"]["name"] == "inspect_cards"
    assert next_requests[0] == first_requests[1]
    assert resumed.last_trace["transport_attempts"][0]["outcome"] == "saved_response_reused"
    public = json.dumps(resumed.last_trace, ensure_ascii=False)
    assert "reasoning_content" not in public
    assert REASONING not in public
    assert resumed.usage["total_tokens"] == 300


def test_existing_retry_starts_a_fresh_stream_accumulator(make_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)
    agent, requests, _ = make_agent([
        _tool_turn()[:3] + [httpx.ReadError("synthetic interruption")],
        _tool_turn(),
    ])
    agent._max_provider_connection_attempts = 2
    agent.set_recovery_scope(RecoveryStore(tmp_path / "recovery.sqlite"), "duel:g:d1:p1")
    assert agent.predict(_decision()).arguments == {"accept": True}
    assert requests[0] == requests[1]
    assert [t["outcome"] for t in agent.last_trace["transport_attempts"]] == ["error", "success"]
    assert agent.usage["total_tokens"] == 150


@pytest.mark.parametrize(("profile", "url"), [
    ("external", "https://other.example/v1"),
    ("external", "https://cjrouter.cn.other.example/v1"),
    ("openai", BASE),
    ("external", "http://cjrouter.cn/v1"),
    ("external", None),
])
def test_other_profiles_and_endpoints_are_unchanged(profile, url):
    def create(**kwargs):
        return kwargs

    def original_log():
        return {"name": "fixture"}
    provider = SimpleNamespace(
        _client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        provider_config_for_log=original_log,
    )
    assert cjrouter_streaming_configuration(profile, url) == {}
    assert configure_cjrouter_streaming(provider, provider_name=profile, base_url=url) is provider
    assert provider._client.chat.completions.create is create
    assert provider.provider_config_for_log is original_log


def test_probe_entry_uses_same_stream_transport_and_records_it(make_agent):
    configured, requests, _ = make_agent([_tool_turn()], entry="probe")
    create = configured._client.chat.completions.create
    configure_cjrouter_streaming(configured, provider_name="external", base_url=BASE)
    assert configured._client.chat.completions.create is create
    assert configured.respond(system="s", messages=[], tools=[]).tool_calls
    assert len(requests) == 1
    assert requests[0]["stream"] is True
    descriptor = probes.probe_evaluator_identity("external", MODEL)["configuration"]
    assert descriptor["runtime_stream"] is True
    assert descriptor["runtime_response_transport"] == "cjrouter-sse-v1"
