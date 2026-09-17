import json
from types import SimpleNamespace

import pytest

from backend import replay_service
from ygobench.agents.action_space import (
    OPCODE_ISCODE,
    OPCODE_OR,
    _declarable,
)
from ygobench.agents.llm_agent import (
    LLMFullDuelAgent,
    ProviderProtocolError,
    _find_card,
    _select_place_tool,
    compact_prompt_state,
)
from ygobench.bench.duel_metrics import summarize_duels
from ygobench.engine.full_duel import _normalize_action
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.engine.visibility import sanitize_events_for_player


def test_find_card_searches_all_nested_zones() -> None:
    observation = {
        "you": {"hand": [{"code": 1, "name": "First"}]},
        "opponent": {"monster_zone": [{"code": 99, "name": "Target"}]},
    }
    assert _find_card(observation, 99)["name"] == "Target"
    assert _find_card(observation, 404)["name"] is None


def test_announce_card_postfix_filter() -> None:
    opcodes = [123, OPCODE_ISCODE, 456, OPCODE_ISCODE, OPCODE_OR]
    assert _declarable({"code": 123, "alias": 0, "type": 1}, [], opcodes)
    assert _declarable({"code": 456, "alias": 0, "type": 1}, [], opcodes)
    assert not _declarable({"code": 789, "alias": 0, "type": 1}, [], opcodes)


def test_llm_agent_skips_model_for_forced_response() -> None:
    agent = LLMFullDuelAgent.__new__(LLMFullDuelAgent)
    request = DecisionRequest(
        player=0,
        observation={},
        legal_actions=(ActionChoice("select_chain", {"index": None}),),
        decision_type="select_chain",
    )
    action = agent.predict(request)
    assert action.arguments == {"index": None}
    assert agent.last_trace["automatic"] is True


def test_select_place_uses_semantic_zone_names_before_engine_normalization() -> None:
    core = SimpleNamespace(LOCATION_MZONE=4, LOCATION_SZONE=8)
    tools = SimpleNamespace(coerce_args=lambda tool, args: {"tool": tool, **args})
    normalized = _normalize_action(
        ActionChoice(
            "select_place",
            {"places": [{"player": 0, "location": "monster_zone", "sequence": 2}]},
        ),
        core,
        tools,
    )
    assert normalized == {
        "tool": "select_place",
        "places": [{"player": 0, "location": 4, "sequence": 2}],
    }

    schema = _select_place_tool()["input_schema"]
    location = schema["properties"]["places"]["items"]["properties"]["location"]
    assert location == {
        "type": "string",
        "enum": ["monster_zone", "spell_zone", "pendulum_zone"],
    }


def test_llm_agent_retries_length_response_with_action_only() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    text="",
                    tool_calls=[],
                    stop_reason="length",
                    usage={},
                    wallclock_seconds=0.1,
                    provider_data={},
                )
            return SimpleNamespace(
                text="",
                tool_calls=[
                    SimpleNamespace(
                        id="call-1",
                        name="select_yesno",
                        arguments={"accept": True},
                    )
                ],
                stop_reason="tool_calls",
                usage={},
                wallclock_seconds=0.1,
                provider_data={},
            )

    agent = LLMFullDuelAgent.__new__(LLMFullDuelAgent)
    agent._provider = Provider()
    agent._tool_defs = {
        "inspect_cards": {"name": "inspect_cards"},
        "select_yesno": {"name": "select_yesno"},
    }
    agent._max_inspection_rounds = 1
    agent._max_forced_retries = 2
    agent._system_prompt = "system"
    agent._observation_template = "{{STATE_JSON}} {{REQUIRED_RESPONDER}}"
    agent.usage = {}
    agent.model_calls = 0
    agent.invalid_outputs = 0
    agent.last_trace = {}
    request = DecisionRequest(
        player=0,
        observation={"decision": {"responder": "select_yesno"}},
        legal_actions=(
            ActionChoice("select_yesno", {"accept": False}),
            ActionChoice("select_yesno", {"accept": True}),
        ),
        decision_type="select_yesno",
    )
    action = agent.predict(request)
    assert action.arguments == {"accept": True}
    assert agent._provider.calls == 2
    assert agent.last_trace["fallback"] is False


def test_llm_agent_batches_one_inspection_round_then_requires_action() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def respond(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    text="inspect together",
                    tool_calls=[
                        SimpleNamespace(
                            id="inspect-1",
                            name="inspect_cards",
                            arguments={"card_ids": [1, 99, 1]},
                        )
                    ],
                    stop_reason="tool_calls",
                    usage={"input_tokens": 100, "output_tokens": 5},
                    wallclock_seconds=0.1,
                    provider_data={},
                )
            assert [tool["name"] for tool in kwargs["tools"]] == ["select_yesno"]
            tool_messages = [m for m in kwargs["messages"] if m["role"] == "tool"]
            assert len(tool_messages) == 1
            result = json.loads(tool_messages[0]["content"])
            assert result["requested_card_ids"] == [1, 99]
            assert [card["name"] for card in result["cards"]] == ["First", "Target"]
            assert result["cache_hits"] == []
            assert result["cache_added"] == [1, 99]
            assert result["next_required_tool"] == "select_yesno"
            return SimpleNamespace(
                text="",
                tool_calls=[
                    SimpleNamespace(
                        id="action-1",
                        name="select_yesno",
                        arguments={"accept": True},
                    )
                ],
                stop_reason="tool_calls",
                usage={"input_tokens": 200, "output_tokens": 7},
                wallclock_seconds=0.1,
                provider_data={},
            )

    agent = LLMFullDuelAgent.__new__(LLMFullDuelAgent)
    agent._provider = Provider()
    agent._tool_defs = {
        "inspect_cards": {"name": "inspect_cards"},
        "select_yesno": {"name": "select_yesno"},
    }
    agent._max_inspection_rounds = 1
    agent._max_forced_retries = 2
    agent._system_prompt = "system"
    agent._observation_template = "{{STATE_JSON}} {{REQUIRED_RESPONDER}}"
    agent.usage = {}
    agent.model_calls = 0
    agent.invalid_outputs = 0
    agent.last_trace = {}
    request = DecisionRequest(
        player=0,
        observation={
            "decision": {"responder": "select_yesno"},
            "you": {"hand": [{"code": 1, "name": "First"}]},
            "opponent": {"monster_zone": [{"code": 99, "name": "Target"}]},
        },
        legal_actions=(
            ActionChoice("select_yesno", {"accept": False}),
            ActionChoice("select_yesno", {"accept": True}),
        ),
        decision_type="select_yesno",
    )

    action = agent.predict(request)

    assert action.arguments == {"accept": True}
    assert len(agent._provider.calls) == 2
    stages = agent.last_trace["input_token_breakdown"]["stages"]
    assert stages["initial_decision"]["input_tokens"] == 100
    assert stages["card_inspection"]["input_tokens"] == 200
    assert agent.last_trace["card_cache"] == {
        "available_before": [],
        "hits": [],
        "added": [1, 99],
        "size_after": 2,
    }


def test_llm_agent_reuses_duel_card_cache_and_reset_clears_it() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def respond(self, **kwargs):
            self.calls.append(kwargs)
            call_number = len(self.calls)
            if call_number == 1:
                return SimpleNamespace(
                    text="",
                    tool_calls=[
                        SimpleNamespace(
                            id="inspect-1",
                            name="inspect_cards",
                            arguments={"card_ids": [7]},
                        )
                    ],
                    stop_reason="tool_calls",
                    usage={},
                    wallclock_seconds=0.1,
                    provider_data={},
                )
            if call_number == 3:
                prompt = kwargs["messages"][0]["content"]
                assert "Duel-level inspected card cache" in prompt
                assert '"code": 7' in prompt
                assert '"name": "Cached Card"' in prompt
                assert '"description": "Previously verified text."' in prompt
            return SimpleNamespace(
                text="",
                tool_calls=[
                    SimpleNamespace(
                        id=f"action-{call_number}",
                        name="select_yesno",
                        arguments={"accept": True},
                    )
                ],
                stop_reason="tool_calls",
                usage={},
                wallclock_seconds=0.1,
                provider_data={},
            )

    agent = LLMFullDuelAgent.__new__(LLMFullDuelAgent)
    agent._provider = Provider()
    agent._tool_defs = {
        "inspect_cards": {"name": "inspect_cards"},
        "select_yesno": {"name": "select_yesno"},
    }
    agent._max_inspection_rounds = 1
    agent._max_forced_retries = 2
    agent._system_prompt = "system"
    agent._observation_template = "{{STATE_JSON}} {{REQUIRED_RESPONDER}}"
    agent.usage = {}
    agent.model_calls = 0
    agent.invalid_outputs = 0
    agent.last_trace = {}
    agent._inspected_card_cache = {}
    request = DecisionRequest(
        player=0,
        observation={
            "decision": {"responder": "select_yesno"},
            "you": {
                "hand": [
                    {
                        "code": 7,
                        "name": "Cached Card",
                        "description": "Previously verified text.",
                        "attack": 9999,
                        "location": "hand",
                    }
                ]
            },
        },
        legal_actions=(
            ActionChoice("select_yesno", {"accept": False}),
            ActionChoice("select_yesno", {"accept": True}),
        ),
        decision_type="select_yesno",
    )

    agent.predict(request)
    assert agent._inspected_card_cache == {
        7: {
            "code": 7,
            "name": "Cached Card",
            "description": "Previously verified text.",
        }
    }
    agent.predict(request)
    assert agent.last_trace["card_cache"] == {
        "available_before": [7],
        "hits": [],
        "added": [],
        "size_after": 1,
    }

    agent.reset()
    assert agent._inspected_card_cache == {}


def test_llm_agent_forfeits_after_three_missing_action_attempts() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def respond(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                text="no action",
                tool_calls=[],
                stop_reason="stop",
                usage={},
                wallclock_seconds=0.1,
                provider_data={},
            )

    agent = LLMFullDuelAgent.__new__(LLMFullDuelAgent)
    agent._provider = Provider()
    agent._tool_defs = {
        "inspect_cards": {"name": "inspect_cards"},
        "select_yesno": {"name": "select_yesno"},
    }
    agent._max_inspection_rounds = 1
    agent._max_forced_retries = 2  # Initial attempt + two corrections = three total.
    agent._max_provider_connection_attempts = 3
    agent._system_prompt = "system"
    agent._observation_template = "{{STATE_JSON}} {{REQUIRED_RESPONDER}}"
    agent.usage = {}
    agent.model_calls = 0
    agent.invalid_outputs = 0
    agent.last_trace = {}
    # A single legal action is deliberately automatic, so use two exact options.
    request = DecisionRequest(
        player=0,
        observation={"decision": {"responder": "select_yesno"}},
        legal_actions=(
            ActionChoice("select_yesno", {"accept": False}),
            ActionChoice("select_yesno", {"accept": True}),
        ),
        decision_type="select_yesno",
    )
    with pytest.raises(ProviderProtocolError) as error:
        agent.predict(request)

    assert agent._provider.calls == 3
    assert error.value.diagnostics["model_action_attempts"] == 3
    assert agent.last_trace["fallback"] is False
    assert agent.last_trace["terminal_model_failure"] is True


def test_compact_prompt_state_keeps_tactics_but_drops_card_text() -> None:
    state = {
        "perspective_player": 1,
        "phase": "main1",
        "decision": {"responder": "select_idlecmd"},
        "you": {
            "lp": 8000,
            "hand_count": 1,
            "hand": [{"code": 7, "name": "Card", "description": "oracle text"}],
        },
        "opponent": {"lp": 7900},
    }
    compact = compact_prompt_state(state)
    assert compact["perspective_player"] == 1
    assert compact["you"]["hand"][0] == {"code": 7, "name": "Card"}


def test_private_opponent_draw_codes_are_redacted() -> None:
    events = [
        {
            "msg_name": "MSG_DRAW",
            "player": 1,
            "count": 2,
            "cards": [
                {"code": 111, "position": 10},
                {"code": 222, "position": 10},
            ],
        },
        {
            "msg_name": "MSG_DRAW",
            "player": 0,
            "count": 1,
            "cards": [{"code": 333, "position": 10}],
        },
    ]

    sanitized = sanitize_events_for_player(events, perspective=0)

    assert sanitized[0]["cards"] == [
        {"face_down": True, "position": 10},
        {"face_down": True, "position": 10},
    ]
    assert sanitized[1]["cards"][0]["code"] == 333


def test_arena_summary_tracks_seats_and_illegal_actions() -> None:
    games = [
        {
            "agents": ["alpha", "beta"],
            "decks": ["A", "B"],
            "winner": 0,
            "game_over": True,
            "turn_count": 3,
            "decisions": 8,
            "decisions_by_player": [5, 3],
            "illegal_actions": [0, 1],
            "decision_seconds": [0.5, 0.6],
            "model_usage_totals": {"input_tokens": [100, 20], "output_tokens": [10, 2]},
        },
        {
            "agents": ["beta", "alpha"],
            "decks": ["B", "A"],
            "winner": 1,
            "game_over": True,
            "turn_count": 4,
            "decisions": 10,
            "decisions_by_player": [4, 6],
            "illegal_actions": [0, 0],
            "decision_seconds": [0.4, 0.6],
            "model_usage_totals": {"input_tokens": [20, 100], "output_tokens": [2, 10]},
        },
    ]
    metrics = summarize_duels(games)
    assert metrics["benchmark_type"] == "full_duel_arena"
    assert metrics["engine_completion_rate"] == 1
    alpha = next(row for row in metrics["leaderboard"] if row["agent"] == "alpha")
    beta = next(row for row in metrics["leaderboard"] if row["agent"] == "beta")
    assert alpha["win_rate"] == 1
    assert alpha["first_win_rate"] == 1
    assert alpha["second_win_rate"] == 1
    assert beta["illegal_action_rate"] == pytest.approx(1 / 7)


def test_replay_frames_join_state_action_and_result(tmp_path, monkeypatch) -> None:
    run = tmp_path / "run"
    run.mkdir()
    replay = run / "duel.jsonl"
    events = [
        {"type": "config", "agents": ["alpha", "beta"]},
        {
            "type": "observation",
            "player": 0,
            "state": {
                "perspective_player": 0,
                "phase": "main1",
                "you": {"lp": 8000, "monster_zone": []},
                "opponent": {"lp": 8000, "monster_zone": []},
            },
        },
        {
            "type": "model_turn",
            "player": 0,
            "agent": "alpha",
            "tool_calls": [{"name": "select_idlecmd", "arguments": {"command": "end"}}],
            "trace": {"fallback": True},
        },
        {"type": "tool_result", "events": [{"event": "phase_changed"}]},
        {"type": "outcome", "winner": 0},
    ]
    replay.write_text("\n".join(json.dumps(event) for event in events))
    monkeypatch.setattr(replay_service, "RUNS_ROOT", tmp_path)

    result = replay_service.load_replay_frames("run", "duel.jsonl")
    assert result is not None
    assert len(result["frames"]) == 1
    assert result["frames"][0]["action"]["agent"] == "alpha"
    assert result["frames"][0]["engine_events"] == [{"event": "phase_changed"}]
    assert result["frames"][0]["fallback"] is True
    assert result["outcome"]["winner"] == 0
