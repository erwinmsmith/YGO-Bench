import json
from types import SimpleNamespace

import pytest

from backend import replay_service
from ygobench.agents.action_space import (
    OPCODE_ISCODE,
    OPCODE_OR,
    _declarable,
)
from ygobench.agents.llm_agent import LLMFullDuelAgent, _find_card, compact_prompt_state
from ygobench.bench.duel_metrics import summarize_duels
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
        "inspect_card": {"name": "inspect_card"},
        "select_yesno": {"name": "select_yesno"},
    }
    agent._max_inspections = 4
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
