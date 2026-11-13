import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ygobench.experiments.runner as runner
from ygobench.agents.llm_agent import LLMFullDuelAgent
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.experiments.config import ExperimentConfig
from ygobench.experiments.io import JsonlJournal
from ygobench.experiments.recovery import RecoveryConflict, RecoveryStore
from ygobench.experiments.session import DuelSession


def make_agent(provider, path=None):
    agent = LLMFullDuelAgent.__new__(LLMFullDuelAgent)
    agent._provider = provider
    agent.provider_config = {"thinking_enabled": True, "reasoning_effort": "low"}
    agent._tool_defs = {
        name: {"name": name}
        for name in (
            "inspect_cards",
            "select_yesno",
            "announce_card",
        )
    }
    agent._max_inspection_rounds = 1
    agent._max_forced_retries = 2
    agent._max_provider_connection_attempts = 1
    agent._system_prompt = "system"
    agent._observation_template = "{{STATE_JSON}} {{REQUIRED_RESPONDER}}"
    agent.reset()
    if path is not None:
        agent.set_recovery_scope(RecoveryStore(path), "test:decision")
    return agent


def response(tool, arguments):
    return SimpleNamespace(
        text="",
        tool_calls=[SimpleNamespace(id="call", name=tool, arguments=arguments)],
        stop_reason="tool_calls",
        usage={"input_tokens": 10, "output_tokens": 2},
        wallclock_seconds=0.1,
        provider_data={"reasoning_content": "private fixture"},
    )


def test_reordered_committed_history_reuses_saved_response_but_changed_state_conflicts(tmp_path):
    path = tmp_path / "recovery.sqlite"
    decision = DecisionRequest(
        player=0,
        observation={
            "decision": {"responder": "select_yesno", "options": [True, False]},
            "recent_actions": [
                {
                    "decision_id": "d000001",
                    "action_id": "a1",
                    "turn": 1,
                    "player": 1,
                    "tool": "select_idlecmd",
                    "arguments": {"command": "activate", "index": 0},
                }
            ],
        },
        legal_actions=(
            ActionChoice("select_yesno", {"accept": True}),
            ActionChoice("select_yesno", {"accept": False}),
        ),
        decision_type="select_yesno",
        legal_actions_complete=True,
    )
    agent = make_agent(
        SimpleNamespace(respond=lambda **kw: response("select_yesno", {"accept": True})), path
    )
    agent.predict(decision)
    journal = JsonlJournal(tmp_path / "trajectory.jsonl")
    journal.append({"observation": decision.observation})
    replayed = replace(decision, observation=journal.recover()[0]["observation"])
    assert list(decision.observation["recent_actions"][0]) != list(
        replayed.observation["recent_actions"][0]
    )
    resumed = make_agent(
        SimpleNamespace(
            respond=lambda **kw: pytest.fail(
                "A reordered journal must not cause a new provider request"
            )
        ),
        path,
    )
    assert resumed.predict(replayed).arguments == {"accept": True}
    assert resumed.last_trace["transport_attempts"][0]["outcome"] == "saved_response_reused"
    replayed.observation["recent_actions"][0]["arguments"]["index"] = 9
    resumed.set_recovery_scope(RecoveryStore(path), "test:decision")
    with pytest.raises(RecoveryConflict):
        resumed.predict(replayed)


class SyntheticSession:
    instances = []
    fail_replay = False

    def __init__(self, _deck1, _deck2, *, seed, deck_shuffle_seeds):
        self.progress = 0
        self.closed = False
        self.engine_seeds = (seed,) * 4
        self.deck_shuffle_seeds = deck_shuffle_seeds or (1, 2)
        self.deck_order_hashes = ("deck0", "deck1")
        self.duel = SimpleNamespace(
            state=SimpleNamespace(
                lp=[8000, 8000], turn_count=1, game_over=False, winner=None, win_reason=None
            ),
            pending=True,
        )
        self.instances.append(self)

    @property
    def done(self):
        return self.duel.state.game_over or self.duel.pending is None

    def oracle(self):
        return {"state_hash": f"state-{self.progress}", "tracked": {"lp": [8000, 8000]}}

    def view(self, recent_actions):
        responder = "select_yesno" if self.progress == 0 else "announce_card"
        actions = (
            (ActionChoice(responder, {"accept": True}), ActionChoice(responder, {"accept": False}))
            if self.progress == 0
            else (
                ActionChoice(responder, {"card_code": 2}),
                ActionChoice(responder, {"card_code": 3}),
            )
        )
        return {
            "player": 0,
            "observation": {
                "turn": 1,
                "decision": {"responder": responder},
                "recent_actions": recent_actions[-8:],
            },
            "actions": actions,
            "legal": {"expected_responder": responder, "enumeration_complete": self.progress == 0},
            "oracle": self.oracle(),
        }

    def execute(self, action):
        if self.progress == 0:
            assert action.tool == "select_yesno"
            self.progress = 88 if self.fail_replay and len(self.instances) > 1 else 1
        else:
            assert self.progress == 1, "Must reconstruct the rejected engine first"
            if action.arguments.get("card_code") != 2:
                self.progress = 999
                self.duel.pending = None
                raise runner.EngineResponseRejected("Engine rejected response (MSG_RETRY 6 times).")
            self.progress = 2
            self.duel.state.game_over = True
            self.duel.state.winner = 0
            self.duel.state.win_reason = 1
            self.duel.pending = None
        return SimpleNamespace(events=[])

    def action_signature(self, action):
        return action.tool, action.arguments

    def close(self):
        self.closed = True


def setup_duel(monkeypatch, provider):
    SyntheticSession.instances = []
    SyntheticSession.fail_replay = False
    monkeypatch.setattr(runner, "DuelSession", SyntheticSession)
    monkeypatch.setattr(runner, "build_oracle_state", lambda s: s.oracle())
    monkeypatch.setattr(runner, "create_agent", lambda *a, **kw: make_agent(provider))
    monkeypatch.setattr(runner, "_export_web_replay", lambda *a: None)
    return ExperimentConfig.build(
        run_id="rejection",
        deck1="BlueEyes",
        deck2="Branded",
        agent1="react-fast:test:model",
        agent2="react-fast:test:model",
        seed=607,
    )


class Provider:
    def __init__(self, *, fail=False, always_invalid=False):
        self.fail = fail
        self.always_invalid = always_invalid
        self.calls = []

    def respond(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["tools"][-1]["name"] == "select_yesno":
            return response("select_yesno", {"accept": True})
        if len(kwargs["tools"]) == 2:
            return response("announce_card", {"card_code": 1})
        packet = json.loads(kwargs["messages"][0]["content"])
        assert packet["engine_rejections"][-1]["action"]["arguments"] == {"card_code": 1}
        assert "not an authoritative legal set" in kwargs["system"]
        assert "legal_responses" not in packet
        assert "oracle" not in packet
        if self.fail:
            raise TimeoutError("simulated API interruption during correction")
        return response("announce_card", {"card_code": 1 if self.always_invalid else 2})


def test_rejected_action_corrected_after_verified_replay_and_recorded(monkeypatch, tmp_path):
    provider = Provider()
    config = setup_duel(monkeypatch, provider)
    outcome = runner.run_evidence_duel(config, root=tmp_path)
    assert outcome["termination"] == "game_over"
    assert outcome["technical_valid"] is True
    assert outcome["illegal_actions"] == [1, 0]
    assert outcome["model_usage_totals"]["model_calls"] == [3, 0]
    assert len(provider.calls) == 3
    assert len(SyntheticSession.instances) == 2
    assert all(s.closed for s in SyntheticSession.instances)
    row = JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()[-1]
    assert row["attempted_action"]["arguments"] == {"card_code": 1}
    assert row["executed_action"]["arguments"] == {"card_code": 2}
    assert row["validation"]["recovery"] == "corrected_retry"
    assert row["validation"]["model_action_attempts"] == 2
    assert row["validation"]["engine_error"] is None
    assert row["validation"]["engine_rejection_error"]
    assert row["correction_trace"]["engine_rejections"]
    assert "reasoning_content" not in json.dumps(row)


def test_resume_rejection_after_api_interruption_reuses_initial_request(monkeypatch, tmp_path):
    provider = Provider(fail=True)
    config = setup_duel(monkeypatch, provider)
    first = runner.run_evidence_duel(config, root=tmp_path)
    assert first["termination"] == "api_failure_abort"
    rows = JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()
    assert len(rows) == 1
    assert len(provider.calls) == 3
    provider.fail = False
    result = runner.run_evidence_duel(config, root=tmp_path)
    assert result["termination"] == "game_over"
    assert len(provider.calls) == 4  # Only the interrupted correction calls the API again.
    rows = JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()
    assert len(rows) == 2
    assert rows[-1]["trace"]["transport_attempts"][0]["outcome"] == "saved_response_reused"
    assert (
        rows[-1]["correction_trace"]["attempts"][0]["trace"]["transport_attempts"][0]["outcome"]
        == "success"
    )


def test_engine_rejections_obey_three_attempt_limit(monkeypatch, tmp_path):
    provider = Provider(always_invalid=True)
    config = setup_duel(monkeypatch, provider)
    result = runner.run_evidence_duel(config, root=tmp_path)
    assert result["termination"] == "model_retry_exhausted_forfeit"
    assert result["winner"] == 1
    assert result["technical_valid"] is True
    assert len(provider.calls) == 4  # One preceding decision, three attempts at the next.
    row = JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()[-1]
    assert row["validation"]["model_action_attempts"] == 3
    assert row["executed_action"] is None
    assert row["oracle_before_hash"] == row["oracle_after_hash"]
    assert len(row["correction_trace"]["engine_rejections"]) == 3


def test_replay_mismatch_aborts_without_asking_model_to_correct(monkeypatch, tmp_path):
    provider = Provider()
    config = setup_duel(monkeypatch, provider)
    SyntheticSession.fail_replay = True
    result = runner.run_evidence_duel(config, root=tmp_path)
    assert result["termination"] == "engine_failure_abort"
    assert result["technical_valid"] is False
    assert "transition mismatch" in result["technical_failure_error"]
    assert len(provider.calls) == 2
    assert len(JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()) == 1
    assert all(s.closed for s in SyntheticSession.instances)


@pytest.mark.parametrize(
    "message, rejected",
    [
        ("Engine rejected response (MSG_RETRY 6 times).", True),
        ("unexpected engine failure", False),
    ],
)
def test_session_only_classifies_explicit_retry(monkeypatch, message, rejected):
    import ygobench.experiments.session as module

    class HarnessError(RuntimeError):
        pass

    def reject(**kwargs):
        raise HarnessError(message)

    session = DuelSession.__new__(DuelSession)
    session.harness_module = SimpleNamespace(HarnessError=HarnessError)
    session.tools_module = SimpleNamespace(TOOL_TO_HARNESS_METHOD={"announce_card": "respond"})
    session.core = None
    session.duel = SimpleNamespace(respond=reject)
    monkeypatch.setattr(module, "_normalize_action", lambda a, *_: a.arguments)
    expected = module.EngineResponseRejected if rejected else HarnessError
    with pytest.raises(expected):
        session.execute(ActionChoice("announce_card", {"card_code": 1}))


def test_saved_correction_survives_crash_before_decision_commit(monkeypatch, tmp_path):
    provider = Provider()
    config = setup_duel(monkeypatch, provider)
    original = RecoveryStore.commit_decision
    interrupted = False

    def crash_once(store, game_id, row, oracle_row, **kwargs):
        nonlocal interrupted
        if row["decision_index"] == 2 and not interrupted:
            interrupted = True
            raise OSError("simulated interruption after the API response was saved")
        return original(store, game_id, row, oracle_row, **kwargs)

    monkeypatch.setattr(RecoveryStore, "commit_decision", crash_once)
    with pytest.raises(OSError):
        runner.run_evidence_duel(config, root=tmp_path)
    assert len(provider.calls) == 3
    result = runner.run_evidence_duel(config, root=tmp_path)
    assert result["termination"] == "game_over"
    assert len(provider.calls) == 3
    rows = JsonlJournal(config.game_dir(tmp_path) / "trajectory.jsonl").recover()
    assert len(rows) == 2
    correction = rows[-1]["correction_trace"]["attempts"][0]["trace"]
    assert correction["transport_attempts"][0]["outcome"] == "saved_response_reused"


def test_rejection_of_exact_legal_action_is_still_a_technical_failure(monkeypatch, tmp_path):
    provider = Provider()
    config = setup_duel(monkeypatch, provider)

    def reject_exact(_self, _action):
        raise runner.EngineResponseRejected("Engine rejected response (MSG_RETRY 6 times).")

    monkeypatch.setattr(SyntheticSession, "execute", reject_exact)
    result = runner.run_evidence_duel(config, root=tmp_path)
    assert result["technical_failure_type"] == "engine_failure"
    assert result["technical_valid"] is False
    assert len(provider.calls) == 1


def test_real_engine_msg_retry_is_restored_before_accepting_correct_response(monkeypatch):
    config = ExperimentConfig.build(
        run_id="native-rejection",
        deck1="BlueEyes",
        deck2="BlueEyes",
        agent1="passive",
        agent2="passive",
        seed=29,
    )
    decks = runner.PROJECT_ROOT / "resources" / "decks"
    session = DuelSession(
        decks / "BlueEyes.ydk", decks / "BlueEyes.ydk", seed=config.seed, deck_shuffle_seeds=None
    )
    restored = None
    try:
        view = session.view([])
        assert view["legal"]["expected_responder"] == "select_chain"
        # Bypass the Python range check to exercise a real ocgcore MSG_RETRY.
        monkeypatch.setattr(
            session.duel, "respond_select_chain", lambda **kw: session.duel._respond_int(2147483647)
        )
        with pytest.raises(runner.EngineResponseRejected):
            session.execute(ActionChoice("select_chain", {"index": None}))
        restored = runner._restore_rejected_session(config, [], view)
        restored.execute(ActionChoice("select_chain", {"index": None}))
        assert restored.view([])["oracle"]["state_hash"] != view["oracle"]["state_hash"]
    finally:
        session.close()
        if restored is not None:
            restored.close()


def test_real_engine_replays_committed_prefix_with_original_shuffle_seeds():
    from dataclasses import asdict

    config = ExperimentConfig.build(
        run_id="native-prefix",
        deck1="BlueEyes",
        deck2="Branded",
        agent1="passive",
        agent2="passive",
        seed=607,
        deck_shuffle_seeds=(13804715669135220558, 1670743812400104014),
    )
    decks = runner.PROJECT_ROOT / "resources" / "decks"
    session = DuelSession(
        decks / "BlueEyes.ydk",
        decks / "Branded.ydk",
        seed=config.seed,
        deck_shuffle_seeds=config.deck_shuffle_seeds,
    )
    restored = None
    rows = []
    recent = []
    agent = runner.create_agent("passive")
    try:
        for index in range(3):
            view = session.view(recent)
            action = agent.predict(
                DecisionRequest(
                    player=view["player"],
                    observation=view["observation"],
                    legal_actions=view["actions"],
                    decision_type=view["legal"]["expected_responder"],
                    legal_actions_complete=view["legal"]["enumeration_complete"],
                )
            )
            session.execute(action)
            summary = {
                "decision_id": f"d{index + 1:06d}",
                "action_id": f"a{index}",
                "turn": 1,
                "player": view["player"],
                "tool": action.tool,
                "arguments": action.arguments,
            }
            rows.append(
                {
                    "executed_action": asdict(action),
                    "action_summary": summary,
                    "oracle_before_hash": view["oracle"]["state_hash"],
                    "oracle_after_hash": runner.build_oracle_state(session)["state_hash"],
                    "legal": view["legal"],
                }
            )
            recent.append(summary)
        expected = session.view(recent)
        restored = runner._restore_rejected_session(
            config, json.loads(json.dumps(rows, sort_keys=True)), expected
        )
        assert restored.deck_shuffle_seeds == config.deck_shuffle_seeds
        assert restored.deck_order_hashes == session.deck_order_hashes
        assert runner.build_oracle_state(restored) == runner.build_oracle_state(session)
    finally:
        session.close()
        if restored is not None:
            restored.close()
