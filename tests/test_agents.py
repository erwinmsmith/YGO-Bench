from ygobench.agents.random_agent import RandomAgent
from ygobench.engine.protocol import ActionChoice, DecisionRequest


def test_random_agent_returns_legal_action() -> None:
    actions = (
        ActionChoice("select_yesno", {"accept": True}),
        ActionChoice("select_yesno", {"accept": False}),
    )
    request = DecisionRequest(
        player=0,
        observation={},
        legal_actions=actions,
        decision_type="select_yesno",
    )
    assert RandomAgent(seed=7).predict(request) in actions


def test_random_agent_rejects_empty_action_set() -> None:
    request = DecisionRequest(player=0, observation={}, legal_actions=(), decision_type="end")
    try:
        RandomAgent().predict(request)
    except ValueError as exc:
        assert "No legal actions" in str(exc)
    else:
        raise AssertionError("expected ValueError")

