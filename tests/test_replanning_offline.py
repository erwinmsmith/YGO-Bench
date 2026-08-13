from ygobench.experiments.io import JsonlJournal
from ygobench.experiments.replanning import find_offline_counterfactual_candidates


def _row(
    decision_id: str,
    *,
    player: int,
    responder: str,
    action: dict,
    actions: list[dict],
) -> dict:
    return {
        "game_id": "game_test",
        "decision_id": decision_id,
        "decision_index": int(decision_id[1:]),
        "turn": 1,
        "player": player,
        "validation": {"valid": True},
        "executed_action": action,
        "legal": {
            "expected_responder": responder,
            "enumeration_complete": True,
            "actions": actions,
        },
    }


def test_offline_exp6_requires_an_opponent_interruption_after_commitment(tmp_path) -> None:
    game_dir = tmp_path / "game"
    journal = JsonlJournal(game_dir / "trajectory.jsonl")
    pass_chain = {"tool": "select_chain", "arguments": {"index": None}}
    interrupt = {"tool": "select_chain", "arguments": {"index": 0}}
    journal.append(
        _row(
            "d000001",
            player=0,
            responder="select_idlecmd",
            action={"tool": "select_idlecmd", "arguments": {"command": "activate", "index": 0}},
            actions=[],
        )
    )
    # This is the focal player's own optional chain and must not be selected.
    journal.append(
        _row(
            "d000002",
            player=0,
            responder="select_chain",
            action=pass_chain,
            actions=[pass_chain, interrupt],
        )
    )
    journal.append(
        _row(
            "d000003",
            player=1,
            responder="select_chain",
            action=pass_chain,
            actions=[pass_chain, interrupt],
        )
    )

    found = find_offline_counterfactual_candidates(game_dir)

    assert found["commitment_points"] == 1
    assert found["eligible_counterfactuals"] == 1
    candidate = found["candidates"][0]
    assert candidate["focal_player"] == 0
    assert candidate["interruption_player"] == 1
    assert candidate["response_decision_id"] == "d000003"
    assert candidate["counterfactual_response"] == interrupt
