"""Legal-action evidence with an explicit completeness contract."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from math import comb
from typing import Any

from ygobench.agents.action_space import legal_actions_from_pending

# These responders are enumerated directly from finite choices in the pending
# engine message. Other responders retain the bounded policy action set and are
# never presented as an exact legality oracle.
EXACT_RESPONDERS = {
    "rock_paper_scissors",
    "select_battlecmd",
    "select_chain",
    "select_effectyn",
    "select_idlecmd",
    "select_option",
    "select_position",
    "select_unselect_card",
    "select_yesno",
    "announce_number",
}


def _action_signature(action: Any) -> tuple[str, str]:
    return (
        action.tool,
        json.dumps(action.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def exact_legal_action_match(
    action: Any,
    legal_actions: tuple[Any, ...],
    *,
    signature: Callable[[Any], Any] = _action_signature,
) -> bool:
    """Return whether an action exactly matches a concrete legal candidate.

    Labels are explanatory model text and are not part of the engine response.
    The tool name and JSON arguments are the complete comparison key.
    """

    action_key = signature(action)
    return any(action_key == signature(legal) for legal in legal_actions)


def _combination_count(size: int, minimum: int, maximum: int) -> int:
    return sum(
        comb(size, count)
        for count in range(max(0, minimum), min(maximum, size) + 1)
    )


def build_legal_evidence(
    pending: Any,
    *,
    card_db: Any,
    replay_module: Any,
    state_module: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    actions = legal_actions_from_pending(
        pending,
        card_db=card_db,
        replay_module=replay_module,
        state_module=state_module,
    )
    decision = state_module.build_decision(pending, card_db)
    responder = str(decision.get("responder", ""))
    complete = responder in EXACT_RESPONDERS
    if responder == "select_card":
        size = len(decision.get("cards", []))
        complete = (
            _combination_count(
                size,
                int(decision.get("min", 1)),
                int(decision.get("max", 1)),
            )
            <= 128
        )
    elif responder == "select_place":
        size = len(decision.get("places", []))
        count = int(decision.get("count", 1))
        complete = _combination_count(size, count, count) <= 64
    return actions, {
        "expected_responder": responder,
        "legal_action_count": len(actions),
        "requires_model_reasoning": len(actions) > 1,
        "forced": len(actions) == 1,
        "enumeration_complete": complete,
        "enumeration_method": "pending_message_exact" if complete else "bounded_policy_candidates",
        "incompleteness_reason": (
            None
            if complete
            else "engine legality is not fully enumerable or candidate enumeration hit its cap"
        ),
        "actions": [asdict(action) for action in actions],
    }
