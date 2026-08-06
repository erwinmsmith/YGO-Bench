"""Translate one ocgcore pending decision into concrete legal actions."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any

from ygobench.engine.protocol import ActionChoice

OPCODE_ADD = 0x4000000000000000
OPCODE_SUB = 0x4000000100000000
OPCODE_MUL = 0x4000000200000000
OPCODE_DIV = 0x4000000300000000
OPCODE_AND = 0x4000000400000000
OPCODE_OR = 0x4000000500000000
OPCODE_NEG = 0x4000000600000000
OPCODE_NOT = 0x4000000700000000
OPCODE_BAND = 0x4000000800000000
OPCODE_BOR = 0x4000000900000000
OPCODE_BNOT = 0x4000001000000000
OPCODE_BXOR = 0x4000001100000000
OPCODE_LSHIFT = 0x4000001200000000
OPCODE_RSHIFT = 0x4000001300000000
OPCODE_ALLOW_ALIASES = 0x4000001400000000
OPCODE_ALLOW_TOKENS = 0x4000001500000000
OPCODE_ISCODE = 0x4000010000000000
OPCODE_ISSETCARD = 0x4000010100000000
OPCODE_ISTYPE = 0x4000010200000000
OPCODE_ISRACE = 0x4000010300000000
OPCODE_ISATTRIBUTE = 0x4000010400000000
OPCODE_GETCODE = 0x4000010500000000
OPCODE_GETTYPE = 0x4000010700000000
OPCODE_GETRACE = 0x4000010800000000
OPCODE_GETATTRIBUTE = 0x4000010900000000


def _declarable(card: dict[str, Any], setcodes: list[int], opcodes: list[int]) -> bool:
    """Mirror ocgcore's postfix announce-card filter."""

    stack: list[int] = []
    allow_aliases = allow_tokens = False
    binary = {
        OPCODE_ADD: lambda left, right: left + right,
        OPCODE_SUB: lambda left, right: left - right,
        OPCODE_MUL: lambda left, right: left * right,
        OPCODE_DIV: lambda left, right: int(left / right) if right else 0,
        OPCODE_AND: lambda left, right: int(bool(left) and bool(right)),
        OPCODE_OR: lambda left, right: int(bool(left) or bool(right)),
        OPCODE_BAND: lambda left, right: left & right,
        OPCODE_BOR: lambda left, right: left | right,
        OPCODE_BXOR: lambda left, right: left ^ right,
        OPCODE_LSHIFT: lambda left, right: left << right,
        OPCODE_RSHIFT: lambda left, right: left >> right,
    }
    for opcode in opcodes:
        if opcode in binary and len(stack) >= 2:
            right, left = stack.pop(), stack.pop()
            stack.append(binary[opcode](left, right))
        elif opcode == OPCODE_NEG and stack:
            stack.append(-stack.pop())
        elif opcode == OPCODE_NOT and stack:
            stack.append(int(not stack.pop()))
        elif opcode == OPCODE_BNOT and stack:
            stack.append(~stack.pop())
        elif opcode == OPCODE_ISCODE and stack:
            stack.append(int(card.get("code", 0) == (stack.pop() & 0xFFFFFFFF)))
        elif opcode == OPCODE_ISTYPE and stack:
            stack.append(int(bool(card.get("type", 0) & stack.pop())))
        elif opcode == OPCODE_ISRACE and stack:
            stack.append(int(bool(card.get("race", 0) & stack.pop())))
        elif opcode == OPCODE_ISATTRIBUTE and stack:
            stack.append(int(bool(card.get("attribute", 0) & stack.pop())))
        elif opcode == OPCODE_ISSETCARD and stack:
            requested = stack.pop()
            set_type, set_subtype = requested & 0xFFF, requested & 0xF000
            stack.append(
                int(
                    any(
                        (value & 0xFFF) == set_type
                        and (value & 0xF000 & set_subtype) == set_subtype
                        for value in setcodes
                    )
                )
            )
        elif opcode == OPCODE_GETCODE:
            stack.append(int(card.get("code", 0)))
        elif opcode == OPCODE_GETTYPE:
            stack.append(int(card.get("type", 0)))
        elif opcode == OPCODE_GETRACE:
            stack.append(int(card.get("race", 0)))
        elif opcode == OPCODE_GETATTRIBUTE:
            stack.append(int(card.get("attribute", 0)))
        elif opcode == OPCODE_ALLOW_ALIASES:
            allow_aliases = True
        elif opcode == OPCODE_ALLOW_TOKENS:
            allow_tokens = True
        else:
            stack.append(opcode)
    card_type = int(card.get("type", 0))
    is_token = card_type & 0x4001 == 0x4001
    return (
        len(stack) == 1
        and bool(stack[0])
        and (allow_aliases or not card.get("alias"))
        and (allow_tokens or not is_token)
    )


def _announce_card_actions(card_db: Any, opcodes: list[int], limit: int = 64):
    cache = getattr(card_db, "_cache", {})
    emitted = 0
    for code, card in cache.items():
        if _declarable(card, card_db.get_setcodes(code), opcodes):
            yield _choice("announce_card", {"card_code": code}, card.get("name", str(code)))
            emitted += 1
            if emitted >= limit:
                return


def _choice(tool: str, arguments: dict[str, Any], label: str = "") -> ActionChoice:
    return ActionChoice(tool=tool, arguments=arguments, label=label or f"{tool} {arguments}")


def _bounded_combinations(size: int, min_count: int, max_count: int, limit: int = 128):
    emitted = 0
    for count in range(max(0, min_count), min(max_count, size) + 1):
        for indices in combinations(range(size), count):
            yield list(indices)
            emitted += 1
            if emitted >= limit:
                return


def legal_actions_from_pending(
    pending: Any,
    *,
    card_db: Any,
    replay_module: Any,
    state_module: Any,
) -> tuple[ActionChoice, ...]:
    """Return a bounded, concrete action set for rule/random agents.

    The engine remains the final legality oracle. Combinatorial selection
    prompts are capped because some cards can expose hundreds of subsets.
    The first action is always the upstream passive/first-legal response.
    """

    decision = state_module.build_decision(pending, card_db)
    responder = str(decision.get("responder", ""))
    passive_tool, passive_args = replay_module._pick_passive_opponent_response(pending)
    if pending.msg_name == "rock_paper_scissors":
        passive_args = {"hand": 1 if pending.player == 0 else 2}
    if responder == "select_place":
        count = int(decision.get("count", 1))
        passive_args = {"places": list(decision.get("places", []))[:count]}
    actions: list[ActionChoice] = [_choice(passive_tool, passive_args, "passive / first legal")]

    if responder in {"select_idlecmd", "select_battlecmd"}:
        for option in decision.get("choices", []):
            args = {key: option[key] for key in ("command", "index") if key in option}
            actions.append(_choice(responder, args, str(option.get("card", option))))
    elif responder in {"select_effectyn", "select_yesno"}:
        actions.extend(
            (
                _choice(responder, {"accept": False}, "decline"),
                _choice(responder, {"accept": True}, "accept"),
            )
        )
    elif responder == "select_option":
        actions.extend(
            _choice(responder, {"index": idx}, str(value))
            for idx, value in enumerate(decision.get("options", []))
        )
    elif responder in {"select_card", "select_tribute"}:
        cards = decision.get("cards", [])
        for indices in _bounded_combinations(
            len(cards), int(decision.get("min", 1)), int(decision.get("max", 1))
        ):
            names = [cards[index].get("name", f"#{index}") for index in indices]
            actions.append(
                _choice(
                    responder,
                    {"indices": indices, "cancel": False},
                    ", ".join(names),
                )
            )
        if decision.get("cancelable"):
            actions.append(_choice(responder, {"indices": [], "cancel": True}, "cancel"))
    elif responder == "select_unselect_card":
        size = len(decision.get("selectable_cards", [])) + len(
            decision.get("selected_cards", [])
        )
        actions.extend(
            _choice(responder, {"index": idx}, f"card index {idx}") for idx in range(size)
        )
        if decision.get("finishable") or decision.get("cancelable"):
            actions.append(_choice(responder, {"index": None}, "finish"))
    elif responder == "select_chain":
        actions.extend(
            _choice(responder, {"index": idx}, card.get("name", f"chain {idx}"))
            for idx, card in enumerate(decision.get("cards", []))
        )
        if not decision.get("forced"):
            actions.append(_choice(responder, {"index": None}, "decline chain"))
    elif responder == "select_position":
        actions.extend(
            _choice(responder, {"position": position}, name)
            for position, name in zip(
                decision.get("allowed_positions", []),
                decision.get("allowed_positions_named", []),
                strict=False,
            )
        )
    elif responder == "select_place":
        places = decision.get("places", [])
        count = int(decision.get("count", 1))
        for picks in _bounded_combinations(len(places), count, count, limit=64):
            chosen = [places[index] for index in picks]
            actions.append(_choice(responder, {"places": chosen}, str(chosen)))
    elif responder == "select_sum":
        cards = decision.get("optional_cards", [])
        for indices in _bounded_combinations(
            len(cards),
            int(decision.get("min", 1)),
            int(decision.get("max", len(cards))),
            limit=128,
        ):
            actions.append(_choice(responder, {"indices": indices}, f"sum indices {indices}"))
    elif responder == "announce_number":
        actions.extend(
            _choice(responder, {"index": idx}, str(number))
            for idx, number in enumerate(decision.get("numbers", []))
        )
    elif responder == "announce_card":
        candidates = list(_announce_card_actions(card_db, decision.get("opcodes", [])))
        if candidates:
            actions = [candidates[0], *candidates[1:]]
    elif responder == "rock_paper_scissors":
        actions.extend(
            _choice(responder, {"hand": hand}, label)
            for hand, label in ((1, "rock"), (2, "scissors"), (3, "paper"))
        )

    unique: dict[tuple[str, str], ActionChoice] = {}
    for action in actions:
        key = (action.tool, json.dumps(action.arguments, sort_keys=True))
        unique.setdefault(key, action)
    return tuple(unique.values())
