"""Visibility guards for information passed to duel agents."""

from __future__ import annotations

from typing import Any


def _hidden_card_stub(card: Any) -> dict[str, Any]:
    if not isinstance(card, dict):
        return {"face_down": True}
    stub: dict[str, Any] = {"face_down": True}
    if "position" in card:
        stub["position"] = card["position"]
    return stub


def sanitize_events_for_player(
    events: list[dict[str, Any]] | None,
    *,
    perspective: int,
) -> list[dict[str, Any]]:
    """Redact private opponent identities from raw ocgcore event payloads.

    The upstream state builder correctly hides the opponent hand, but raw draw and
    set events can still contain database codes. Those codes must never enter an
    agent prompt or an evidence replay from that player's perspective.
    """

    sanitized: list[dict[str, Any]] = []
    for raw_event in events or []:
        event = dict(raw_event)
        name = event.get("msg_name")
        event_player = event.get("player")

        if name == "MSG_DRAW" and event_player != perspective:
            cards = event.get("cards") or []
            event["count"] = int(event.get("count") or len(cards))
            event["cards"] = [_hidden_card_stub(card) for card in cards]
        elif name == "MSG_SHUFFLE_HAND" and event_player != perspective:
            codes = event.pop("codes", [])
            event["count"] = len(codes)
        elif name == "MSG_SET" and event.get("con") != perspective:
            event.pop("code", None)
        elif name == "MSG_MOVE":
            previous = event.get("previous") or {}
            current = event.get("current") or {}
            controller = current.get("con", previous.get("con"))
            if (
                controller != perspective
                and _location_is_hidden(previous)
                and _location_is_hidden(current)
            ):
                event.pop("code", None)

        sanitized.append(event)
    return sanitized


def _location_is_hidden(location: dict[str, Any]) -> bool:
    zone = int(location.get("loc", 0) or 0)
    position = int(location.get("pos", 0) or 0)
    return zone in {1, 2, 64} or bool(position & 0x8)
