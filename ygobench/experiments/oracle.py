"""Oracle-only engine snapshots. Never pass these records to an agent."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from ygobench.experiments.io import content_hash


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, bytes):
        return {"hex": value.hex(), "length": len(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_oracle_state(session: Any) -> dict[str, Any]:
    core = session.core
    engine = session.engine
    locations = {
        "deck": core.LOCATION_DECK,
        "hand": core.LOCATION_HAND,
        "monster_zone": core.LOCATION_MZONE,
        "spell_trap_zone": core.LOCATION_SZONE,
        "graveyard": core.LOCATION_GRAVE,
        "banished": core.LOCATION_REMOVED,
        "extra_deck": core.LOCATION_EXTRA,
    }
    players = []
    for player in (0, 1):
        zones = {
            name: _jsonable(engine.query_location(player, location, core.QUERY_FULL_CARD))
            for name, location in locations.items()
        }
        players.append({"player": player, "zones": zones})
    pending = session.duel.pending
    value = {
        "players": players,
        "field": _jsonable(engine.query_field()),
        "field_raw": engine.query_field_raw().hex(),
        "tracked": _jsonable(session.duel.state),
        "pending": (
            {
                "msg_type": pending.msg_type,
                "msg_name": pending.msg_name,
                "player": pending.player,
                "parsed": _jsonable(pending.parsed),
            }
            if pending is not None
            else None
        ),
        "engine_seeds": list(session.engine_seeds),
        "limitations": {
            "internal_rng_state_exposed": False,
            "effect_registry_exposed": False,
            "native_engine_serialization": False,
        },
    }
    value["state_hash"] = content_hash(value)
    return value
