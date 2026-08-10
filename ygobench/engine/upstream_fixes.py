"""Compatibility fixes for the pinned upstream YGO engine modules.

The upstream observation renderer encodes absolute seat 0 as ``you`` and
absolute seat 1 as ``opponent`` inside decision card labels.  That is only
correct from player 0's perspective.  The engine can also expose a negative
terminal LP value as its wrapped uint32 representation.  Apply both fixes at
the project boundary so they are versioned with YGO-Bench without carrying an
unpublishable dirty submodule.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

_UINT32_MODULUS = 1 << 32
_UINT32_SIGN_BIT = 1 << 31


def normalize_lp(value: Any) -> Any:
    """Return game-rule LP, clamping a wrapped negative uint32 value to zero."""

    if not isinstance(value, int) or isinstance(value, bool):
        return value
    if value < 0:
        return 0
    if _UINT32_SIGN_BIT <= value < _UINT32_MODULUS:
        return 0
    return value


def _normalize_lp_record(record: Any) -> None:
    if not isinstance(record, dict) or "lp" not in record:
        return
    raw_lp = record["lp"]
    normalized = normalize_lp(raw_lp)
    if normalized != raw_lp:
        record["lp_raw_u32"] = raw_lp
        record["lp"] = normalized


def relativize_controller_labels(value: Any, *, perspective: int) -> Any:
    """Convert upstream's absolute-seat controller labels to relative labels."""

    if perspective not in (0, 1):
        raise ValueError(f"perspective must be 0 or 1, got {perspective}")
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "controller" and item in ("you", "opponent"):
                absolute_controller = 0 if item == "you" else 1
                value[key] = "you" if absolute_controller == perspective else "opponent"
            else:
                relativize_controller_labels(item, perspective=perspective)
    elif isinstance(value, list):
        for item in value:
            relativize_controller_labels(item, perspective=perspective)
    return value


def apply_upstream_fixes(core_module: Any, state_module: Any) -> None:
    """Patch the pinned modules once per process with perspective/LP fixes."""

    if getattr(state_module, "_ygobench_perspective_fix_applied", False):
        return

    original_build_decision = state_module.build_decision

    @wraps(original_build_decision)
    def build_decision(decision: Any, card_db: Any, *, perspective: int | None = None) -> dict:
        rendered = original_build_decision(decision, card_db)
        acting_perspective = int(decision.player) if perspective is None else perspective
        return relativize_controller_labels(rendered, perspective=acting_perspective)

    original_build_state = state_module.build_state

    @wraps(original_build_state)
    def build_state(
        harness: Any,
        card_db: Any,
        *,
        perspective: int = 0,
        include_decision: bool = True,
        events: list[dict] | None = None,
    ) -> dict:
        rendered = original_build_state(
            harness,
            card_db,
            perspective=perspective,
            include_decision=include_decision,
            events=events,
        )
        if include_decision and harness.pending is not None:
            rendered["decision"] = build_decision(
                harness.pending,
                card_db,
                perspective=perspective,
            )
        return rendered

    original_query_field = core_module.OCGEngine.query_field

    @wraps(original_query_field)
    def query_field(engine: Any) -> dict:
        field = original_query_field(engine)
        for player in field.get("players", []):
            _normalize_lp_record(player)
        return field

    original_parse_message = core_module.OCGEngine._parse_single_message

    @wraps(original_parse_message)
    def parse_single_message(engine: Any, msg_type: int, reader: Any) -> Any:
        parsed = original_parse_message(engine, msg_type, reader)
        if msg_type == core_module.MSG_LPUPDATE:
            _normalize_lp_record(parsed)
        return parsed

    state_module.build_decision = build_decision
    state_module.build_state = build_state
    state_module._ygobench_perspective_fix_applied = True
    core_module.OCGEngine.query_field = query_field
    core_module.OCGEngine._parse_single_message = parse_single_message

