"""Tool-calling LLM policy for complete Yu-Gi-Oh! duels."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ygobench.agents.base import BaseAgent
from ygobench.config import ModelConfig
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.engine.upstream import UpstreamLayout
from ygobench.engine.visibility import sanitize_events_for_player

PROMPT_ROOT = Path(__file__).parent / "prompts"


def _provider_runtime():
    layout = UpstreamLayout()
    layout.require_runtime()
    source = str(layout.root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from engine import tools  # type: ignore[import-not-found]
    from providers import get_provider  # type: ignore[import-not-found]

    return tools, get_provider


def _strip_card(card: Any) -> Any:
    if not isinstance(card, dict):
        return card
    keep = (
        "zone_index",
        "empty",
        "face_down",
        "code",
        "name",
        "position",
        "attack",
        "defense",
        "level",
        "rank",
        "link",
        "attribute",
        "race",
        "type_flags",
        "location",
        "sequence",
    )
    return {key: card[key] for key in keep if key in card}


def compact_prompt_state(state: dict[str, Any]) -> dict[str, Any]:
    """Remove repeated oracle text while retaining all tactical facts."""

    result = {
        key: state.get(key)
        for key in (
            "perspective_player",
            "phase",
            "turn_player",
            "turn",
            "game_over",
            "winner",
            "win_reason",
            "chain",
            "decision",
            "events_since_last_decision",
            "recent_actions",
        )
        if key in state
    }
    perspective = int(state.get("perspective_player", 0) or 0)
    if "events_since_last_decision" in result:
        result["events_since_last_decision"] = sanitize_events_for_player(
            result["events_since_last_decision"],
            perspective=perspective,
        )
    for side_name in ("you", "opponent"):
        side = state.get(side_name) or {}
        compact = {
            key: side.get(key)
            for key in (
                "lp",
                "deck_count",
                "hand_count",
                "grave_count",
                "banished_count",
                "extra_deck_count",
            )
        }
        for zone in (
            "monster_zone",
            "spell_trap_zone",
            "hand",
            "graveyard",
            "banished",
            "extra_deck",
            "pendulum_zone",
        ):
            compact[zone] = [_strip_card(card) for card in side.get(zone, [])]
        compact["field_zone"] = _strip_card(side.get("field_zone"))
        result[side_name] = compact
    return result


class LLMFullDuelAgent(BaseAgent):
    """Select exactly one ocgcore response tool for each pending decision."""

    def __init__(
        self,
        model: ModelConfig,
        *,
        max_tokens: int = 32768,
        temperature: float = 0.0,
        max_inspections: int = 8,
        max_forced_retries: int = 2,
        thinking_enabled: bool = True,
        profile: str = "react",
    ) -> None:
        tools_module, get_provider = _provider_runtime()
        kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
        if model.base_url:
            kwargs["base_url"] = model.base_url
        self._provider = get_provider(model.provider, model.model, **kwargs)
        if model.provider == "deepseek" and not thinking_enabled:
            self._provider.reasoning_effort = None
            self._provider.thinking_enabled = False
        self._tool_defs = {tool["name"]: tool for tool in tools_module.TOOLS}
        self._max_inspections = max_inspections
        self._max_forced_retries = max_forced_retries
        self._system_prompt = (PROMPT_ROOT / "full_duel_system.md").read_text()
        self._observation_template = (PROMPT_ROOT / "full_duel_observation.md").read_text()
        self.name = f"{profile}:{model.provider}:{model.model}"
        self.provider_config = self._provider.provider_config_for_log()
        self.usage: dict[str, float] = {}
        self.model_calls = 0
        self.invalid_outputs = 0
        self.last_trace: dict[str, Any] = {}

    def reset(self) -> None:
        self.usage = {}
        self.model_calls = 0
        self.invalid_outputs = 0
        self.last_trace = {}

    def _accumulate(self, usage: dict[str, Any], elapsed: float) -> None:
        self.model_calls += 1
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                self.usage[key] = self.usage.get(key, 0.0) + float(value)
        self.usage["wallclock_seconds"] = self.usage.get("wallclock_seconds", 0.0) + elapsed

    def predict(self, decision: DecisionRequest) -> ActionChoice:
        if len(decision.legal_actions) == 1:
            action = decision.legal_actions[0]
            self.last_trace = {
                "automatic": True,
                "reason": "only one legal engine response",
                "selected": {"name": action.tool, "arguments": action.arguments},
            }
            return ActionChoice(
                tool=action.tool,
                arguments=action.arguments,
                label="automatic / only legal response",
            )
        state = compact_prompt_state(decision.observation)
        responder = str(state.get("decision", {}).get("responder", decision.decision_type))
        response_tool = self._tool_defs.get(responder)
        if response_tool is None:
            self.invalid_outputs += 1
            self.last_trace = {"error": f"No tool schema for responder {responder}"}
            return decision.legal_actions[0]

        prompt = self._observation_template.replace(
            "{{STATE_JSON}}", json.dumps(state, ensure_ascii=False, indent=2, default=str)
        ).replace("{{REQUIRED_RESPONDER}}", responder)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        available_tools = [self._tool_defs["inspect_card"], response_tool]
        active_system_prompt = self._system_prompt
        inspections = 0
        forced_retries = 0
        traces: list[dict[str, Any]] = []

        while inspections <= self._max_inspections:
            turn = self._provider.respond(
                system=active_system_prompt,
                messages=messages,
                tools=available_tools,
            )
            self._accumulate(turn.usage, turn.wallclock_seconds)
            trace = {
                "text": turn.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
                "stop_reason": turn.stop_reason,
                "usage": turn.usage,
                "elapsed_seconds": turn.wallclock_seconds,
                "provider_data": turn.provider_data,
            }
            traces.append(trace)
            action_call = next((call for call in turn.tool_calls if call.name == responder), None)
            if action_call is not None:
                self.last_trace = {"turns": traces, "fallback": False}
                return ActionChoice(
                    tool=action_call.name,
                    arguments=action_call.arguments,
                    label=turn.text.strip() or action_call.name,
                )

            inspect_calls = [call for call in turn.tool_calls if call.name == "inspect_card"]
            if inspect_calls and inspections < self._max_inspections:
                messages.append(
                    {
                        "role": "assistant",
                        "text": turn.text,
                        "tool_calls": trace["tool_calls"],
                        "provider_data": turn.provider_data,
                    }
                )
                for inspect_call in inspect_calls:
                    if inspections < self._max_inspections:
                        card_code = int(inspect_call.arguments.get("card_code", 0))
                        card_info = _find_card(decision.observation, card_code)
                        inspections += 1
                    else:
                        card_info = {
                            "name": None,
                            "note": "Per-decision card inspection budget exhausted.",
                        }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": inspect_call.id,
                            "content": json.dumps(card_info, ensure_ascii=False, default=str),
                            "is_error": card_info.get("name") is None,
                        }
                    )
                if inspections >= self._max_inspections:
                    available_tools = [response_tool]
                continue

            if forced_retries < self._max_forced_retries:
                forced_retries += 1
                available_tools = [response_tool]
                active_system_prompt = (
                    "You are a Yu-Gi-Oh! engine responder. Return exactly one tool call and "
                    "no prose. The supplied legal responses are authoritative."
                )
                retry_packet = {
                    "phase": state.get("phase"),
                    "turn": state.get("turn"),
                    "chain": state.get("chain"),
                    "you_lp": (state.get("you") or {}).get("lp"),
                    "opponent_lp": (state.get("opponent") or {}).get("lp"),
                    "decision": state.get("decision"),
                    "required_responder": responder,
                    "legal_responses": [
                        {"name": action.tool, "arguments": action.arguments}
                        for action in decision.legal_actions
                    ],
                }
                messages = [
                    {
                        "role": "user",
                        "content": (
                            "The previous full-state attempt did not submit an action. Choose "
                            "one legal response from this correction packet and call the "
                            f"{responder} tool now:\n"
                            + json.dumps(retry_packet, ensure_ascii=False, default=str)
                        ),
                    }
                ]
                continue
            break

        self.invalid_outputs += 1
        self.last_trace = {
            "turns": traces,
            "fallback": True,
            "error": "model did not call required responder",
        }
        return decision.legal_actions[0]


def _find_visible_card(value: Any, code: int) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("code") == code and value.get("name"):
            return value
        for nested in value.values():
            found = _find_visible_card(nested, code)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_visible_card(nested, code)
            if found is not None:
                return found
    return None


def _find_card(value: Any, code: int) -> dict[str, Any]:
    found = _find_visible_card(value, code)
    if found is not None:
        return found
    return {
        "code": code,
        "name": None,
        "note": "Card is not visible in the current observation.",
    }
