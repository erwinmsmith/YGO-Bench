"""Tool-calling LLM policy for complete Yu-Gi-Oh! duels."""

from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from ygobench.agents.base import BaseAgent
from ygobench.agents.provider_limits import (
    force_single_tool_call,
    omit_reasoning_model_token_limit,
    scrub_reasoning_content,
)
from ygobench.config import ModelConfig
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.engine.upstream import UpstreamLayout
from ygobench.engine.visibility import sanitize_events_for_player

PROMPT_ROOT = Path(__file__).parent / "prompts"
# These are total attempts, not retries after an initial request.  The duel
# runner treats exhaustion as a model-action failure and awards the game to
# the opponent; it must never silently substitute a game action.
MAX_PROVIDER_CONNECTION_ATTEMPTS = 3
CONNECTION_RETRY_DELAYS_SECONDS = (10.0, 10.0)
MAX_MODEL_ACTION_ATTEMPTS = 3


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


def exact_legal_actions_packet(decision: DecisionRequest) -> str:
    """Render an authoritative, copyable action list for exact decisions."""

    actions = [
        {"name": action.tool, "arguments": action.arguments}
        for action in decision.legal_actions
    ]
    return (
        "\n\n## Authoritative exact legal actions\n"
        "The list below is complete for this engine decision. Choose exactly one "
        "entry and reproduce its tool name and arguments verbatim, including "
        "argument keys and index order. Do not construct a new argument object.\n\n"
        "```json\n"
        + json.dumps(actions, ensure_ascii=False, indent=2, default=str)
        + "\n```"
    )


class ProviderProtocolError(RuntimeError):
    """Raised locally when a tool-call transcript is not API-valid."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = diagnostics
        super().__init__("invalid_tool_message_transcript: " + json.dumps(diagnostics, sort_keys=True))


class ProviderCallError(RuntimeError):
    """A provider request failure after all three bounded attempts are exhausted."""

    provider_failure_type = "provider_call"

    def __init__(self, cause: Exception, attempts: list[dict[str, Any]]) -> None:
        self.attempts = attempts
        self.cause_type = type(cause).__name__
        super().__init__(f"{self.cause_type}: provider connection retries exhausted")


def _tool_protocol_diagnostics(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit assistant tool calls and their immediately following results."""

    pending: list[str] = []
    malformed: list[str] = []
    tool_call_ids: list[str] = []
    tool_result_ids: list[str] = []
    tool_names: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            if pending:
                malformed.append("assistant_before_all_tool_results")
            calls = message.get("tool_calls") or []
            ids = [str(call.get("id", "")) for call in calls]
            tool_call_ids.extend(ids)
            tool_names.extend(str(call.get("name", "")) for call in calls)
            if len(ids) != len(set(ids)) or any(not value for value in ids):
                malformed.append("missing_or_duplicate_tool_call_id")
            pending.extend(ids)
        elif role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            tool_result_ids.append(call_id)
            if call_id not in pending:
                malformed.append("orphan_or_duplicate_tool_result")
            else:
                pending.remove(call_id)
        elif pending:
            malformed.append("non_tool_message_before_all_tool_results")
    return {
        "assistant_tool_call_count": len(tool_call_ids),
        "tool_result_count": len(tool_result_ids),
        "tool_names": tool_names,
        "tool_call_ids": tool_call_ids,
        "tool_result_ids": tool_result_ids,
        "unresolved_tool_call_ids": len(pending),
        "protocol_errors": sorted(set(malformed)),
    }


class LLMFullDuelAgent(BaseAgent):
    """Select exactly one ocgcore response tool for each pending decision."""

    def __init__(
        self,
        model: ModelConfig,
        *,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        max_inspections: int = 8,
        max_forced_retries: int = MAX_MODEL_ACTION_ATTEMPTS - 1,
        thinking_enabled: bool = True,
        profile: str = "react",
    ) -> None:
        tools_module, get_provider = _provider_runtime()
        kwargs: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if model.base_url:
            kwargs["base_url"] = model.base_url
        if model.api_key:
            kwargs["api_key"] = model.api_key
        backend = model.backend or model.provider
        if backend == "openai" and model.base_url:
            kwargs["extra_body"] = {"enable_thinking": thinking_enabled}
        self._provider = get_provider(backend, model.model, **kwargs)
        if backend in {"deepseek", "dashscope"} and max_tokens is None:
            omit_reasoning_model_token_limit(self._provider)
        # DashScope's documented compatible-mode example does not expose this
        # optional OpenAI flag.  Prompt/schema validation enforce one action,
        # while avoiding a provider-specific unsupported request parameter.
        if backend != "dashscope":
            force_single_tool_call(self._provider)
        if backend in {"deepseek", "dashscope"} and not thinking_enabled:
            self._provider.reasoning_effort = None
            self._provider.thinking_enabled = False
        self._tool_defs = {tool["name"]: tool for tool in tools_module.TOOLS}
        self._max_inspections = max_inspections
        self._max_forced_retries = max_forced_retries
        self._max_provider_connection_attempts = MAX_PROVIDER_CONNECTION_ATTEMPTS
        self._system_prompt = (PROMPT_ROOT / "full_duel_system.md").read_text()
        self._observation_template = (PROMPT_ROOT / "full_duel_observation.md").read_text()
        self.name = f"{profile}:{model.provider}:{model.model}"
        self.provider_config = self._provider.provider_config_for_log()
        self.provider_config["profile"] = profile
        self.provider_config["thinking_enabled"] = thinking_enabled
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

    def _respond_once(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        requests: list[dict[str, Any]],
        transport_attempts: list[dict[str, Any]],
    ) -> Any:
        """Validate the transcript and make at most three provider requests.

        The provider SDK may label failures as connection, timeout, status, or
        decoding errors.  All occur before a game action exists, so they share
        the same bounded retry policy and are preserved in the evidence trace.
        """

        diagnostics = _tool_protocol_diagnostics(messages)
        if diagnostics["unresolved_tool_call_ids"] or diagnostics["protocol_errors"]:
            raise ProviderProtocolError(diagnostics)
        requests.append(
            scrub_reasoning_content(
                {"system": system, "messages": deepcopy(messages), "tools": deepcopy(tools)}
            )
        )
        max_connection_attempts = getattr(
            self, "_max_provider_connection_attempts", MAX_PROVIDER_CONNECTION_ATTEMPTS
        )
        for attempt in range(max_connection_attempts):
            try:
                turn = self._provider.respond(system=system, messages=messages, tools=tools)
                self._accumulate(turn.usage, turn.wallclock_seconds)
                transport_attempts.append(
                    {"attempt": attempt + 1, "outcome": "success", "error_type": None}
                )
                return turn
            except Exception as exc:  # noqa: BLE001
                transport_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "outcome": "error",
                        "error_type": type(exc).__name__,
                        "retryable": True,
                    }
                )
                if attempt + 1 >= max_connection_attempts:
                    raise ProviderCallError(exc, transport_attempts) from exc
                time.sleep(CONNECTION_RETRY_DELAYS_SECONDS[attempt])

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
            diagnostics = {
                "protocol_errors": ["missing_response_tool_schema"],
                "responder": responder,
                "model_action_attempts": 0,
            }
            self.last_trace = {"fallback": False, "error": diagnostics}
            raise ProviderProtocolError(diagnostics)

        prompt = self._observation_template.replace(
            "{{STATE_JSON}}", json.dumps(state, ensure_ascii=False, indent=2, default=str)
        ).replace("{{REQUIRED_RESPONDER}}", responder)
        if decision.legal_actions_complete:
            prompt += exact_legal_actions_packet(decision)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        available_tools = [self._tool_defs["inspect_card"], response_tool]
        active_system_prompt = self._system_prompt
        inspections = 0
        forced_retries = 0
        traces: list[dict[str, Any]] = []
        requests: list[dict[str, Any]] = []
        transport_attempts: list[dict[str, Any]] = []
        protocol_failures: list[dict[str, Any]] = []

        while inspections <= self._max_inspections:
            turn = self._respond_once(
                system=active_system_prompt,
                messages=messages,
                tools=available_tools,
                requests=requests,
                transport_attempts=transport_attempts,
            )
            trace = {
                "text": turn.text,
                "tool_calls": [
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
                "stop_reason": turn.stop_reason,
                "usage": turn.usage,
                "elapsed_seconds": turn.wallclock_seconds,
                # Keep provider reasoning only in the live conversation below;
                # never persist it in evidence logs.
                "provider_data": scrub_reasoning_content(turn.provider_data),
            }
            traces.append(trace)
            if len(turn.tool_calls) > 1:
                protocol_failures.append(
                    {
                        "attempt": len(traces),
                        "protocol_errors": ["multiple_tool_calls_in_single_model_turn"],
                        "assistant_tool_call_count": len(turn.tool_calls),
                        "tool_names": [call.name for call in turn.tool_calls],
                        "tool_call_ids": [call.id for call in turn.tool_calls],
                    }
                )
                action_call = None
            else:
                action_call = next(
                    (call for call in turn.tool_calls if call.name == responder), None
                )
            if action_call is not None:
                self.last_trace = {
                    "requests": requests,
                    "turns": traces,
                    "transport_attempts": transport_attempts,
                    "fallback": False,
                }
                return ActionChoice(
                    tool=action_call.name,
                    arguments=action_call.arguments,
                    label=turn.text.strip() or action_call.name,
                )

            inspect_calls = [call for call in turn.tool_calls if call.name == "inspect_card"]
            if len(turn.tool_calls) == 1 and inspect_calls and inspections < self._max_inspections:
                messages.append(
                    {
                        "role": "assistant",
                        "text": turn.text,
                        "tool_calls": trace["tool_calls"],
                        "provider_data": turn.provider_data,
                    }
                )
                inspect_call = inspect_calls[0]
                card_code = int(inspect_call.arguments.get("card_code", 0))
                card_info = _find_card(decision.observation, card_code)
                inspections += 1
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
        diagnostics = {
            "protocol_errors": ["missing_required_responder_tool_call"],
            "model_action_attempts": forced_retries + 1,
            "forced_corrections": forced_retries,
            "attempt_failures": protocol_failures,
        }
        self.last_trace = {
            "requests": requests,
            "turns": traces,
            "transport_attempts": transport_attempts,
            "fallback": False,
            "terminal_model_failure": True,
            "error": diagnostics,
        }
        raise ProviderProtocolError(diagnostics)

    def correct_invalid_action(self, decision: DecisionRequest) -> tuple[ActionChoice | None, dict[str, Any]]:
        """Request one replacement from the authoritative legal action list.

        This is deliberately a narrow correction call: it cannot inspect cards
        or select a different responder, and is used only after an attempted
        action has failed the pre-engine exact-legal gate.
        """

        responder = decision.decision_type
        response_tool = self._tool_defs.get(responder)
        if response_tool is None:
            return None, {"error": f"No tool schema for responder {responder}"}
        packet = {
            "required_responder": responder,
            "legal_responses": [
                {"name": action.tool, "arguments": action.arguments}
                for action in decision.legal_actions
            ],
        }
        requests: list[dict[str, Any]] = []
        transport_attempts: list[dict[str, Any]] = []
        turn = self._respond_once(
            system=(
                "Return exactly one tool call. The legal_responses list is authoritative; "
                "choose one entry verbatim and do not provide prose."
            ),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(packet, ensure_ascii=False, default=str),
                }
            ],
            tools=[response_tool],
            requests=requests,
            transport_attempts=transport_attempts,
        )
        trace = {
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in turn.tool_calls
            ],
            "stop_reason": turn.stop_reason,
            "usage": turn.usage,
            "elapsed_seconds": turn.wallclock_seconds,
            "provider_data": scrub_reasoning_content(turn.provider_data),
            "transport_attempts": transport_attempts,
        }
        if len(turn.tool_calls) != 1:
            trace["protocol_errors"] = ["multiple_tool_calls_in_single_model_turn"]
            return None, trace
        call = next((item for item in turn.tool_calls if item.name == responder), None)
        if call is None:
            trace["protocol_errors"] = ["missing_required_responder_tool_call"]
            return None, trace
        return ActionChoice(tool=call.name, arguments=call.arguments, label=call.name), trace


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
