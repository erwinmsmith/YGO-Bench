"""Provider request policies shared by online agents and post-hoc probes."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any


def scrub_reasoning_content(value: Any) -> Any:
    """Return a copy with provider chain-of-thought fields removed recursively."""

    if isinstance(value, Mapping):
        return {
            key: scrub_reasoning_content(item)
            for key, item in value.items()
            if key != "reasoning_content"
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [scrub_reasoning_content(item) for item in value]
    return value


_GEMINI_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")


def _normalize_gemini_tool_name(value: Any) -> str | None:
    """Return a Gemini-compatible function name or ``None`` if unusable."""

    if not isinstance(value, str):
        return None
    name = value.strip()
    if not name:
        return None
    if _GEMINI_TOOL_NAME_RE.fullmatch(name):
        return name
    name = re.sub(r"[^A-Za-z0-9_.:-]", "_", name)
    if not name or not re.match(r"^[A-Za-z_]", name):
        name = f"tool_{name}"
    name = name[:128]
    return name if _GEMINI_TOOL_NAME_RE.fullmatch(name) else None


def _normalize_openai_tool_names(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Normalize OpenAI tool declarations and return safe-name aliases."""

    normalized: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    seen: set[str] = set()
    for raw_tool in tools:
        tool = dict(raw_tool)
        raw_function = tool.get("function")
        if isinstance(raw_function, Mapping):
            function = dict(raw_function)
        else:
            function = {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema"),
            }
        safe_name = _normalize_gemini_tool_name(function.get("name"))
        if safe_name is None or safe_name in seen:
            continue
        original_name = function.get("name")
        if original_name != safe_name:
            aliases[safe_name] = str(original_name)
        function["name"] = safe_name
        function["parameters"] = _normalize_gemini_schema(function.get("parameters"))
        tool["function"] = function
        normalized.append(tool)
        seen.add(safe_name)
    return normalized, aliases


def _restore_gemini_tool_names(response: Any, aliases: dict[str, str]) -> None:
    """Restore names changed before sending an OpenAI-compatible request."""

    if not aliases:
        return
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        for call in getattr(message, "tool_calls", []) or []:
            function = getattr(call, "function", None)
            safe_name = getattr(function, "name", None)
            original_name = aliases.get(safe_name)
            if function is not None and original_name is not None:
                try:
                    function.name = original_name
                except Exception:  # noqa: BLE001
                    object.__setattr__(function, "name", original_name)


def _gagawenai_gemini_payload(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> dict[str, Any]:
    """Translate an OpenAI request into the extra Gemini-native fields.

    The Gagawenai Gemini gateway currently validates both the OpenAI
    ``messages`` envelope and Gemini-native ``contents``/``tools`` fields.
    Keep the original OpenAI messages on the request and add a synchronized
    native representation rather than changing the shared upstream provider.
    """

    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []
    tool_names: dict[str, str] = {}

    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role == "system":
            if content:
                system_parts.append({"text": str(content)})
            continue
        if role == "user":
            contents.append(
                {"role": "user", "parts": [{"text": str(content or "")}]}
            )
            continue
        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": str(content)})
            for call in message.get("tool_calls", []) or []:
                function = call.get("function") or {}
                name = str(function.get("name", ""))
                call_id = str(call.get("id", ""))
                raw_arguments = function.get("arguments", "{}")
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except (TypeError, json.JSONDecodeError):
                    arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                if call_id:
                    tool_names[call_id] = name
                parts.append(
                    {
                        "functionCall": {
                            "name": name,
                            "args": arguments,
                        },
                        # The OpenAI-compatible gateway drops Gemini response signatures.
                        # Google documents this placeholder for replayed function calls.
                        "thoughtSignature": "skip_thought_signature_validator",
                    }
                )
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            name = tool_names.get(call_id, "tool_result")
            raw_result = content or ""
            try:
                response = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            except json.JSONDecodeError:
                response = {"result": str(raw_result)}
            if not isinstance(response, Mapping):
                response = {"result": response}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": dict(response),
                            }
                        }
                    ],
                }
            )

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}

    def normalize_schema(value: Any) -> Any:
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            for key, item in value.items():
                if key == "uniqueItems":
                    continue
                if key == "type" and isinstance(item, list):
                    non_null = [schema_type for schema_type in item if schema_type != "null"]
                    if len(non_null) == 1:
                        normalized["type"] = non_null[0]
                        if "null" in item:
                            normalized["nullable"] = True
                        continue
                normalized[key] = normalize_schema(item)
            return normalized
        if isinstance(value, list):
            return [normalize_schema(item) for item in value]
        return value

    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") or {}
        if function:
            name = function.get("name")
            description = function.get("description", "")
            parameters = function.get("parameters")
        else:
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = tool.get("input_schema")
        if not name:
            continue
        declarations.append(
            {
                "name": name,
                "description": description,
                "parameters": normalize_schema(parameters)
                or {"type": "object", "properties": {}},
            }
        )
    if declarations:
        payload["tools"] = [{"functionDeclarations": declarations}]
        payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return payload


def _gemini_messages_with_thought_signatures(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add the documented Gemini signature placeholder to tool-call history."""

    enriched: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            enriched.append(message)
            continue
        updated = dict(message)
        calls: list[dict[str, Any]] = []
        for raw_call in message.get("tool_calls", []) or []:
            call = dict(raw_call)
            extra_content = dict(call.get("extra_content") or {})
            google = dict(extra_content.get("google") or {})
            google.setdefault("thought_signature", "skip_thought_signature_validator")
            extra_content["google"] = google
            call["extra_content"] = extra_content
            calls.append(call)
        updated["tool_calls"] = calls
        enriched.append(updated)
    return enriched


def adapt_gagawenai_gemini(provider: Any) -> Any:
    """Adapt OpenAI SDK requests to Gagawenai's Gemini gateway contract."""

    if getattr(provider, "_ygobench_gagawenai_gemini", False):
        return provider
    completions = provider._client.chat.completions
    create = completions.create

    @wraps(create)
    def create_with_gemini_payload(*args: Any, **kwargs: Any) -> Any:
        raw_messages = kwargs.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError("Gagawenai Gemini requests require an OpenAI messages list")
        raw_tools = kwargs.pop("tools", None)
        tools = raw_tools if isinstance(raw_tools, list) else None
        tools, tool_name_aliases = _normalize_openai_tool_names(tools or [])
        messages = _gemini_messages_with_thought_signatures(raw_messages)
        kwargs["messages"] = messages
        # Gagawenai accepts standard OpenAI tools, but rejects the native
        # Gemini functionDeclarations envelope with a misleading name error.
        native = _gagawenai_gemini_payload(messages, None)
        raw_extra = kwargs.get("extra_body")
        extra = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}
        extra.update(native)
        kwargs["extra_body"] = extra
        if isinstance(raw_tools, list):
            kwargs["tools"] = tools
        # The gateway rejects none/off for Gemini 3.x. Preserve the supported
        # low effort value, while dropping other OpenAI-only controls.
        kwargs.pop("parallel_tool_calls", None)
        requested_reasoning_effort = kwargs.pop("reasoning_effort", None)
        if requested_reasoning_effort == "low":
            kwargs["reasoning_effort"] = "low"
        response = create(*args, **kwargs)
        _restore_gemini_tool_names(response, tool_name_aliases)
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if usage is not None and reasoning_tokens is not None:
            try:
                usage.reasoning_tokens = reasoning_tokens
            except Exception:  # noqa: BLE001
                object.__setattr__(usage, "reasoning_tokens", reasoning_tokens)
        return response

    completions.create = create_with_gemini_payload
    provider.name = "gagawenai-gemini"
    provider._ygobench_gagawenai_gemini = True
    provider._ygobench_thinking_control = "unsupported-by-gateway"
    provider.thinking_enabled = True
    return provider


def force_gagawenai_gemini_thinking_low(provider: Any) -> Any:
    """Force Gemini 3.x Flash to use the gateway's lowest thinking level.

    Gemini 3.7 Flash does not support a strict none/off setting.
    Gagawenai exposes the OpenAI-compatible reasoning_effort field, so
    keep the request at low even when the caller uses the formal duel
    default of thinking_enabled=False for other providers.
    """

    provider.thinking_enabled = True
    provider.reasoning_effort = "low"
    provider._ygobench_thinking_control = "gagawenai-gemini.reasoning_effort"
    if getattr(provider, "_ygobench_gemini_low_thinking", False):
        return provider

    completions = provider._client.chat.completions
    create = completions.create

    @wraps(create)
    def create_with_low_gemini_thinking(*args: Any, **kwargs: Any) -> Any:
        kwargs["reasoning_effort"] = "low"
        return create(*args, **kwargs)

    completions.create = create_with_low_gemini_thinking
    provider._ygobench_gemini_low_thinking = True
    return provider


def omit_reasoning_model_token_limit(provider: Any) -> Any:
    """Make supported reasoning-model requests omit ``max_tokens``.

    DashScope Qwen and DeepSeek both permit long reasoning/tool-use turns.
    The vendored providers supply a constructor default even when YGO-Bench
    intentionally has no cap, so remove the wire field at the SDK boundary.
    """
    if getattr(provider, "name", None) not in {"deepseek", "dashscope"}:
        return provider
    return omit_provider_token_limit(provider)


def omit_provider_token_limit(provider: Any) -> Any:
    """Remove ``max_tokens`` for an explicitly selected compatible provider."""

    if getattr(provider, "_ygobench_uncapped", False):
        return provider
    try:
        completions = provider._client.chat.completions
    except AttributeError as exc:
        raise ValueError(
            f"provider {getattr(provider, 'name', 'unknown')!r} cannot omit its token limit"
        ) from exc
    create = completions.create

    @wraps(create)
    def create_without_token_limit(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("max_tokens", None)
        return create(*args, **kwargs)

    completions.create = create_without_token_limit
    provider.max_tokens = None
    provider._ygobench_uncapped = True
    return provider


def omit_deepseek_token_limit(provider: Any) -> Any:
    """Backward-compatible alias for the former DeepSeek-only helper."""

    return omit_reasoning_model_token_limit(provider)


def force_deepseek_thinking_mode(provider: Any, *, enabled: bool) -> Any:
    """Send an explicit DeepSeek thinking toggle on every API request.

    DeepSeek V4 enables thinking by default when the request omits the
    ``thinking`` field.  Merely clearing ``reasoning_effort`` therefore does
    not disable hidden reasoning.  Inject the requested mode at the SDK
    boundary while preserving unrelated ``extra_body`` fields.
    """

    if getattr(provider, "name", None) != "deepseek":
        return provider

    provider._ygobench_thinking_mode = "enabled" if enabled else "disabled"
    provider._ygobench_thinking_control = "deepseek.extra_body.thinking.type"
    provider.thinking_enabled = enabled
    if not enabled:
        provider.reasoning_effort = None

    if getattr(provider, "_ygobench_thinking_toggle_wrapped", False):
        return provider

    completions = provider._client.chat.completions
    create = completions.create

    @wraps(create)
    def create_with_explicit_thinking_mode(*args: Any, **kwargs: Any) -> Any:
        raw_extra_body = kwargs.get("extra_body")
        extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, Mapping) else {}
        raw_thinking = extra_body.get("thinking")
        thinking = dict(raw_thinking) if isinstance(raw_thinking, Mapping) else {}
        thinking["type"] = provider._ygobench_thinking_mode
        extra_body["thinking"] = thinking
        kwargs["extra_body"] = extra_body
        return create(*args, **kwargs)

    completions.create = create_with_explicit_thinking_mode
    provider._ygobench_thinking_toggle_wrapped = True
    return provider


def force_qwen_thinking_mode(provider: Any, *, enabled: bool) -> Any:
    """Send DashScope's explicit ``enable_thinking`` flag on every request."""

    provider._ygobench_qwen_thinking_enabled = enabled
    provider._ygobench_thinking_control = "dashscope.extra_body.enable_thinking"
    provider.thinking_enabled = enabled
    if not enabled and hasattr(provider, "reasoning_effort"):
        provider.reasoning_effort = None
    if getattr(provider, "_ygobench_qwen_thinking_toggle_wrapped", False):
        return provider

    completions = provider._client.chat.completions
    create = completions.create

    @wraps(create)
    def create_with_explicit_qwen_thinking(*args: Any, **kwargs: Any) -> Any:
        raw_extra_body = kwargs.get("extra_body")
        extra_body = dict(raw_extra_body) if isinstance(raw_extra_body, Mapping) else {}
        extra_body["enable_thinking"] = provider._ygobench_qwen_thinking_enabled
        kwargs["extra_body"] = extra_body
        return create(*args, **kwargs)

    completions.create = create_with_explicit_qwen_thinking
    provider._ygobench_qwen_thinking_toggle_wrapped = True
    return provider


def force_openai_thinking_disabled(provider: Any) -> Any:
    """Send the OpenAI-compatible non-reasoning setting on every request.

    OpenAI-compatible GPT endpoints may enable reasoning when the field is
    omitted.  Setting an attribute on the provider is only useful for logging;
    this wrapper changes the actual wire request.
    """

    provider.thinking_enabled = False
    provider.reasoning_effort = "none"
    provider._ygobench_thinking_control = "openai.reasoning_effort"
    if getattr(provider, "_ygobench_openai_thinking_disabled", False):
        return provider

    completions = provider._client.chat.completions
    create = completions.create

    @wraps(create)
    def create_without_reasoning(*args: Any, **kwargs: Any) -> Any:
        kwargs["reasoning_effort"] = "none"
        return create(*args, **kwargs)

    completions.create = create_without_reasoning
    provider._ygobench_openai_thinking_disabled = True
    return provider


def force_provider_thinking_disabled(
    provider: Any, *, provider_name: str | None = None
) -> Any:
    """Apply the provider-specific thinking policy at the SDK boundary.

    Attribute assignment is insufficient for APIs that enable reasoning when
    the request field is omitted.  Providers without a verified off switch.
    are left at their gateway default and expose that fact in metadata.
    """

    name = provider_name or getattr(provider, "name", None)
    if name == "deepseek":
        return force_deepseek_thinking_mode(provider, enabled=False)
    if name in {"bailian", "dashscope", "qwen"}:
        return force_qwen_thinking_mode(provider, enabled=False)
    if name in {"openai", "azopenai"}:
        return force_openai_thinking_disabled(provider)
    if name == "gagawenai-gemini":
        # The gateway currently ignores all tested disable fields.  Do not
        # block a diagnostic duel; reasoning usage remains visible in traces.
        return provider
    raise ValueError(
        f"provider {name!r} has no explicit thinking-disable adapter; formal runs fail closed"
    )


def force_single_tool_call(provider: Any) -> Any:
    """Inject the OpenAI-compatible flag that disables parallel tool calls.

    This wrapper deliberately lives in YGO-Bench rather than modifying the
    vendored provider submodule, so experiment provenance remains reproducible
    from this repository alone.
    """

    if getattr(provider, "_ygobench_single_tool_call", False):
        return provider
    completions = provider._client.chat.completions
    create = completions.create

    @wraps(create)
    def create_with_single_tool_call(*args: Any, **kwargs: Any) -> Any:
        kwargs["parallel_tool_calls"] = False
        return create(*args, **kwargs)

    completions.create = create_with_single_tool_call
    provider._ygobench_single_tool_call = True
    return provider
def _normalize_gemini_schema(value: Any) -> Any:
    """Make JSON Schema compatible with the Gagawenai Gemini gateway."""
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "uniqueItems":
                continue
            if key == "enum" and isinstance(item, list):
                # The gateway rejects numeric enum literals.
                if any(not isinstance(enum_value, str) for enum_value in item):
                    continue
            if key == "type" and isinstance(item, list):
                non_null = [schema_type for schema_type in item if schema_type != "null"]
                if len(non_null) == 1:
                    normalized["type"] = non_null[0]
                    if "null" in item:
                        normalized["nullable"] = True
                    continue
            normalized[key] = _normalize_gemini_schema(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_gemini_schema(item) for item in value]
    return value
