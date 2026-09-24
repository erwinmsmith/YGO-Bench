"""Provider request policies shared by online agents and post-hoc probes."""

from __future__ import annotations

import json
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
                parts.append({"functionCall": {"name": name, "args": arguments}})
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

    declarations: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") or {}
        if not function.get("name"):
            continue
        declarations.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    if declarations:
        payload["tools"] = [{"functionDeclarations": declarations}]
        payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return payload


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
        native = _gagawenai_gemini_payload(raw_messages, tools)
        raw_extra = kwargs.get("extra_body")
        extra = dict(raw_extra) if isinstance(raw_extra, Mapping) else {}
        extra.update(native)
        kwargs["extra_body"] = extra
        # These OpenAI-only controls are rejected or ignored by the gateway.
        kwargs.pop("parallel_tool_calls", None)
        kwargs.pop("reasoning_effort", None)
        response = create(*args, **kwargs)
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
    """Explicitly disable hidden reasoning for every supported probe provider.

    Attribute assignment is insufficient for APIs that enable reasoning when
    the request field is omitted.  Exp3/Exp5 use this helper so every request
    carries the provider-specific off switch at the SDK boundary.
    """

    name = provider_name or getattr(provider, "name", None)
    if name == "deepseek":
        return force_deepseek_thinking_mode(provider, enabled=False)
    if name in {"bailian", "dashscope", "qwen"}:
        return force_qwen_thinking_mode(provider, enabled=False)
    if name in {"openai", "azopenai"}:
        return force_openai_thinking_disabled(provider)
    if name == "gagawenai-gemini":
        raise ValueError(
            "Gagawenai Gemini cannot be used in a formal thinking-disabled run: "
            "the gateway accepted all tested disable fields but still returned "
            "non-zero completion_tokens_details.reasoning_tokens"
        )
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
