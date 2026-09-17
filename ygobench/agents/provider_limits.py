"""Provider request policies shared by online agents and post-hoc probes."""

from __future__ import annotations

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
    completions = provider._client.chat.completions
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
    return provider


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
