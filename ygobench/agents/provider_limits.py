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
    if getattr(provider, "name", None) not in {"deepseek", "dashscope"} or getattr(
        provider, "_ygobench_uncapped", False
    ):
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
