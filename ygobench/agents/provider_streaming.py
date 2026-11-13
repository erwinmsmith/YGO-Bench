"""CJRouter SSE transport behind the existing complete-response provider interface."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import wraps
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit


class IncompleteStreamError(RuntimeError):
    """A stream cannot safely supply a complete, accounted model turn."""


def cjrouter_streaming_configuration(
    provider_name: str, base_url: str | None
) -> dict[str, Any]:
    """Select the existing external CJRouter profile without affecting other endpoints."""
    url = urlsplit(base_url or "")
    if provider_name != "external" or url.scheme != "https" or url.hostname != "cjrouter.cn":
        return {}
    return {
        "response_transport": "cjrouter-sse-v1",
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _collect_completion(stream: Any) -> Any:
    # Keep this adapter in the main repository, not the pinned vendor submodule.
    from openai.types.chat import ChatCompletion

    metadata: dict[str, Any] = {}
    parts: dict[str, list[str]] = {"content": [], "reasoning_content": [], "refusal": []}
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    usage = None
    try:
        for chunk in stream:
            data = chunk.model_dump(exclude_none=True)
            for key in ("id", "model", "created", "system_fingerprint", "service_tier"):
                if key in data:
                    metadata[key] = data[key]
            for choice in data.get("choices") or []:
                if choice.get("index") != 0:
                    raise IncompleteStreamError("Expected a single streamed choice")
                delta = choice.get("delta") or {}
                if finish is not None and delta:
                    raise IncompleteStreamError("Received model output after finish_reason")
                for key, fragments in parts.items():
                    value = delta.get(key)
                    if value is not None:
                        if not isinstance(value, str):
                            raise IncompleteStreamError(f"Invalid streamed {key} fragment")
                        fragments.append(value)
                for item in delta.get("tool_calls") or []:
                    index = item.get("index")
                    if not isinstance(index, int) or index < 0:
                        raise IncompleteStreamError("Tool-call fragment has no valid index")
                    if item.get("type", "function") != "function":
                        raise IncompleteStreamError("Unsupported streamed tool type")
                    call = calls.setdefault(index, {
                        "id": [], "name": [], "arguments": [],
                    })
                    fields = {"id": item.get("id"), **(item.get("function") or {})}
                    for key in call:
                        value = fields.get(key)
                        if value is not None:
                            if not isinstance(value, str):
                                raise IncompleteStreamError("Invalid tool-call fragment")
                            call[key].append(value)
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
            # A pre-finish usage update may be partial. Only final usage is authoritative.
            if finish is not None and data.get("usage") is not None:
                usage = data["usage"]

        if finish not in {"stop", "tool_calls"}:
            raise IncompleteStreamError("Stream missing a successful finish_reason")
        if not isinstance(usage, dict) or any(
            not isinstance(usage.get(key), int) or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise IncompleteStreamError("Stream missing final token usage; not counted as zero")

        tool_calls = []
        for index in sorted(calls):
            call = {key: "".join(value) for key, value in calls[index].items()}
            if not call["id"] or not call["name"]:
                raise IncompleteStreamError("Streamed tool call is missing its ID or name")
            try:
                arguments = json.loads(call["arguments"])
            except (ValueError, TypeError):
                raise IncompleteStreamError("Streamed tool arguments are incomplete JSON") from None
            if not isinstance(arguments, dict):
                raise IncompleteStreamError("Streamed tool arguments must be a JSON object")
            tool_calls.append({
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            })
        if finish == "tool_calls" and not tool_calls:
            raise IncompleteStreamError("Stream finished tool_calls without any tool calls")
        if len({call["id"] for call in tool_calls}) != len(tool_calls):
            raise IncompleteStreamError("Stream contains duplicate tool-call IDs")

        # The vendored provider reads flat usage fields; keep the original details too.
        for field, details in (
            ("reasoning_tokens", "completion_tokens_details"),
            ("cached_tokens", "prompt_tokens_details"),
        ):
            value = (usage.get(details) or {}).get(field)
            if value is not None:
                usage.setdefault(field, value)
        message = {key: "".join(value) for key, value in parts.items() if value}
        response = ChatCompletion.model_validate({
            **metadata,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "finish_reason": finish,
                "message": {
                    **message,
                    "role": "assistant",
                    "content": message.get("content") or None,
                    "tool_calls": tool_calls or None,
                },
            }],
            "usage": usage,
        })
        headers = getattr(getattr(stream, "response", None), "headers", {})
        # Match the existing recovery header allow-list, never retaining credentials.
        object.__setattr__(response, "_response", SimpleNamespace(headers={
            key: value for key, value in headers.items()
            if key.lower() in {"x-request-id", "x-oneapi-request-id", "request-id"}
        }))
        return response
    finally:
        # Also close on timeout, SSE error, malformed output, and process interruption.
        stream.close()


def configure_cjrouter_streaming(
    provider: Any, *, provider_name: str, base_url: str | None
) -> Any:
    """Aggregate SSE before the vendored provider parses/saves a complete model turn."""
    config = cjrouter_streaming_configuration(provider_name, base_url)
    if not config or getattr(provider, "_ygobench_cjrouter_streaming", False):
        return provider
    completions = provider._client.chat.completions
    create = completions.create
    log_config = provider.provider_config_for_log

    @wraps(create)
    def create_streamed(*args: Any, **kwargs: Any) -> Any:
        extra = dict(kwargs.get("extra_body") or {})
        # extra_body can override SDK keyword arguments; enforce one transport.
        extra.pop("stream", None)
        extra.pop("stream_options", None)
        if extra:
            kwargs["extra_body"] = extra
        else:
            kwargs.pop("extra_body", None)
        kwargs["stream"] = True
        kwargs["stream_options"] = {
            **(kwargs.get("stream_options") or {}), "include_usage": True,
        }
        return _collect_completion(create(*args, **kwargs))

    @wraps(log_config)
    def streaming_log_config() -> dict[str, Any]:
        return {**log_config(), **deepcopy(config)}

    completions.create = create_streamed
    provider.provider_config_for_log = streaming_log_config
    provider._ygobench_cjrouter_streaming = True
    return provider
