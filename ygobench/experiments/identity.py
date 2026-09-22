"""Stable, secret-free identities for duel and probe policies."""

from __future__ import annotations

from typing import Any

from ygobench.experiments.config import stable_id

POLICY_IDENTITY_VERSION = "3.0.0"
TOOL_SCHEMA_VERSION = "full-duel-tools-v4-batch-inspection-cache"
CONTEXT_POLICY = "compact-public-state-with-duel-card-cache-v2"
RETRY_POLICY = {
    "model_action_attempts": 3,
    "provider_attempts": 3,
    "provider_retry_delays_seconds": [10.0, 10.0],
    "model_failure_action": "rated_forfeit",
    "technical_failure_action": "unrated_abort",
}


def policy_descriptor(
    agent_id: str,
    *,
    prompt_hashes: dict[str, str] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact policy configuration without credentials."""
    parts = agent_id.split(":", 2)
    profile = parts[0]
    provider = parts[1] if len(parts) > 1 else "builtin"
    model = parts[2] if len(parts) > 2 else agent_id
    descriptor: dict[str, Any] = {
        "identity_version": POLICY_IDENTITY_VERSION,
        "agent_id": agent_id,
        "profile": profile,
        "provider": provider,
        "model": model,
        "thinking_enabled": False,
        "reasoning_mode": "disabled",
        "token_limit_policy": "omitted_from_request",
        "prompt_hashes": dict(sorted((prompt_hashes or {}).items())),
        "tool_schema_version": TOOL_SCHEMA_VERSION,
        "retry_policy": RETRY_POLICY,
        "context_policy": CONTEXT_POLICY,
    }
    if runtime:
        # Provider log configurations are intended to be credential-free, but
        # allow-list fields here so a future SDK cannot leak a token by adding
        # an unexpected property.
        for key in (
            "name",
            "provider",
            "model",
            "backend",
            "profile",
            "thinking_enabled",
            "thinking_control",
            "reasoning_effort",
            "temperature",
            "max_tokens",
            "token_limit_policy",
        ):
            if key in runtime:
                descriptor[f"runtime_{key}"] = runtime[key]
        if "thinking_enabled" in runtime:
            descriptor["thinking_enabled"] = bool(runtime["thinking_enabled"])
        if runtime.get("reasoning_effort") not in (None, "none", "off", "disabled"):
            descriptor["reasoning_mode"] = str(runtime["reasoning_effort"])
    return descriptor


def policy_id(descriptor: dict[str, Any]) -> str:
    return stable_id("policy", descriptor, length=24)


def model_configuration_id(descriptor: dict[str, Any]) -> str:
    """Cross-experiment model key; excludes task-specific prompt/tool schemas."""
    value = {
        "provider": descriptor.get("runtime_provider", descriptor.get("provider")),
        "model": descriptor.get("runtime_model", descriptor.get("model")),
        "thinking_enabled": descriptor.get("thinking_enabled"),
        "reasoning_mode": "disabled"
        if descriptor.get("reasoning_mode") in (None, "none", "off", "disabled")
        else descriptor.get("reasoning_mode"),
        "temperature": descriptor.get("runtime_temperature"),
        "token_limit_policy": descriptor.get(
            "runtime_token_limit_policy", descriptor.get("token_limit_policy")
        ),
    }
    return stable_id("modelcfg", value, length=24)


def legacy_policy_descriptor(agent_id: str) -> dict[str, Any]:
    """Describe old runs explicitly instead of silently merging identities."""
    descriptor = policy_descriptor(agent_id)
    descriptor["identity_quality"] = "legacy_inferred"
    return descriptor


def outcome_policy_id(outcome: dict[str, Any], seat: int) -> str:
    identities = outcome.get("policy_identities") or []
    if seat < len(identities) and isinstance(identities[seat], dict):
        value = identities[seat].get("policy_id")
        if value:
            return str(value)
    agents = outcome.get("agents") or []
    agent_id = str(agents[seat]) if seat < len(agents) else f"unknown-seat-{seat}"
    return policy_id(legacy_policy_descriptor(agent_id))
