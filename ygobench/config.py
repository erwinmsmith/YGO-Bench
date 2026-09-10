"""Project paths and model/provider configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    backend: str | None = None


_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-v4-flash",
    "dashscope": "qwen3.7-flash",
    "openai": "gpt-5",
    "vllm": "local-model",
    "claude-cli": "claude-sonnet-4-6",
}

_PROFILE_DEFAULTS = {
    "bailian": {
        "model_env": "BAILIAN_MODEL",
        "model": "qwen3.7-flash",
        "key_env": "BAILIAN_API_KEY",
        "fallback_key_env": "DASHSCOPE_API_KEY",
        "base_env": "BAILIAN_BASE_URL",
        "fallback_base_env": "DASHSCOPE_BASE_URL",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "external": {
        "model_env": "EXTERNAL_MODEL",
        "model": "qwen3.5:9b-128k",
        "key_env": "EXTERNAL_API_KEY",
        "base_env": "EXTERNAL_BASE_URL",
        "base_url": "https://api.code-soul.com/v1",
    },
}

_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def default_model_config(provider: str | None = None, model: str | None = None) -> ModelConfig:
    selected_provider = (provider or os.getenv("LLM_PROVIDER", "deepseek")).strip().lower()
    if selected_provider in _PROFILE_DEFAULTS:
        profile = _PROFILE_DEFAULTS[selected_provider]
        selected_model = (
            model or os.getenv(profile["model_env"], "").strip() or profile["model"]
        ).strip()
        api_key = os.getenv(profile["key_env"], "").strip() or None
        fallback_key_env = profile.get("fallback_key_env")
        if not api_key and fallback_key_env:
            api_key = os.getenv(fallback_key_env, "").strip() or None
        base_url = os.getenv(profile["base_env"], "").strip() or profile["base_url"]
        fallback_base_env = profile.get("fallback_base_env")
        if fallback_base_env and not os.getenv(profile["base_env"], "").strip():
            base_url = os.getenv(fallback_base_env, "").strip() or base_url
        return ModelConfig(
            provider=selected_provider,
            model=selected_model,
            base_url=base_url,
            api_key=api_key,
            backend="openai",
        )
    if selected_provider not in _DEFAULT_MODELS:
        known = ", ".join(sorted((*_DEFAULT_MODELS, *_PROFILE_DEFAULTS)))
        raise ValueError(f"Unknown provider {selected_provider!r}; choose one of: {known}")
    # An explicit provider must not inherit an unrelated global LLM_MODEL.
    # For example, `react:dashscope` should resolve to Qwen even when .env
    # keeps DeepSeek as the default for legacy runs.
    inherited_model = os.getenv("LLM_MODEL") if provider is None else None
    selected_model = (model or inherited_model or _DEFAULT_MODELS[selected_provider]).strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    return ModelConfig(
        provider=selected_provider,
        model=selected_model,
        base_url=base_url,
        backend=selected_provider,
    )


def missing_api_key(model_or_provider: ModelConfig | str) -> str | None:
    if isinstance(model_or_provider, ModelConfig):
        if model_or_provider.api_key:
            return None
        provider = model_or_provider.provider
    else:
        provider = model_or_provider
    profile = _PROFILE_DEFAULTS.get(provider)
    key_name = profile["key_env"] if profile else _KEY_ENV.get(provider)
    if key_name and not os.getenv(key_name, "").strip():
        fallback = profile.get("fallback_key_env") if profile else None
        if not fallback or not os.getenv(fallback, "").strip():
            return key_name
    return None

