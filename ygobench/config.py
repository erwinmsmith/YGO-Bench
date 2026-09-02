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


_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-v4-flash",
    "dashscope": "qwen3.7-flash",
    "openai": "gpt-5",
    "vllm": "local-model",
    "claude-cli": "claude-sonnet-4-6",
}

_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def default_model_config(provider: str | None = None, model: str | None = None) -> ModelConfig:
    selected_provider = (provider or os.getenv("LLM_PROVIDER", "deepseek")).strip().lower()
    if selected_provider not in _DEFAULT_MODELS:
        known = ", ".join(sorted(_DEFAULT_MODELS))
        raise ValueError(f"Unknown provider {selected_provider!r}; choose one of: {known}")
    # An explicit provider must not inherit an unrelated global LLM_MODEL.
    # For example, `react:dashscope` should resolve to Qwen even when .env
    # keeps DeepSeek as the default for legacy runs.
    inherited_model = os.getenv("LLM_MODEL") if provider is None else None
    selected_model = (model or inherited_model or _DEFAULT_MODELS[selected_provider]).strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    return ModelConfig(provider=selected_provider, model=selected_model, base_url=base_url)


def missing_api_key(provider: str) -> str | None:
    key_name = _KEY_ENV.get(provider)
    if key_name and not os.getenv(key_name, "").strip():
        return key_name
    return None

