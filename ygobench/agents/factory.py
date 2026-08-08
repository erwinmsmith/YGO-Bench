"""Create full-duel agents from stable CLI identifiers."""

from __future__ import annotations

from ygobench.agents.base import BaseAgent
from ygobench.agents.llm_agent import LLMFullDuelAgent
from ygobench.agents.passive_agent import PassiveAgent
from ygobench.agents.random_agent import RandomAgent
from ygobench.config import default_model_config


def create_agent(agent_id: str, *, seed: int = 0) -> BaseAgent:
    if agent_id == "passive":
        return PassiveAgent()
    if agent_id == "random":
        return RandomAgent(seed=seed)
    if agent_id.startswith("react-fast"):
        parts = agent_id.split(":", 2)
        provider = parts[1] if len(parts) > 1 and parts[1] else None
        model = parts[2] if len(parts) > 2 and parts[2] else None
        return LLMFullDuelAgent(
            default_model_config(provider, model),
            thinking_enabled=False,
            profile="react-fast",
        )
    if agent_id.startswith("react"):
        parts = agent_id.split(":", 2)
        provider = parts[1] if len(parts) > 1 and parts[1] else None
        model = parts[2] if len(parts) > 2 and parts[2] else None
        return LLMFullDuelAgent(default_model_config(provider, model))
    raise ValueError(
        f"Unknown full-duel agent {agent_id!r}; use passive, random, "
        "react[:provider:model], or react-fast[:provider:model]."
    )
