"""Base interface shared by rule, LLM and RL duel agents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ygobench.engine.protocol import ActionChoice, DecisionRequest


class BaseAgent(ABC):
    name = "agent"

    @abstractmethod
    def predict(self, decision: DecisionRequest) -> ActionChoice:
        """Choose exactly one response from ``decision.legal_actions``."""

    def reset(self) -> None:
        """Reset per-duel state."""
        return None

    def post_game(self, *, result: str, metadata: dict | None = None) -> None:
        """Receive the terminal result for optional learning/reflection."""
        return None
