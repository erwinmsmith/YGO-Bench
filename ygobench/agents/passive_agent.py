"""Deterministic no-op/first-legal baseline for full duels."""

from __future__ import annotations

from ygobench.agents.base import BaseAgent
from ygobench.engine.protocol import ActionChoice, DecisionRequest


class PassiveAgent(BaseAgent):
    name = "passive"

    def predict(self, decision: DecisionRequest) -> ActionChoice:
        if not decision.legal_actions:
            raise ValueError("No legal actions are available")
        return decision.legal_actions[0]
