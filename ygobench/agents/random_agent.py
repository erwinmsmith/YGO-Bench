"""Uniform random legal-action baseline."""

from __future__ import annotations

import random

from ygobench.agents.base import BaseAgent
from ygobench.engine.protocol import ActionChoice, DecisionRequest


class RandomAgent(BaseAgent):
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def reset(self) -> None:
        self._rng.seed(self._seed)

    def predict(self, decision: DecisionRequest) -> ActionChoice:
        if not decision.legal_actions:
            raise ValueError("No legal actions are available")
        return self._rng.choice(decision.legal_actions)

