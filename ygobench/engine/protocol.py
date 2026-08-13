"""Game-agnostic protocol used by the future full-duel environment.

Unlike PTCG, ocgcore pauses at a sequence of protocol-level decisions.  Each
``ActionChoice`` is therefore a fully concrete legal response to the current
pending engine message, not a free-form natural-language command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ActionChoice:
    tool: str
    arguments: dict[str, Any]
    label: str = ""


@dataclass(frozen=True)
class DecisionRequest:
    player: int
    observation: dict[str, Any]
    legal_actions: tuple[ActionChoice, ...]
    decision_type: str
    # True only when ``legal_actions`` is the complete engine-derived action
    # set, rather than a bounded policy candidate set.
    legal_actions_complete: bool = False


@dataclass(frozen=True)
class StepResult:
    decision: DecisionRequest | None
    done: bool
    winner: int | None = None
    reward: tuple[float, float] = (0.0, 0.0)
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DuelEnvironment(Protocol):
    def reset(self, *, seed: int | None = None) -> StepResult: ...

    def step(self, action: ActionChoice) -> StepResult: ...

    def close(self) -> None: ...

