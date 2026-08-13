"""Single source of truth for creating and stepping a complete duel."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from ygobench.engine.full_duel import (
    _add_deck,
    _create_match,
    _derive_deck_shuffle_seeds,
    _load_upstream,
    _normalize_action,
    _parse_deck,
)
from ygobench.engine.visibility import sanitize_events_for_player
from ygobench.experiments.legal import build_legal_evidence, canonical_engine_arguments
from ygobench.experiments.oracle import build_oracle_state


class DuelSession:
    def __init__(self, deck1: Path, deck2: Path, *, seed: int) -> None:
        (
            self.layout,
            self.core,
            self.harness_module,
            self.replay_module,
            self.state_module,
            self.tools_module,
        ) = _load_upstream()
        self.card_db = self.core.CardDB(self.layout.root / "vendor" / "distribution" / "expansions")
        self.engine = self.core.OCGEngine(
            dylib_path=self.layout.engine_library,
            card_db=self.card_db,
            script_dir=self.layout.root / "vendor" / "distribution" / "script",
            card_script_dir=self.layout.root / "vendor" / "distribution" / "script" / "official",
        )
        self.duel = self.harness_module.Harness(self.engine)
        self.engine_seeds = _create_match(
            self.engine, self.core, seed=seed, flags=self.core.DUEL_MODE_MR5
        )
        self.deck_shuffle_seeds = _derive_deck_shuffle_seeds(seed)
        self.deck_order_hashes = (
            _add_deck(
                self.engine,
                self.core,
                player=0,
                deck=_parse_deck(deck1),
                shuffle_seed=self.deck_shuffle_seeds[0],
            ),
            _add_deck(
                self.engine,
                self.core,
                player=1,
                deck=_parse_deck(deck2),
                shuffle_seed=self.deck_shuffle_seeds[1],
            ),
        )
        self.engine.start_duel()
        self.step_result = self.duel.advance()

    @property
    def done(self) -> bool:
        return bool(self.duel.state.game_over or self.duel.pending is None)

    def view(self, recent_actions: list[dict[str, Any]]) -> dict[str, Any]:
        if self.duel.pending is None:
            raise RuntimeError("duel has no pending decision")
        player = int(self.duel.pending.player)
        observation = self.state_module.build_state(
            self.duel,
            self.card_db,
            perspective=player,
            events=self.step_result.events,
        )
        observation["events_since_last_decision"] = sanitize_events_for_player(
            observation.get("events_since_last_decision"), perspective=player
        )
        observation["recent_actions"] = recent_actions[-8:]
        actions, legal = build_legal_evidence(
            self.duel.pending,
            card_db=self.card_db,
            replay_module=self.replay_module,
            state_module=self.state_module,
        )
        return {
            "player": player,
            "observation": observation,
            "actions": actions,
            "legal": legal,
            "oracle": build_oracle_state(self),
        }

    def execute(self, action: Any) -> Any:
        method = getattr(self.duel, self.tools_module.TOOL_TO_HARNESS_METHOD[action.tool])
        self.step_result = method(**_normalize_action(action, self.core, self.tools_module))
        return self.step_result

    def action_signature(self, action: Any) -> tuple[str, dict[str, Any]]:
        """Canonicalize a tool call exactly as its bound engine method sees it."""

        method = getattr(self.duel, self.tools_module.TOOL_TO_HARNESS_METHOD[action.tool])
        arguments = _normalize_action(action, self.core, self.tools_module)
        bound = inspect.signature(method).bind(**arguments)
        bound.apply_defaults()
        return action.tool, canonical_engine_arguments(action.tool, dict(bound.arguments))

    def close(self) -> None:
        self.engine.destroy()
