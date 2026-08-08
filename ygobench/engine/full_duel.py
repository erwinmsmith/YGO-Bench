"""Headless full-duel runner on the pinned EDOPro ocgcore build."""

from __future__ import annotations

import ctypes
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ygobench.agents.action_space import legal_actions_from_pending
from ygobench.agents.base import BaseAgent
from ygobench.agents.passive_agent import PassiveAgent
from ygobench.config import PROJECT_ROOT
from ygobench.engine.protocol import ActionChoice, DecisionRequest
from ygobench.engine.upstream import UpstreamLayout
from ygobench.engine.visibility import sanitize_events_for_player


@dataclass(frozen=True)
class FullDuelResult:
    winner: int | None
    game_over: bool
    decisions: int
    turn_count: int
    lp: tuple[int, int]
    termination: str
    replay_path: Path
    agent1: str = "passive"
    agent2: str = "passive"
    illegal_actions: tuple[int, int] = (0, 0)
    decision_seconds: tuple[float, float] = (0.0, 0.0)
    model_calls: tuple[int, int] = (0, 0)
    input_tokens: tuple[int, int] = (0, 0)
    output_tokens: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["replay_path"] = str(self.replay_path)
        return data


def _load_upstream():
    layout = UpstreamLayout()
    layout.require_runtime()
    source = str(layout.root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from engine import core, harness, replay, state, tools  # type: ignore[import-not-found]

    return layout, core, harness, replay, state, tools


def _parse_deck(path: Path) -> dict[str, list[int]]:
    sections: dict[str, list[int]] = {"main": [], "extra": [], "side": []}
    current: str | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line == "#main":
            current = "main"
        elif line == "#extra":
            current = "extra"
        elif line == "!side":
            current = "side"
        elif line.isdigit() and current:
            sections[current].append(int(line))
    if len(sections["main"]) < 40:
        raise ValueError(f"Deck must contain at least 40 main-deck cards: {path}")
    return sections


def _derive_engine_seeds(seed: int) -> tuple[int, int, int, int]:
    rng = random.Random(seed)
    return tuple(rng.getrandbits(64) for _ in range(4))  # type: ignore[return-value]


def _create_match(engine, core, *, seed: int, flags: int) -> tuple[int, int, int, int]:
    engine_seeds = _derive_engine_seeds(seed)
    options = core.OCG_DuelOptions()
    options.seed0, options.seed1, options.seed2, options.seed3 = engine_seeds
    options.flags = flags
    options.team1 = core.OCG_Player(startingLP=8000, startingDrawCount=5, drawCountPerTurn=1)
    options.team2 = core.OCG_Player(startingLP=8000, startingDrawCount=5, drawCountPerTurn=1)
    options.cardReader = engine._make_card_reader()
    options.payload1 = None
    options.scriptReader = engine._make_script_reader()
    options.payload2 = None
    options.logHandler = engine._make_log_handler()
    options.payload3 = None
    options.cardReaderDone = engine._make_card_reader_done()
    options.payload4 = None
    options.enableUnsafeLibraries = 0

    duel_pointer = ctypes.c_void_p()
    status = engine.lib.OCG_CreateDuel(ctypes.byref(duel_pointer), ctypes.byref(options))
    if status != 0:
        raise RuntimeError(f"OCG_CreateDuel failed with status {status}")
    engine.duel = duel_pointer
    engine._log_messages = []
    engine._load_script("constant.lua")
    engine._load_script("utility.lua")
    return engine_seeds


def _add_deck(engine, core, *, player: int, deck: dict[str, list[int]]) -> None:
    for code in deck["main"]:
        card = core.OCG_NewCardInfo(
            team=player,
            duelist=0,
            code=code,
            con=player,
            loc=core.LOCATION_DECK,
            seq=0,
            pos=core.POS_FACEDOWN_DEFENSE,
        )
        engine.lib.OCG_DuelNewCard(engine.duel, ctypes.byref(card))
    for code in reversed(deck["extra"]):
        card = core.OCG_NewCardInfo(
            team=player,
            duelist=0,
            code=code,
            con=player,
            loc=core.LOCATION_EXTRA,
            seq=0,
            pos=core.POS_FACEDOWN_DEFENSE,
        )
        engine.lib.OCG_DuelNewCard(engine.duel, ctypes.byref(card))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "agent"


def _usage(agent: BaseAgent, key: str) -> int:
    usage = getattr(agent, "usage", {})
    value = usage.get(key, 0) if isinstance(usage, dict) else 0
    return int(value) if isinstance(value, (int, float)) else 0


def _normalize_action(action: ActionChoice, core: Any, tools_module: Any) -> dict[str, Any]:
    args = dict(action.arguments)
    if action.tool == "select_place":
        location_map = {
            "monster_zone": core.LOCATION_MZONE,
            "spell_zone": core.LOCATION_SZONE,
            "pendulum_zone": core.LOCATION_SZONE,
        }
        args["places"] = [
            {
                **place,
                "location": location_map.get(place.get("location"), place.get("location")),
            }
            for place in args.get("places", [])
        ]
    return tools_module.coerce_args(action.tool, args)


def _passive_fallback(pending: Any, replay_module: Any) -> ActionChoice:
    tool, arguments = replay_module._pick_passive_opponent_response(pending)
    if "ROCK_PAPER_SCISSORS" in str(pending.msg_name).upper():
        arguments = {"hand": 1 if pending.player == 0 else 2}
    return ActionChoice(tool=tool, arguments=arguments, label="engine fallback")


def run_duel(
    deck1_path: Path,
    deck2_path: Path,
    *,
    agent1: BaseAgent,
    agent2: BaseAgent,
    seed: int = 0,
    max_decisions: int = 2000,
    output_dir: Path | None = None,
    replay_name: str | None = None,
) -> FullDuelResult:
    """Run one complete match and write a replayable evidence trace."""

    started_at = time.perf_counter()
    layout, core, harness_module, replay_module, state_module, tools_module = _load_upstream()
    card_db = core.CardDB(layout.root / "vendor" / "distribution" / "expansions")
    engine = core.OCGEngine(
        dylib_path=layout.engine_library,
        card_db=card_db,
        script_dir=layout.root / "vendor" / "distribution" / "script",
        card_script_dir=layout.root / "vendor" / "distribution" / "script" / "official",
    )
    agents = (agent1, agent2)
    for agent in agents:
        agent.reset()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = f"{_safe_name(agent1.name)}_vs_{_safe_name(agent2.name)}"
    output_dir = output_dir or PROJECT_ROOT / "bench_data" / "runs" / f"{stamp}_full_duel_{label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / (replay_name or f"duel_seed_{seed}.jsonl")
    replay_path.write_text("")

    def log(payload: dict[str, Any]) -> None:
        with replay_path.open("a") as handle:
            handle.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")

    duel = harness_module.Harness(engine)
    decisions = 0
    decisions_by_player = [0, 0]
    illegal = [0, 0]
    decision_seconds = [0.0, 0.0]
    termination = "decision_budget_exhausted"
    forfeit_winner: int | None = None
    recent_actions: list[dict[str, Any]] = []
    try:
        deck1 = _parse_deck(deck1_path)
        deck2 = _parse_deck(deck2_path)
        _create_match(engine, core, seed=seed, flags=core.DUEL_MODE_MR5)
        _add_deck(engine, core, player=0, deck=deck1)
        _add_deck(engine, core, player=1, deck=deck2)
        log(
            {
                "type": "config",
                "seed": seed,
                "deck1": str(deck1_path),
                "deck2": str(deck2_path),
                "agents": [agent1.name, agent2.name],
                "agent_configs": [
                    getattr(agent1, "provider_config", {"name": agent1.name}),
                    getattr(agent2, "provider_config", {"name": agent2.name}),
                ],
                "rules": "MR5",
                "starting_lp": 8000,
                "starting_hand": 5,
                "draw_per_turn": 1,
                "prompt_template": "full_duel_system.md@v1",
            }
        )
        engine.start_duel()
        step = duel.advance()
        log({"type": "start", "events": step.events})
        while (
            not duel.state.game_over
            and duel.pending is not None
            and (max_decisions <= 0 or decisions < max_decisions)
        ):
            player = int(duel.pending.player)
            observation = state_module.build_state(
                duel, card_db, perspective=player, events=step.events
            )
            observation["events_since_last_decision"] = sanitize_events_for_player(
                observation.get("events_since_last_decision"),
                perspective=player,
            )
            observation["recent_actions"] = recent_actions[-8:]
            legal_actions = legal_actions_from_pending(
                duel.pending,
                card_db=card_db,
                replay_module=replay_module,
                state_module=state_module,
            )
            if request_decision := observation.get("decision"):
                if request_decision.get("responder") == "announce_card":
                    request_decision["candidate_cards"] = [
                        {"card_code": action.arguments["card_code"], "name": action.label}
                        for action in legal_actions
                    ]
            request = DecisionRequest(
                player=player,
                observation=observation,
                legal_actions=legal_actions,
                decision_type=str(observation.get("decision", {}).get("responder", "")),
            )
            log({"type": "observation", "player": player, "state": observation})
            agent = agents[player]
            call_started = time.perf_counter()
            agent_error: str | None = None
            try:
                action = agent.predict(request)
            except Exception as exc:  # noqa: BLE001
                action = _passive_fallback(duel.pending, replay_module)
                illegal[player] += 1
                agent_error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - call_started
            decision_seconds[player] += elapsed

            expected = request.decision_type
            if action.tool != expected:
                agent_error = f"expected {expected}, got {action.tool}"
                illegal[player] += 1
                action = _passive_fallback(duel.pending, replay_module)

            trace = getattr(agent, "last_trace", {})
            log(
                {
                    "type": "model_turn",
                    "player": player,
                    "agent": agent.name,
                    "text": action.label,
                    "tool_calls": [{"name": action.tool, "arguments": action.arguments}],
                    "elapsed_seconds": round(elapsed, 6),
                    "agent_error": agent_error,
                    "trace": trace,
                }
            )
            try:
                method = getattr(duel, tools_module.TOOL_TO_HARNESS_METHOD[action.tool])
                step = method(**_normalize_action(action, core, tools_module))
                is_fallback = False
            except Exception as exc:  # noqa: BLE001
                illegal[player] += 1
                log(
                    {
                        "type": "invalid_action",
                        "player": player,
                        "agent": agent.name,
                        "attempted": {"name": action.tool, "arguments": action.arguments},
                        "error": f"{type(exc).__name__}: {exc}",
                        "resolution": "forfeit",
                    }
                )
                forfeit_winner = 1 - player
                termination = "illegal_action_forfeit"
                break
            decisions += 1
            decisions_by_player[player] += 1
            action_record = {
                "decision": decisions,
                "turn": duel.state.turn_count,
                "player": player,
                "agent": agent.name,
                "tool": action.tool,
                "arguments": action.arguments,
                "fallback": is_fallback,
            }
            recent_actions.append(action_record)
            log(
                {
                    "type": "tool_result",
                    "player": player,
                    "tool": action.tool,
                    "events": step.events,
                    "fallback": is_fallback,
                }
            )

        if forfeit_winner is not None:
            termination = "illegal_action_forfeit"
        elif duel.state.game_over:
            termination = "game_over"
        elif duel.pending is None:
            termination = "no_pending_decision"
        elapsed_total = round(time.perf_counter() - started_at, 3)
        winner_value = forfeit_winner if forfeit_winner is not None else duel.state.winner
        logical_game_over = duel.state.game_over or forfeit_winner is not None
        outcome = {
            "type": "outcome",
            "benchmark_type": "full_duel",
            "termination": termination,
            "game_over": logical_game_over,
            "winner": winner_value,
            "winner_agent": (agents[winner_value].name if winner_value is not None else None),
            "turn_count": duel.state.turn_count,
            "lp": list(duel.state.lp),
            "tool_calls_used": decisions,
            "decisions_by_player": decisions_by_player,
            "illegal_actions": illegal,
            "decision_seconds": [round(value, 6) for value in decision_seconds],
            "model_usage_totals": {
                "model_calls": [int(getattr(agent, "model_calls", 0)) for agent in agents],
                "input_tokens": [_usage(agent, "input_tokens") for agent in agents],
                "output_tokens": [_usage(agent, "output_tokens") for agent in agents],
            },
            "elapsed": elapsed_total,
        }
        log(outcome)
    finally:
        engine.destroy()

    result = FullDuelResult(
        winner=forfeit_winner if forfeit_winner is not None else duel.state.winner,
        game_over=duel.state.game_over or forfeit_winner is not None,
        decisions=decisions,
        turn_count=duel.state.turn_count,
        lp=tuple(duel.state.lp),
        termination=termination,
        replay_path=replay_path,
        agent1=agent1.name,
        agent2=agent2.name,
        illegal_actions=tuple(illegal),
        decision_seconds=tuple(round(value, 6) for value in decision_seconds),
        model_calls=tuple(int(getattr(agent, "model_calls", 0)) for agent in agents),
        input_tokens=tuple(_usage(agent, "input_tokens") for agent in agents),
        output_tokens=tuple(_usage(agent, "output_tokens") for agent in agents),
    )
    elapsed_total = round(time.perf_counter() - started_at, 3)
    summary_row = {
        **result.to_dict(),
        "replay_path": str(result.replay_path),
        "elapsed": elapsed_total,
        "deck1": deck1_path.stem,
        "deck2": deck2_path.stem,
        "seed": seed,
    }
    summary = {
        "benchmark_type": "full_duel",
        "counts": {termination: 1},
        "per_instance": {replay_path.stem: summary_row},
    }
    (output_dir / "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metrics = {
        "benchmark_type": "full_duel",
        "total": 1,
        "engine_completed": int(result.game_over),
        "engine_completion_rate": float(result.game_over),
        "winner": result.winner,
        "winner_agent": agents[result.winner].name if result.winner is not None else None,
        "agents": [agent1.name, agent2.name],
        "decks": [deck1_path.stem, deck2_path.stem],
        "turn_count": result.turn_count,
        "avg_tool_calls": float(result.decisions),
        "illegal_actions": list(result.illegal_actions),
        "illegal_action_rate": sum(result.illegal_actions) / max(1, result.decisions),
        "avg_decision_seconds": sum(result.decision_seconds) / max(1, result.decisions),
        "input_tokens": sum(result.input_tokens),
        "output_tokens": sum(result.output_tokens),
        "elapsed_seconds": elapsed_total,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return result


def run_passive_duel(
    deck1_path: Path,
    deck2_path: Path,
    *,
    seed: int = 0,
    max_decisions: int = 2000,
    output_dir: Path | None = None,
) -> FullDuelResult:
    """Backward-compatible full-duel smoke baseline."""

    return run_duel(
        deck1_path,
        deck2_path,
        agent1=PassiveAgent(),
        agent2=PassiveAgent(),
        seed=seed,
        max_decisions=max_decisions,
        output_dir=output_dir,
    )
