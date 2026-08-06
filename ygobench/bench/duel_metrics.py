"""Arena metrics for complete duels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ygobench.bench.glicko2 import Glicko2, GlickoPlayer


def summarize_duels(games: list[dict[str, Any]]) -> dict[str, Any]:
    agent_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "first_games": 0,
            "first_wins": 0,
            "second_games": 0,
            "second_wins": 0,
            "illegal_actions": 0,
            "decisions": 0,
            "decision_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )
    matchup: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"games": 0, "a_wins": 0, "b_wins": 0, "draws": 0}
    )
    deck_matchup: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"games": 0, "deck1_wins": 0, "deck2_wins": 0, "draws": 0}
    )
    elo: dict[str, float] = defaultdict(lambda: 1500.0)
    glicko: dict[str, GlickoPlayer] = defaultdict(GlickoPlayer)
    completed = 0
    total_turns = total_decisions = 0

    for game in games:
        agents = game["agents"]
        decks = game["decks"]
        winner = game.get("winner")
        if game.get("game_over"):
            completed += 1
        total_turns += int(game.get("turn_count", 0))
        total_decisions += int(game.get("decisions", 0))
        for seat, agent in enumerate(agents):
            row = agent_rows[agent]
            row["games"] += 1
            row["first_games" if seat == 0 else "second_games"] += 1
            if winner is None:
                row["draws"] += 1
            elif winner == seat:
                row["wins"] += 1
                row["first_wins" if seat == 0 else "second_wins"] += 1
            else:
                row["losses"] += 1
            row["illegal_actions"] += int(game.get("illegal_actions", [0, 0])[seat])
            row["decisions"] += int(game.get("decisions_by_player", [0, 0])[seat])
            row["decision_seconds"] += float(game.get("decision_seconds", [0, 0])[seat])
            usage = game.get("model_usage_totals", {})
            row["input_tokens"] += int(usage.get("input_tokens", [0, 0])[seat])
            row["output_tokens"] += int(usage.get("output_tokens", [0, 0])[seat])

        key = tuple(sorted(agents))
        match = matchup[key]
        match["games"] += 1
        if winner is None:
            match["draws"] += 1
        else:
            winning_agent = agents[winner]
            match["a_wins" if winning_agent == key[0] else "b_wins"] += 1

        deck_key = (decks[0], decks[1])
        deck_row = deck_matchup[deck_key]
        deck_row["games"] += 1
        if winner is None:
            deck_row["draws"] += 1
        else:
            deck_row["deck1_wins" if winner == 0 else "deck2_wins"] += 1

        if winner is None:
            scores = (0.5, 0.5)
        else:
            scores = (1.0, 0.0) if winner == 0 else (0.0, 1.0)
        expected0 = 1.0 / (1.0 + 10 ** ((elo[agents[1]] - elo[agents[0]]) / 400.0))
        elo[agents[0]] += 24.0 * (scores[0] - expected0)
        elo[agents[1]] += 24.0 * (scores[1] - (1.0 - expected0))

    snapshots = dict(glicko)
    for agent in agent_rows:
        results = []
        for game in games:
            if agent not in game["agents"]:
                continue
            seat = game["agents"].index(agent)
            opponent = game["agents"][1 - seat]
            winner = game.get("winner")
            score = 0.5 if winner is None else (1.0 if winner == seat else 0.0)
            results.append((snapshots.get(opponent, GlickoPlayer()), score))
        glicko[agent] = Glicko2.update(snapshots.get(agent, GlickoPlayer()), results)

    leaderboard = []
    for agent, row in agent_rows.items():
        games_played = row["games"] or 1
        decisions = row["decisions"] or 1
        leaderboard.append(
            {
                "agent": agent,
                **row,
                "win_rate": row["wins"] / games_played,
                "first_win_rate": row["first_wins"] / max(1, row["first_games"]),
                "second_win_rate": row["second_wins"] / max(1, row["second_games"]),
                "illegal_action_rate": row["illegal_actions"] / decisions,
                "avg_decision_seconds": row["decision_seconds"] / decisions,
                "elo": round(elo[agent], 2),
                "glicko2": round(glicko[agent].rating, 2),
                "rating_deviation": round(glicko[agent].deviation, 2),
            }
        )
    leaderboard.sort(key=lambda row: (row["glicko2"], row["win_rate"]), reverse=True)

    return {
        "benchmark_type": "full_duel_arena",
        "total": len(games),
        "engine_completed": completed,
        "engine_completion_rate": completed / max(1, len(games)),
        "avg_turns": total_turns / max(1, len(games)),
        "avg_decisions": total_decisions / max(1, len(games)),
        "leaderboard": leaderboard,
        "agent_matchups": [
            {"agent_a": key[0], "agent_b": key[1], **value}
            for key, value in sorted(matchup.items())
        ],
        "deck_matchups": [
            {"deck1": key[0], "deck2": key[1], **value}
            for key, value in sorted(deck_matchup.items())
        ],
    }
