"""Deterministic, in-memory deck sampling and paired seat assignments.

This module deliberately does not create a tournament roster or start duels.
The same sampled deck pairs can be reused for every model pairing.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from ygobench.experiments.config import canonical_json

DeckPair = tuple[str, str]


def _digest(*parts: object) -> bytes:
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).digest()


def _seed64(*parts: object) -> int:
    return int.from_bytes(_digest(*parts)[:8], "big")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("master seed must be an integer")


def _validate_balanced_pairs(deck_pairs: Sequence[DeckPair]) -> tuple[DeckPair, ...]:
    pairs = tuple(deck_pairs)
    if len(pairs) != 10 or len(set(pairs)) != 10:
        raise ValueError("exactly ten distinct deck pairs are required")
    if any(len(pair) != 2 or pair[0] >= pair[1] for pair in pairs):
        raise ValueError("deck pairs must contain two distinct names in sorted order")
    degree = Counter(deck for pair in pairs for deck in pair)
    if len(degree) != 8 or sorted(degree.values()) != [2, 2, 2, 2, 3, 3, 3, 3]:
        raise ValueError("eight decks must each appear two or three times")
    return pairs


def sample_balanced_deck_pairs(decks: Sequence[str], *, seed: int) -> tuple[DeckPair, ...]:
    """Draw ten of the 28 non-mirror pairs, conditional on balanced coverage.

    Each candidate is a hash-ranked pseudo-random ten-subset. Rejection keeps
    the draw exchangeable across the eight decks, while the fixed seed makes it
    reproducible across Python versions and machines.
    """

    _validate_seed(seed)
    names = tuple(sorted(decks))
    if len(names) != 8 or len(set(names)) != 8 or any(not name for name in names):
        raise ValueError("exactly eight distinct non-empty deck names are required")
    possible = tuple(combinations(names, 2))
    for attempt in range(100_000):
        candidate = tuple(
            sorted(
                sorted(
                    possible,
                    key=lambda pair: (_digest("deck-sample", seed, attempt, pair), pair),
                )[:10]
            )
        )
        try:
            return _validate_balanced_pairs(candidate)
        except ValueError:
            continue
    raise RuntimeError("could not sample ten balanced deck pairs")


def _home_deck_mask(deck_pairs: tuple[DeckPair, ...], seed: int) -> int:
    degree = Counter(deck for pair in deck_pairs for deck in pair)
    eligible = []
    for mask in range(1 << len(deck_pairs)):
        home = Counter(
            pair[0] if mask & (1 << index) else pair[1]
            for index, pair in enumerate(deck_pairs)
        )
        if all(abs(2 * home[deck] - count) <= 1 for deck, count in degree.items()):
            eligible.append(mask)
    if not eligible:
        raise ValueError("deck pairs cannot be assigned fairly to two models")
    return min(eligible, key=lambda mask: (_digest("deck-sides", seed, deck_pairs, mask), mask))


def assign_balanced_decks(
    deck_pairs: Sequence[DeckPair],
    *,
    model_ids: Sequence[str],
    model_a: str,
    model_b: str,
    seed: int,
) -> tuple[DeckPair, ...]:
    """Give each model one deck per pair and balance deck exposure globally.

    The seven models form a regular cyclic tournament: each model receives the
    home deck pattern against three opponents and its complement against three.
    Returned pairs are ordered as (deck for model_a, deck for model_b).
    """

    _validate_seed(seed)
    pairs = _validate_balanced_pairs(deck_pairs)
    models = tuple(sorted(model_ids))
    if len(models) != 7 or len(set(models)) != 7 or any(not model for model in models):
        raise ValueError("exactly seven distinct non-empty model IDs are required")
    if model_a == model_b or model_a not in models or model_b not in models:
        raise ValueError("the model pair must contain two distinct registered models")
    index_a, index_b = models.index(model_a), models.index(model_b)
    a_is_home = 0 < (index_b - index_a) % 7 <= 3
    mask = _home_deck_mask(pairs, seed)
    assignments = []
    for index, (left, right) in enumerate(pairs):
        home, away = (left, right) if mask & (1 << index) else (right, left)
        assignments.append((home, away) if a_is_home else (away, home))
    return tuple(assignments)


@dataclass(frozen=True)
class SeatSwappedDuel:
    player0: str
    deck0: str
    player1: str
    deck1: str
    seed: int
    deck_shuffle_seeds: tuple[int, int]


def paired_seat_swaps(
    *, model_a: str, model_b: str, deck_a: str, deck_b: str, seed: int
) -> tuple[SeatSwappedDuel, SeatSwappedDuel]:
    """Describe two duels that differ only in the models' starting seats.

    Engine seed is shared by the two legs. Shuffle seeds follow each
    model/deck binding when its seat changes, preserving its opening deck order.
    """

    _validate_seed(seed)
    if not model_a or not model_b or model_a == model_b:
        raise ValueError("two distinct model IDs are required")
    if not deck_a or not deck_b or deck_a == deck_b:
        raise ValueError("two distinct deck names are required")
    models = tuple(sorted((model_a, model_b)))
    decks = tuple(sorted((deck_a, deck_b)))
    pair_seed = _seed64("paired-engine", seed, models, decks)
    shuffle_a = _seed64("paired-shuffle", seed, models, decks, model_a, deck_a)
    shuffle_b = _seed64("paired-shuffle", seed, models, decks, model_b, deck_b)
    return (
        SeatSwappedDuel(model_a, deck_a, model_b, deck_b, pair_seed, (shuffle_a, shuffle_b)),
        SeatSwappedDuel(model_b, deck_b, model_a, deck_a, pair_seed, (shuffle_b, shuffle_a)),
    )
