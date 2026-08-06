"""Parse curated .ydk files and join local BabelCDB metadata."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.card_service import get_cards
from ygobench.config import PROJECT_ROOT

DECK_ROOT = PROJECT_ROOT / "resources" / "decks"


def _parse_ydk(path: Path) -> dict[str, list[int]]:
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
    return sections


def _card_rows(card_ids: list[int], metadata: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(card_ids)
    rows = []
    for card_id, count in counts.items():
        card = metadata.get(card_id, {"id": card_id, "name": f"Card #{card_id}"})
        rows.append({**card, "count": count})
    return rows


@lru_cache(maxsize=1)
def list_decks() -> list[dict[str, Any]]:
    manifest_path = DECK_ROOT / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text())
    parsed: list[tuple[str, dict[str, Any], dict[str, list[int]]]] = []
    all_ids: list[int] = []
    for deck_id, deck_info in manifest.items():
        path = DECK_ROOT / f"{deck_id}.ydk"
        if not path.is_file():
            continue
        sections = _parse_ydk(path)
        all_ids.extend(card_id for values in sections.values() for card_id in values)
        parsed.append((deck_id, deck_info, sections))

    metadata = get_cards(all_ids)
    decks = []
    for deck_id, deck_info, sections in parsed:
        main_cards = _card_rows(sections["main"], metadata)
        cover = main_cards[0].get("image_url") if main_cards else None
        decks.append(
            {
                "id": deck_id,
                **deck_info,
                "source": "sbl1996/ygo-agent",
                "main_count": len(sections["main"]),
                "extra_count": len(sections["extra"]),
                "side_count": len(sections["side"]),
                "cover_image": cover,
                "sections": {
                    name: _card_rows(ids, metadata) for name, ids in sections.items()
                },
            }
        )
    return decks


def get_deck(deck_id: str) -> dict[str, Any] | None:
    return next((deck for deck in list_decks() if deck["id"] == deck_id), None)

