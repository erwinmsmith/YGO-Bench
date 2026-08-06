"""Card metadata and cache-on-demand image service."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from ygobench.config import PROJECT_ROOT
from ygobench.engine.upstream import UpstreamLayout

IMAGE_CACHE = PROJECT_ROOT / "backend" / ".cache" / "card_images"
IMAGE_ORIGIN = "https://images.ygoprodeck.com/images/cards/{code}.jpg"

_TYPE_NAMES = (
    (0x1, "Monster"),
    (0x2, "Spell"),
    (0x4, "Trap"),
    (0x10, "Normal"),
    (0x20, "Effect"),
    (0x40, "Fusion"),
    (0x80, "Ritual"),
    (0x1000, "Tuner"),
    (0x2000, "Synchro"),
    (0x10000, "Quick-Play"),
    (0x20000, "Continuous"),
    (0x40000, "Equip"),
    (0x80000, "Field"),
    (0x100000, "Counter"),
    (0x200000, "Flip"),
    (0x800000, "Xyz"),
    (0x1000000, "Pendulum"),
    (0x4000000, "Link"),
)


def _database_path() -> Path:
    return UpstreamLayout().root / "vendor" / "distribution" / "expansions" / "cards.cdb"


def _row_to_card(row: sqlite3.Row) -> dict[str, Any]:
    type_mask = int(row[4] or 0)
    return {
        "id": int(row[0]),
        "name": row[1] or f"Card #{row[0]}",
        "description": row[2] or "",
        "alias": int(row[3] or 0),
        "type": [name for bit, name in _TYPE_NAMES if type_mask & bit],
        "attack": int(row[5]) if row[5] is not None and row[5] >= 0 else None,
        "defense": int(row[6]) if row[6] is not None and row[6] >= 0 else None,
        "level": int(row[7] or 0) & 0xFF,
        "race": int(row[8] or 0),
        "attribute": int(row[9] or 0),
        "image_url": f"/api/cards/{int(row[0])}/image",
    }


def get_cards(card_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
    ids = list(dict.fromkeys(int(card_id) for card_id in card_ids))
    db_path = _database_path()
    if not ids or not db_path.is_file():
        return {}
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT d.id, t.name, t.desc, d.alias, d.type, d.atk, d.def,
               d.level, d.race, d.attribute
        FROM datas d JOIN texts t ON t.id = d.id
        WHERE d.id IN ({placeholders})
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return {int(row[0]): _row_to_card(row) for row in connection.execute(query, ids)}


@lru_cache(maxsize=4096)
def get_card(card_id: int) -> dict[str, Any] | None:
    return get_cards([card_id]).get(int(card_id))


async def cache_card_image(card_id: int) -> Path:
    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    destination = IMAGE_CACHE / f"{int(card_id)}.jpg"
    if destination.is_file() and destination.stat().st_size > 1024:
        return destination

    url = IMAGE_ORIGIN.format(code=int(card_id))
    headers = {"User-Agent": "YGO-Bench/0.1 research card-image cache"}
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if not content_type.startswith("image/"):
        raise ValueError(f"Unexpected card image content type: {content_type}")
    if len(response.content) > 5_000_000:
        raise ValueError("Card image exceeds 5 MB safety limit")
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination
