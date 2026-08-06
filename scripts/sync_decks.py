"""Copy the curated, ygo-agent-tested decks into YGO-Bench resources."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "vendor" / "ygo-agent" / "assets" / "deck"
TARGET = ROOT / "resources" / "decks"
MANIFEST = TARGET / "manifest.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    TARGET.mkdir(parents=True, exist_ok=True)
    for deck_id in manifest:
        source = SOURCE / f"{deck_id}.ydk"
        if not source.is_file():
            raise FileNotFoundError(f"Missing upstream deck: {source}")
        destination = TARGET / source.name
        shutil.copyfile(source, destination)
        print(f"synced {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

