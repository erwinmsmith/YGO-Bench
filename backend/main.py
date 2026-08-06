"""FastAPI service for decks, cards, benchmark runs and replay timelines."""

from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.card_service import cache_card_image, get_card
from backend.deck_service import get_deck, list_decks
from backend.replay_service import list_replays, list_runs, load_replay, load_replay_frames
from ygobench.config import PROJECT_ROOT
from ygobench.engine.upstream import UpstreamLayout

app = FastAPI(title="YGO-Bench API", version="0.1.0")
origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "YGO-Bench API", "version": "0.1.0"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    layout = UpstreamLayout()
    return {
        "status": "ready" if not layout.runtime_errors() else "setup_required",
        "engine_ready": layout.engine_library.is_file(),
        "dataset_ready": layout.dataset.is_file(),
        "deck_count": len(list_decks()),
        "run_count": len(list_runs()),
        "runtime_errors": layout.runtime_errors(),
    }


@app.get("/api/decks")
def decks() -> list[dict]:
    return list_decks()


@app.get("/api/decks/{deck_id}")
def deck(deck_id: str) -> dict:
    result = get_deck(deck_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Deck not found")
    return result


@app.get("/api/cards/{card_id}")
def card(card_id: int) -> dict:
    result = get_card(card_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Card metadata unavailable; run setup first")
    return result


@app.get("/api/cards/{card_id}/image", response_class=FileResponse)
async def card_image(card_id: int) -> FileResponse:
    if card_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid card id")
    try:
        path = await cache_card_image(card_id)
    except Exception as exc:
        fallback = PROJECT_ROOT / "frontend" / "public" / "card-back.png"
        if fallback.is_file():
            return FileResponse(fallback, media_type="image/png")
        raise HTTPException(status_code=502, detail=f"Card image unavailable: {exc}") from exc
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/api/runs")
def runs() -> list[dict]:
    return list_runs()


@app.get("/api/replays")
def replays() -> list[dict]:
    return list_replays()


@app.get("/api/replays/{run_id}/{filename}")
def replay(run_id: str, filename: str, full: bool = False) -> dict:
    result = load_replay(run_id, filename, compact=not full)
    if result is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    return result


@app.get("/api/replays/{run_id}/{filename}/frames")
def replay_frames(run_id: str, filename: str) -> dict:
    result = load_replay_frames(run_id, filename)
    if result is None:
        raise HTTPException(status_code=404, detail="Replay not found")
    return result


def run() -> None:
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
