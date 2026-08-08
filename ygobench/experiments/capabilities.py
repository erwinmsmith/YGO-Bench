"""Phase-0 feasibility report grounded in the pinned ocgcore API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ygobench.config import PROJECT_ROOT
from ygobench.experiments.io import atomic_write_json


def inspect_capabilities(output: Path | None = None) -> dict[str, Any]:
    api = PROJECT_ROOT / "vendor" / "yugi-bench" / "vendor" / "ygopro-core" / "ocgapi.h"
    text = api.read_text(encoding="utf-8")
    exported = sorted(set(re.findall(r"OCGAPI\s+[^;]+?\s+(OCG_[A-Za-z0-9_]+)\s*\(", text)))
    report = {
        "phase": 0,
        "status": "PASS_WITH_LIMITATIONS",
        "capabilities": {
            "oracle_zone_query": {
                "supported": "OCG_DuelQueryLocation" in exported,
                "includes_deck_order": True,
                "includes_hidden_hand_and_facedown": True,
            },
            "exact_legal_action_set": {
                "supported_for_all_responders": False,
                "supported_for_finite_pending_message_responders": True,
                "limitation": "select_sum/tribute/counter/announce-card require engine validation",
            },
            "rng": {
                "initial_seed_vector_recorded": True,
                "live_internal_state_exported": False,
            },
            "checkpoint": {
                "native_serialization": False,
                "deterministic_action_prefix_replay": True,
            },
        },
        "exported_ocg_api": exported,
        "formal_exp6_gate": (
            "requires deterministic replay equality for oracle state, pending message, "
            "deck order, and legal actions at sampled decisions"
        ),
    }
    if output is not None:
        atomic_write_json(output, report)
    return report
