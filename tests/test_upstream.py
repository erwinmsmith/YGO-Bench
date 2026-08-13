from pathlib import Path
from types import SimpleNamespace

from ygobench.engine.upstream import UpstreamLayout
from ygobench.engine.upstream_fixes import apply_upstream_fixes, normalize_lp


def test_layout_prefers_enriched_dataset(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    lean = data / "yugioh_bench.jsonl"
    enriched = data / "yugioh_bench.enriched.jsonl"
    lean.write_text("")
    assert UpstreamLayout(tmp_path).dataset == lean
    enriched.write_text("")
    assert UpstreamLayout(tmp_path).dataset == enriched


def test_player_one_decision_controller_labels_are_relative() -> None:
    class FakeEngine:
        def query_field(self) -> dict:
            return {"players": [{"lp": 8000}, {"lp": 8000}]}

        def _parse_single_message(self, msg_type: int, reader: object) -> dict:
            return {"player": 1, "lp": 8000}

    def build_decision(decision: object, card_db: object) -> dict:
        return {
            "choices": [
                {"card": {"name": "Player 0 card", "controller": "you"}},
                {"card": {"name": "Player 1 card", "controller": "opponent"}},
            ]
        }

    def build_state(
        harness: object,
        card_db: object,
        *,
        perspective: int = 0,
        include_decision: bool = True,
        events: list[dict] | None = None,
    ) -> dict:
        return {"decision": state.build_decision(harness.pending, card_db)}

    core = SimpleNamespace(OCGEngine=FakeEngine, MSG_LPUPDATE=1)
    state = SimpleNamespace(build_decision=build_decision, build_state=build_state)
    apply_upstream_fixes(core, state)

    decision = SimpleNamespace(player=1)
    rendered = state.build_decision(decision, {})
    assert rendered["choices"][0]["card"]["controller"] == "opponent"
    assert rendered["choices"][1]["card"]["controller"] == "you"

    harness = SimpleNamespace(pending=decision)
    state_rendered = state.build_state(harness, {}, perspective=1)
    assert state_rendered["decision"]["choices"][1]["card"]["controller"] == "you"


def test_wrapped_terminal_lp_is_clamped_and_raw_value_is_retained() -> None:
    wrapped_negative_2800 = (1 << 32) - 2800

    class FakeEngine:
        def query_field(self) -> dict:
            return {"players": [{"lp": 8000}, {"lp": wrapped_negative_2800}]}

        def _parse_single_message(self, msg_type: int, reader: object) -> dict:
            return {"player": 1, "lp": wrapped_negative_2800}

    core = SimpleNamespace(OCGEngine=FakeEngine, MSG_LPUPDATE=7)
    state = SimpleNamespace(
        build_decision=lambda decision, card_db: {},
        build_state=lambda harness, card_db, **kwargs: {},
    )
    apply_upstream_fixes(core, state)

    engine = core.OCGEngine()
    field = engine.query_field()
    assert normalize_lp(wrapped_negative_2800) == 0
    assert field["players"][1]["lp"] == 0
    assert field["players"][1]["lp_raw_u32"] == wrapped_negative_2800

    event = engine._parse_single_message(core.MSG_LPUPDATE, object())
    assert event["lp"] == 0
    assert event["lp_raw_u32"] == wrapped_negative_2800

