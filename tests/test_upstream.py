from pathlib import Path

from ygobench.engine.upstream import UpstreamLayout


def test_layout_prefers_enriched_dataset(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    lean = data / "yugioh_bench.jsonl"
    enriched = data / "yugioh_bench.enriched.jsonl"
    lean.write_text("")
    assert UpstreamLayout(tmp_path).dataset == lean
    enriched.write_text("")
    assert UpstreamLayout(tmp_path).dataset == enriched

