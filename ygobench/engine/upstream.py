"""Resolve and validate the pinned yugi-bench/ocgcore toolchain."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from ygobench.config import PROJECT_ROOT


@dataclass(frozen=True)
class UpstreamLayout:
    root: Path = PROJECT_ROOT / "vendor" / "yugi-bench"

    @property
    def runner(self) -> Path:
        return self.root / "api-eval" / "runner.py"

    @property
    def setup_script(self) -> Path:
        return self.root / "setup.sh"

    @property
    def dataset(self) -> Path:
        enriched = self.root / "data" / "yugioh_bench.enriched.jsonl"
        lean = self.root / "data" / "yugioh_bench.jsonl"
        return enriched if enriched.exists() else lean

    @property
    def engine_library(self) -> Path:
        suffix = "libocgcore.dylib" if platform.system() == "Darwin" else "libocgcore.so"
        return self.root / "vendor" / "ygopro-core" / "bin" / "release" / suffix

    def source_errors(self) -> list[str]:
        errors = []
        if not self.root.is_dir():
            errors.append("vendor/yugi-bench submodule is missing")
        if not self.runner.is_file():
            errors.append(f"upstream runner is missing: {self.runner}")
        if not self.setup_script.is_file():
            errors.append(f"upstream setup script is missing: {self.setup_script}")
        return errors

    def runtime_errors(self) -> list[str]:
        errors = self.source_errors()
        if not self.engine_library.is_file():
            errors.append(f"ocgcore is not built: {self.engine_library}")
        if not self.dataset.is_file():
            errors.append(f"benchmark dataset is not built: {self.dataset}")
        return errors

    def require_runtime(self) -> None:
        errors = self.runtime_errors()
        if errors:
            detail = "\n- ".join(errors)
            raise RuntimeError(f"YGO engine is not ready:\n- {detail}\nRun: ygo-bench setup")

