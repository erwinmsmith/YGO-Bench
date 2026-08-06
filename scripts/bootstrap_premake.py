"""Install a pinned premake5 binary into the project-local .tools directory."""

from __future__ import annotations

import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / ".tools"
VERSION = "5.0.0-beta8"
BASE_URL = f"https://github.com/premake/premake-core/releases/download/v{VERSION}"


def asset_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        suffix = "macosx-x64" if machine in {"x86_64", "amd64"} else "macosx"
        return f"premake-{VERSION}-{suffix}.tar.gz"
    if system == "Linux":
        return f"premake-{VERSION}-linux.tar.gz"
    raise RuntimeError(f"Automatic premake bootstrap is unsupported on {system}; install premake5")


def main() -> int:
    destination = TOOLS / "premake5"
    if destination.is_file():
        print(destination)
        return 0
    TOOLS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ygobench-premake-") as temporary:
        archive = Path(temporary) / asset_name()
        url = f"{BASE_URL}/{archive.name}"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as bundle:
            bundle.extractall(temporary, filter="data")
        source = Path(temporary) / "premake5"
        if not source.is_file():
            raise RuntimeError("premake5 binary missing from release archive")
        shutil.copy2(source, destination)
    destination.chmod(0o755)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

