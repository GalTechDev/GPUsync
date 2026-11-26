#!/usr/bin/env python3
"""Development tasks, identical on macOS, Linux and Windows.

    python tools/dev.py setup     # creates .venv, installs maturin and pytest, then `pip install -e .` and `-e gfx`
    python tools/dev.py build     # rebuilds the Rust extension in .venv, reinstalls the gfx engine
    python tools/dev.py test      # build + cargo test + pytest (library and engine)
    python tools/dev.py info      # python -m gpusync (device diagnostic)
    python tools/dev.py run SCRIPT [ARGS…]   # runs a script with the Python of .venv

With gs-remote, on every test machine:

    python3 tools/remote/gs_remote.py run all -- @python tools/dev.py test
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEV_DEPS = ["maturin>=1.16,<2", "flit_core>=3.10,<4", "pytest>=8", "numpy>=1.22", "nbclient>=0.10", "ipykernel>=6"]  # Jupyter: M4 test


def step(title: str) -> None:
    print(f"\n── {title}", flush=True)


def venv_env() -> dict:
    """Environment equivalent to an activated venv (maturin looks for its binary on the PATH)."""
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV)
    env["PATH"] = str(PY.parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    return env


def run(cmd: list[str], **kw) -> None:
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=venv_env(), **kw)
    if r.returncode != 0:
        print(f"FAILED ({r.returncode}): {' '.join(map(str, cmd))}", flush=True)
        sys.exit(r.returncode)
    print(f"   ({time.perf_counter() - t:.1f} s)", flush=True)


def setup() -> None:
    if not PY.exists():
        step(f"creating {VENV.name} (Python {sys.version.split()[0]})")
        venv.create(VENV, with_pip=True)
    step("development dependencies")
    run([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip", *DEV_DEPS])
    build()


def build() -> None:
    if not PY.exists():
        return setup()
    step("building the Rust extension (pip install -e .)")
    run([str(PY), "-m", "pip", "install", "-q", "--no-build-isolation", "-e", "."])
    step("gpusync-gfx graphics engine (pip install -e gfx)")
    run([str(PY), "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", "gfx"])


def test() -> None:
    build()
    step("cargo test")
    run(["cargo", "test", "--workspace", "-q"])
    step("pytest")
    run([str(PY), "-m", "pytest", "-q", "-ra"])
    info()


def info() -> None:
    step("python -m gpusync")
    run([str(PY), "-m", "gpusync"])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        sys.exit(subprocess.run([str(PY), *sys.argv[2:]], cwd=ROOT, env=venv_env()).returncode)
    tasks = {"setup": setup, "build": build, "test": test, "info": info}
    if len(sys.argv) != 2 or sys.argv[1] not in tasks:
        sys.exit(__doc__)
    tasks[sys.argv[1]]()
