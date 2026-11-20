#!/usr/bin/env python3
"""Tâches de développement, identiques sur macOS, Linux et Windows.

    python tools/dev.py setup     # crée .venv, installe maturin et pytest, puis `pip install -e .`
    python tools/dev.py build     # recompile l'extension Rust dans .venv (pip install -e .)
    python tools/dev.py test      # build + cargo test + pytest
    python tools/dev.py info      # python -m gpusync (diagnostic des périphériques)
    python tools/dev.py run SCRIPT [ARGS…]   # lance un script avec le Python de .venv

Avec gs-remote, sur toutes les machines de test :

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
DEV_DEPS = ["maturin>=1.16,<2", "pytest>=8", "numpy>=1.22", "nbclient>=0.10", "ipykernel>=6"]  # Jupyter : test M4


def step(title: str) -> None:
    print(f"\n── {title}", flush=True)


def venv_env() -> dict:
    """Environnement équivalent à un venv activé (maturin cherche son binaire dans le PATH)."""
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(VENV)
    env["PATH"] = str(PY.parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    return env


def run(cmd: list[str], **kw) -> None:
    t = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=venv_env(), **kw)
    if r.returncode != 0:
        print(f"ÉCHEC ({r.returncode}) : {' '.join(map(str, cmd))}", flush=True)
        sys.exit(r.returncode)
    print(f"   ({time.perf_counter() - t:.1f} s)", flush=True)


def setup() -> None:
    if not PY.exists():
        step(f"création de {VENV.name} (Python {sys.version.split()[0]})")
        venv.create(VENV, with_pip=True)
    step("dépendances de développement")
    run([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip", *DEV_DEPS])
    build()


def build() -> None:
    if not PY.exists():
        return setup()
    step("compilation de l'extension Rust (pip install -e .)")
    run([str(PY), "-m", "pip", "install", "-q", "--no-build-isolation", "-e", "."])


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
