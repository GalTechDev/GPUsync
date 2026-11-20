#!/usr/bin/env python3
"""Essais à gros volume du jalon M1.5 (ARCHITECTURE.md §9.8) : aucune limite de taille, mémoire bornée.

    python tools/bench_m15.py ram 4 opencl:0             # 4 Gio d'entrée en RAM
    python tools/bench_m15.py memmap 40 metal:0 /chemin  # 40 Gio en fichiers (entrée et sortie memmap)

Kernel : y = x * 2 + 1 (f32). Le résultat est vérifié **en entier**, bloc par bloc.
« mémoire en plus » = pic de mémoire physique du processus (résidente sur macOS et Linux, working set
sous Windows), moins la mémoire au démarrage, moins l'entrée et la sortie quand elles sont en RAM.
Précision : ± 0,2 Gio (temporaires de remplissage).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import gpusync as gs
from bench_m1 import peak_rss

BLOCK = 1 << 24  # éléments par bloc de remplissage et de vérification (64 Mio : temporaires limités)


@gs.kernel
def affine(x: gs.f32[:]) -> gs.f32[:]:
    return x * 2.0 + 1.0


def fill(x: np.ndarray) -> None:
    pattern = (np.arange(BLOCK, dtype=np.uint32) % 1021).astype(np.float32)
    for s in range(0, x.size, BLOCK):
        e = min(s + BLOCK, x.size)
        x[s:e] = pattern[: e - s] + np.float32(s // BLOCK % 7)


def verify(y: np.ndarray) -> int:
    pattern = (np.arange(BLOCK, dtype=np.uint32) % 1021).astype(np.float32)
    bad = 0
    for s in range(0, y.size, BLOCK):
        e = min(s + BLOCK, y.size)
        ref = (pattern[: e - s] + np.float32(s // BLOCK % 7)) * np.float32(2) + np.float32(1)
        bad += int(np.count_nonzero(y[s:e] != ref))
    return bad


def run(kind: str, gib: float, dev: str, folder: str | None) -> None:
    n = int(gib * (1 << 30)) // 4
    start = peak_rss()
    if kind == "memmap":
        folder = Path(folder or ".")
        xin, xout = folder / "gs_bench_in.f32", folder / "gs_bench_out.f32"
        x = np.memmap(xin, dtype=np.float32, mode="w+", shape=(n,))
        fill(x)
        x.flush()
        del x
        x = np.memmap(xin, dtype=np.float32, mode="r", shape=(n,))
        out = np.memmap(xout, dtype=np.float32, mode="w+", shape=(n,))
    else:
        x = np.empty(n, dtype=np.float32)
        fill(x)
        out = None
    t = time.perf_counter()
    y = affine(x, device=dev, out=out)
    res = y.numpy()
    if kind == "memmap":
        res.flush()
    dt = time.perf_counter() - t
    float(res[::1024].sum())  # toucher chaque page du résultat : les pages écrites par le GPU comptent alors
    extra = peak_rss() - start
    if kind == "ram":
        extra -= x.nbytes + res.nbytes  # entrée et sortie en RAM : les données elles-mêmes, pas un surcoût
    bad = verify(res)
    gbs = 2 * n * 4 / dt / 1e9
    ok = "✓" if bad == 0 else f"✗ {bad} éléments faux"
    print(f"  {dev:<9} {kind:<6} {gib:5.1f} Gio  mode {y.mode:<8} {dt:7.1f} s  {gbs:5.1f} Go/s  "
          f"mémoire en plus {extra / 2**30:5.2f} Gio  résultat {ok}", flush=True)
    if kind == "memmap":
        del x, out, y, res
        for f in (xin, xout):
            os.remove(f)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    run(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
