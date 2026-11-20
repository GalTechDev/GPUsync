"""Jalon M3 : kernels avec index découpés par blocs de lignes, sur de gros volumes.

    python tools/dev.py run tools/bench_m3.py pairwise N T DEVICE   # sortie N×N (f32)
    python tools/dev.py run tools/bench_m3.py dtw N T DEVICE

Affiche le mode, la durée (lecture du résultat comprise), la mémoire en plus (pic − entrée −
sortie) et vérifie 64 lignes tirées au hasard contre NumPy.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import gpusync as gs
from bench_m1 import peak_rss

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_index import dtw, dtw_ref, pairwise  # noqa: E402


def run(what: str, n: int, t: int, dev: str) -> None:
    rng = np.random.default_rng(7)
    x = rng.random((n, t), dtype=np.float32)
    start = peak_rss()
    t0 = time.perf_counter()
    y = pairwise(x, device=dev) if what == "pairwise" else dtw(x, 5, device=dev)
    res = y.numpy()
    dt = time.perf_counter() - t0
    float(res[:, ::1024].sum())  # toucher chaque page du résultat
    extra = peak_rss() - start - res.nbytes
    rows = np.sort(rng.choice(n, size=min(64, n), replace=False))
    if what == "pairwise":
        x64 = x.astype(np.float64)
        ref = np.sqrt(((x64[rows, None, :] - x64[None, :, :]) ** 2).sum(-1))
        bad = int((~np.isclose(res[rows], ref, rtol=1e-4, atol=1e-5)).sum())
    else:
        ref = dtw_ref(x[rows], 5)
        bad = int((~np.isclose(res[np.ix_(rows, rows)], ref, rtol=1e-4, atol=1e-5)).sum())
    ok = "✓" if bad == 0 else f"✗ {bad} écarts"
    print(f"  {dev:<9} {what} {n}×{n} (t={t}) sortie {res.nbytes / 2**30:5.2f} Gio  mode {y.mode:<8} "
          f"{dt:7.1f} s  mémoire en plus {extra / 2**30:5.2f} Gio  vérification {ok}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
