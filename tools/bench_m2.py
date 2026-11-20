"""Jalon M2 : temps du DTW (kernel avec index, tableaux locaux) sur chaque périphérique.

    python tools/dev.py run tools/bench_m2.py [n_series longueur]

Mesure à chaud (kernel déjà compilé), résultat lu sur l'hôte ; vérifie l'accord avec cpu:0 sur
un échantillon de lignes.
"""

import sys
import time
from pathlib import Path

import numpy as np

import gpusync as gs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_index import dtw, dtw_ref  # noqa: E402


def bench(x, w=5, repeat=3):
    print(f"DTW {x.shape[0]}×{x.shape[0]} paires, séries de {x.shape[1]}, fenêtre {w}")
    ref = dtw_ref(x[:8], w)
    for d in gs.devices():
        if d.id == "cpu:0" and x.shape[0] > 400:
            continue
        np.asarray(dtw(x[:8], w, device=d.id))  # compilation
        best = float("inf")
        try:
            for _ in range(repeat):
                t0 = time.perf_counter()
                y = np.asarray(dtw(x, w, device=d.id))
                best = min(best, time.perf_counter() - t0)
        except gs.GpusyncError as e:
            print(f"  {d.id:10} {type(e).__name__} : {e}")
            # Le périphérique doit rester utilisable après l'erreur (rouvert si perdu).
            check = np.asarray(dtw(x[:8], w, device=d.id))
            print(f"  {d.id:10} après l'erreur : petit appel {'correct' if np.allclose(check, ref, atol=1e-5) else 'FAUX'}")
            continue
        err = np.abs(y[:8, :8] - ref).max()
        pairs = x.shape[0] ** 2 / best / 1e6
        print(f"  {d.id:10} {best * 1000:9.1f} ms  {pairs:8.2f} M paires/s  écart {err:.1e}")


if __name__ == "__main__":
    bench(np.load(Path(__file__).resolve().parents[1] / "tests/data/oxygene_series.npy"))
    n, t = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (4000, 64)
    bench(np.random.default_rng(0).random((n, t), dtype=np.float32))
