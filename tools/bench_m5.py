"""Jalon M5 : mode explicite (atomiques) sur de gros volumes, comparé à NumPy.

    python tools/dev.py run tools/bench_m5.py [millions]

Histogramme (16 cases, atomiques entiers très disputés) et somme pondérée par case (1000 cases,
atomiques flottants émulés par comparaison-échange). Mesure à chaud, résultat lu ; vérifié contre NumPy.
"""

import sys
import time

import numpy as np

import gpusync as gs

BINS = 16
M = 1000


@gs.kernel(grid="data")
def histogram(data: gs.f32["n"], counts: gs.Out[gs.i32[BINS]]):
    i = gs.index()
    gs.atomic_add(counts, gs.min(int(data[i] * BINS), BINS - 1), 1)


@gs.kernel(grid="idx")
def scatter(idx: gs.i32["n"], w: gs.f32["n"], total: gs.Out[gs.f32[M]]):
    i = gs.index()
    gs.atomic_add(total, idx[i], w[i])


def best(f, repeat=3):
    t = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


def main():
    n = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 100_000_000
    rng = np.random.default_rng(0)
    data = rng.random(n, dtype=np.float32)
    idx = rng.integers(0, M, n).astype(np.int32)
    w = rng.random(n, dtype=np.float32)
    tn, ref_h = best(lambda: np.bincount(np.minimum((data * BINS).astype(np.int64), BINS - 1), minlength=BINS), 1)
    tw, ref_s = best(lambda: np.bincount(idx, w, minlength=M), 1)
    print(f"{n / 1e6:.0f} M éléments — NumPy : histogramme {tn * 1e3:.0f} ms, somme par case {tw * 1e3:.0f} ms")
    for d in gs.devices():
        if d.id == "cpu:0":
            continue
        histogram(data[:1000], device=d.id).numpy()
        scatter(idx[:1000], w[:1000], device=d.id).numpy()
        th, h = best(lambda: histogram(data, device=d.id).numpy())
        ts, s = best(lambda: scatter(idx, w, device=d.id).numpy())
        ok_h = "✓" if (h == ref_h).all() else "✗"
        ok_s = "✓" if np.allclose(s, ref_s, rtol=1e-3) else f"✗ {np.abs(s - ref_s).max()}"
        print(f"  {d.id:10} histogramme {th * 1e3:7.1f} ms ({n / th / 1e9:5.2f} G/s) {ok_h}   "
              f"somme par case (f32) {ts * 1e3:7.1f} ms ({n / ts / 1e9:5.2f} G/s) {ok_s}")


if __name__ == "__main__":
    main()
