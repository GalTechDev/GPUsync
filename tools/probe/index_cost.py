"""Sonde : copie élément par élément contre copie avec index (1-D, 2-D), 64 × 65 536 u32."""
import sys, time
import numpy as np
import gpusync as gs


@gs.kernel
def copy_elem(x: gs.u32[:]) -> gs.u32[:]:
    return x


@gs.kernel
def copy_1d(x: gs.u32["n"]) -> gs.u32["n"]:
    i = gs.index()
    return x[i]


@gs.kernel
def copy_2d(x: gs.u32["b", "n"]) -> gs.u32["b", "n"]:
    i, j = gs.index()
    return x[i, j]


@gs.kernel
def neighbour_2d(x: gs.u32["b", "n"], h: gs.i32) -> gs.u32["b", "n"]:
    i, j = gs.index()
    return x[i, j ^ h]


for dev in sys.argv[1:]:
    x2 = np.random.default_rng(0).integers(0, 1 << 30, (64, 65536), dtype=np.uint32)
    x1 = x2.reshape(-1)
    for name, call, x in (("élément", lambda y: copy_elem(y, device=dev), x1),
                          ("index 1-D", lambda y: copy_1d(y, device=dev), x1),
                          ("index 2-D", lambda y: copy_2d(y, device=dev), x2),
                          ("2-D voisin j^h", lambda y: neighbour_2d(y, 1024, device=dev), x2)):
        y = call(x); np.asarray(y)
        t = time.perf_counter(); y = x
        for _ in range(100):
            y = call(y)
        np.asarray(y); dt = (time.perf_counter() - t) / 100
        print(f"{dev:9} {name:15} {dt*1e3:6.3f} ms  ({2 * x.nbytes / dt / 1e9:5.0f} Go/s)", flush=True)
