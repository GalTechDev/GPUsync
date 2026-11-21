"""Sonde : mémoire unifiée, un appel tableau NumPy → résultat lu, selon la taille."""
import sys, time
import numpy as np
import gpusync as gs


@gs.kernel
def plus1(x: gs.u32[:]) -> gs.u32[:]:
    return x + 1


@gs.kernel
def heavy(x: gs.f32[:]) -> gs.f32[:]:
    s = x
    for k in range(64):
        s = s * 0.999 + 0.001
    return s


for dev in sys.argv[1].split(","):
    for mb in (16, 256, 1024):
        n = mb << 18
        for name, f, dt in (("x+1", plus1, np.uint32), ("64 fma", heavy, np.float32)):
            x = np.ones(n, dt)
            f(x, device=dev).numpy()
            ts = []
            for _ in range(5):
                t = time.perf_counter(); y = f(x, device=dev); y.numpy(); ts.append(time.perf_counter() - t)
            print(f"{dev:9} {mb:5} Mo {name:7} : {min(ts)*1e3:8.2f} ms (mode {y.mode})", flush=True)
