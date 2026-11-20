"""Sonde M6 : CUDA contre OpenCL et Vulkan sur le même GPU (RTX), à chaud, meilleur de 5."""
import sys, time
import numpy as np
import gpusync as gs
sys.path.insert(0, "tests")
from test_index import dtw, pairwise

@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std

def best(f, n=5):
    t = 1e9
    for _ in range(n):
        t0 = time.perf_counter(); np.asarray(f()); t = min(t, time.perf_counter() - t0)
    return t * 1e3

rng = np.random.default_rng(0)
small = rng.random(1000, dtype=np.float32)
big = rng.random(10_000_000, dtype=np.float32)
d400 = rng.random((400, 64), dtype=np.float32)
d4000 = rng.random((4000, 64), dtype=np.float32)
for dev in ["cuda:0", "opencl:0", "vulkan:0"]:
    np.asarray(normalize(small, 0.0, 1.0, device=dev)); np.asarray(dtw(d400[:8], 5, device=dev))
    t0 = time.perf_counter()
    for _ in range(200):
        np.asarray(normalize(small, 0.0, 1.0, device=dev))
    lat = (time.perf_counter() - t0) / 200 * 1e3
    print(f"{dev:9} appel 1000 élém. {lat:6.3f} ms | 10 M {best(lambda: normalize(big, 0.0, 1.0, device=dev)):6.1f} ms"
          f" | DTW 400 {best(lambda: dtw(d400, 5, device=dev)):6.1f} ms | DTW 4000 {best(lambda: dtw(d4000, 5, device=dev), 2):7.1f} ms", flush=True)
