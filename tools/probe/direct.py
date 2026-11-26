"""Probe: unified memory (Direct mode), fresh output at each call against a reused output (out=)."""
import sys, time
import numpy as np
import gpusync as gs


@gs.kernel
def plus1(x: gs.u32[:]) -> gs.u32[:]:
    return x + 1


N = 1 << 22
K = 100
for dev in sys.argv[1].split(","):
    x = np.zeros(N, np.uint32)
    plus1(x, device=dev).numpy()
    t = time.perf_counter()
    for _ in range(K):
        z = plus1(x, device=dev)
    z.numpy(); fresh = (time.perf_counter() - t) / K
    o = np.empty(N, np.uint32)
    plus1(x, device=dev, out=o).numpy()
    t = time.perf_counter()
    for _ in range(K):
        z = plus1(x, device=dev, out=o)
        z.numpy()
    reused = (time.perf_counter() - t) / K
    t = time.perf_counter()
    for _ in range(K):
        a = np.empty(N, np.uint32); a[::1024] = 1
    page_faults = (time.perf_counter() - t) / K
    print(f"{dev:9} mode {z.mode}: fresh output {fresh*1e3:.3f} ms, reused out= {reused*1e3:.3f} ms; "
          f"np.empty + touching the pages {page_faults*1e3:.3f} ms", flush=True)
