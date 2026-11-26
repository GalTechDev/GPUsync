"""M6 probe: where does the time of a CUDA call go (against OpenCL on the same GPU)?"""
import sys, time
import numpy as np
import gpusync as gs
sys.path.insert(0, "tests")
from test_index import dtw, pairwise

x = np.random.default_rng(0).random((400, 64), dtype=np.float32)
for dev in ["cuda:0", "opencl:0"]:
    np.asarray(dtw(x, 5, device=dev))
    for label, f in [("dtw 400x64", lambda: dtw(x, 5, device=dev)), ("pairwise 400x64", lambda: pairwise(x, device=dev))]:
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); y = f(); t1 = time.perf_counter(); y.numpy(); t2 = time.perf_counter()
            ts.append((t1 - t0, t2 - t1))
        a, b = min(ts, key=lambda t: t[0] + t[1])
        print(f"{dev:9} {label:16} call {a*1e3:7.2f} ms  read {b*1e3:7.2f} ms  mode {y.mode}")
    # small sizes: successive calls
    s = np.random.default_rng(1).random((20, 64), dtype=np.float32)
    np.asarray(dtw(s, 5, device=dev))
    t0 = time.perf_counter()
    for _ in range(50):
        np.asarray(dtw(s, 5, device=dev))
    print(f"{dev:9} dtw 20x64: {(time.perf_counter()-t0)/50*1e3:.3f} ms/call")
print(dtw.source("cuda")[:1500])
