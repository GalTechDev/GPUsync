"""Probe: cost of a call on the GPU, a chain y = f(y) against independent calls on the same input."""
import sys, time
import numpy as np
import gpusync as gs


@gs.kernel
def plus1(x: gs.u32[:]) -> gs.u32[:]:
    return x + 1


N = int(sys.argv[2]) if len(sys.argv) > 2 else 1 << 22
K = 200
for dev in sys.argv[1].split(","):
    x = np.zeros(N, np.uint32)
    x0 = plus1(x, device=dev)  # not read: stays on the device
    plus1(x0, device=dev).numpy()
    # 1. independent calls (fixed input on the GPU), reading the last one
    t = time.perf_counter()
    for _ in range(K):
        z = plus1(x0, device=dev)
    t1 = time.perf_counter(); z.numpy(); t2 = time.perf_counter()
    indep = ((t1 - t) / K, (t2 - t) / K)
    # 2. chain
    t = time.perf_counter(); z = x0
    for _ in range(K):
        z = plus1(z, device=dev)
    t1 = time.perf_counter(); v = int(z.numpy()[0]); t2 = time.perf_counter()
    chain = ((t1 - t) / K, (t2 - t) / K)
    # 3. read only
    t = time.perf_counter(); z.numpy(); read_only = time.perf_counter() - t
    t = time.perf_counter(); w = plus1(x0, device=dev); w.numpy(); single = time.perf_counter() - t
    gb = 2 * 4 * N / 1e9
    print(f"{dev:9} N={N:>9} | independent: submission {indep[0]*1e3:.3f} ms, total {indep[1]*1e3:.3f} ms/call ({gb/indep[1]:.0f} GB/s)"
          f" | chain: submission {chain[0]*1e3:.3f}, total {chain[1]*1e3:.3f} ms/call ({gb/chain[1]:.0f} GB/s) [{v}]"
          f" | 1 call+read {single*1e3:.2f} ms, read only (ready) {read_only*1e3:.2f} ms", flush=True)
