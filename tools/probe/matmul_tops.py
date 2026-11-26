"""Probe: throughput of gs.matmul_u8 (Tensor Cores with CUDA, a portable loop elsewhere).

    python tools/probe/matmul_tops.py DEVICE [DEVICE…]

Each group of 256 threads chains REPS products of 64 × 64 × 128 (bytes) in shared memory.
"""
import sys
import time

import numpy as np

import gpusync as gs

M, N, K, REPS, GROUPS = 64, 64, 128, 200, 640


@gs.kernel(group=256)
def mm_bench(a: gs.u32["g", "ka"], b: gs.u32["g", "kb"], tmpl: gs.u32["g", "w"]) -> gs.u32["g", "w"]:
    i, j = gs.index()
    r = gs.lane()
    A = gs.shared(gs.u32, 2048)
    B = gs.shared(gs.u32, 2048)
    C = gs.shared(gs.u32, 4096)
    for t in range(r, 2048, 256):
        A[t] = a[i, t]
        B[t] = b[i, t]
    gs.barrier()
    for rep in range(REPS):
        gs.matmul_u8(C, A, B, 64, 64, 128)
    return C[j]


for dev in sys.argv[1:]:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1 << 32, (GROUPS, M * K // 4), dtype=np.uint32)
    b = rng.integers(0, 1 << 32, (GROUPS, N * K // 4), dtype=np.uint32)
    tmpl = np.zeros((GROUPS, 256), np.uint32)
    np.asarray(mm_bench(a, b, tmpl, device=dev))
    t = time.perf_counter()
    for _ in range(3):
        np.asarray(mm_bench(a, b, tmpl, device=dev))
    dt = (time.perf_counter() - t) / 3
    ops = 2 * M * N * K * REPS * GROUPS
    print(f"{dev:9} {dt*1e3:8.1f} ms par lancement, {ops / dt / 1e12:7.2f} TOPS (int8)", flush=True)
