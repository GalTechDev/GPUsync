#!/usr/bin/env python3
"""Large-volume trials of milestone M1.5 (ARCHITECTURE.md §9.8): no size limit, bounded memory.

    python tools/bench_m15.py ram 4 opencl:0             # 4 GiB of input in RAM
    python tools/bench_m15.py memmap 40 metal:0 /path    # 40 GiB in files (memmap input and output)

Kernel: y = x * 2 + 1 (f32). The result is checked **in full**, chunk by chunk.
"extra memory" = peak physical memory of the process (resident on macOS and Linux, working set
on Windows), minus the memory at startup, minus the input and output when they are in RAM.
Precision: ± 0.2 GiB (filling temporaries).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import gpusync as gs
from bench_m1 import peak_rss

BLOCK = 1 << 24  # elements per filling and checking chunk (64 MiB: limited temporaries)


@gs.kernel
def affine(x: gs.f32[:]) -> gs.f32[:]:
    return x * 2.0 + 1.0


def fill(x: np.ndarray) -> None:
    pattern = (np.arange(BLOCK, dtype=np.uint32) % 1021).astype(np.float32)
    for s in range(0, x.size, BLOCK):
        e = min(s + BLOCK, x.size)
        x[s:e] = pattern[: e - s] + np.float32(s // BLOCK % 7)


def verify(y: np.ndarray) -> int:
    pattern = (np.arange(BLOCK, dtype=np.uint32) % 1021).astype(np.float32)
    bad = 0
    for s in range(0, y.size, BLOCK):
        e = min(s + BLOCK, y.size)
        ref = (pattern[: e - s] + np.float32(s // BLOCK % 7)) * np.float32(2) + np.float32(1)
        bad += int(np.count_nonzero(y[s:e] != ref))
    return bad


def run(kind: str, gib: float, dev: str, folder: str | None) -> None:
    n = int(gib * (1 << 30)) // 4
    start = peak_rss()
    if kind == "memmap":
        folder = Path(folder or ".")
        xin, xout = folder / "gs_bench_in.f32", folder / "gs_bench_out.f32"
        x = np.memmap(xin, dtype=np.float32, mode="w+", shape=(n,))
        fill(x)
        x.flush()
        del x
        x = np.memmap(xin, dtype=np.float32, mode="r", shape=(n,))
        out = np.memmap(xout, dtype=np.float32, mode="w+", shape=(n,))
    else:
        x = np.empty(n, dtype=np.float32)
        fill(x)
        out = None
    t = time.perf_counter()
    y = affine(x, device=dev, out=out)
    res = y.numpy()
    if kind == "memmap":
        res.flush()
    dt = time.perf_counter() - t
    float(res[::1024].sum())  # touch every page of the result: pages written by the GPU then count
    extra = peak_rss() - start
    if kind == "ram":
        extra -= x.nbytes + res.nbytes  # input and output in RAM: the data itself, not an overhead
    bad = verify(res)
    gbs = 2 * n * 4 / dt / 1e9
    ok = "✓" if bad == 0 else f"✗ {bad} wrong elements"
    print(f"  {dev:<9} {kind:<6} {gib:5.1f} Gio  mode {y.mode:<8} {dt:7.1f} s  {gbs:5.1f} Go/s  "
          f"extra memory {extra / 2**30:5.2f} GiB  result {ok}", flush=True)
    if kind == "memmap":
        del x, out, y, res
        for f in (xin, xout):
            os.remove(f)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    run(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
