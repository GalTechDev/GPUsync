#!/usr/bin/env python3
"""Measurements of milestone M1: overhead per call, throughput, peak memory (zero-copy).

    python tools/bench_m1.py latency            # every device
    python tools/bench_m1.py memory opencl:0    # peak memory on 1 GiB (one process per measurement)
    python tools/bench_m1.py all                # latency + memory of each device

The overhead is measured end to end: kernel call + reading the result (np.asarray), on
1,000 elements, where the computation itself is negligible. Prototype reference (M5 Pro, OpenCL):
3.07 ms per call; raw pyopencl: 0.30 ms. M1 goal: ≤ 0.3 ms.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

import gpusync as gs


@gs.kernel
def normalize(x: gs.f32[:], mean: gs.f32, std: gs.f32) -> gs.f32[:]:
    return (x - mean) / std


PEAK_KIND = "ram"  # Windows: "ram" (working set) or "commit" (committed memory)


def peak_rss() -> int:
    """Peak physical memory of the process, in bytes: resident memory (macOS, Linux), working set
    (Windows). On Windows, committed memory (PEAK_KIND = "commit") also counts the VRAM allocations
    reserved by the driver, which does not reflect the RAM used."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                          "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")
            ]

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE  # otherwise the 64-bit handle is truncated
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return pmc.PeakWorkingSetSize if PEAK_KIND == "ram" else pmc.PeakPagefileUsage
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # macOS : octets ; Linux : Kio


def latency(dev: str) -> None:
    x = np.random.rand(1000).astype(np.float32)
    for _ in range(50):
        np.asarray(normalize(x, 0.5, 2.0, device=dev))
    n = 2000
    t = time.perf_counter()
    for _ in range(n):
        np.asarray(normalize(x, 0.5, 2.0, device=dev))
    small = (time.perf_counter() - t) / n * 1e3

    big = np.random.rand(10_000_000).astype(np.float32)
    np.asarray(normalize(big, 0.5, 2.0, device=dev))
    reps = 3 if dev == "cpu:0" else 10
    t = time.perf_counter()
    for _ in range(reps):
        np.asarray(normalize(big, 0.5, 2.0, device=dev))
    big_ms = (time.perf_counter() - t) / reps * 1e3
    gbs = 2 * big.nbytes / (big_ms / 1e3) / 1e9  # read + write
    mark = "✓" if small <= 0.3 else "✗"
    print(f"  {dev:<10} 1,000 elements: {small:7.3f} ms/call {mark}   |   10 M elements: {big_ms:8.1f} ms ({gbs:5.1f} GB/s)", flush=True)


def memory(dev: str) -> None:
    size = 1 << 30  # 1 GiB of input
    info = next(d for d in gs.devices() if d.id == dev)
    if info.max_buffer_bytes is not None and info.max_buffer_bytes < size:
        print(f"  {dev:<10} 1 GiB exceeds the largest buffer size ({info.max_buffer_bytes / 2**30:.2f} GiB): "
              "clear error expected (out-of-core in M3)", flush=True)
        try:
            normalize(np.zeros(size // 4, np.float32), 0.0, 1.0, device=dev)
        except gs.UnsupportedFeature as e:
            print(f"             → gs.UnsupportedFeature: {str(e)[:90]}…", flush=True)
        return
    x = np.empty(size // 4, dtype=np.float32)
    x[:] = 1.0  # pages actually allocated
    base = peak_rss()
    y = normalize(x, 0.0, 2.0, device=dev)
    out = np.asarray(y)
    # Read the whole result: pages written by the GPU only count in the resident memory
    # once touched by the CPU. Honest measurement: the output counts for 1 GiB.
    assert float(out.sum(dtype=np.float64)) == 0.5 * out.size
    extra = peak_rss() - base
    print(f"  {dev:<10} input 1.00 GiB → extra peak {extra / size:4.2f} GiB "
          f"(output alone = 1.00) | zero-copy: {'yes' if y.zero_copy else 'no'}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    devs = [d.id for d in gs.devices()]
    if cmd == "memory" and len(sys.argv) > 2:
        memory(sys.argv[2])
    elif cmd in ("latency", "all"):
        print("Overhead per call (call + reading the result) and throughput:")
        for d in devs:
            latency(d)
        if cmd == "all":
            print("\nPeak memory (one process per device):", flush=True)
            for d in devs:
                if d != "cpu:0":
                    subprocess.run([sys.executable, __file__, "memory", d], check=False)
    else:
        sys.exit(__doc__)
