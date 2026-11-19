#!/usr/bin/env python3
"""Mesures du jalon M1 : surcoût par appel, débit, pic mémoire (zéro-copie).

    python tools/bench_m1.py latence            # tous les périphériques
    python tools/bench_m1.py memoire opencl:0   # pic mémoire sur 1 Gio (un processus par mesure)
    python tools/bench_m1.py tout               # latence + mémoire de chaque périphérique

Le surcoût est mesuré de bout en bout : appel du kernel + lecture du résultat (np.asarray), sur
1 000 éléments, là où le calcul lui-même est négligeable. Référence prototype (M5 Pro, OpenCL) :
3,07 ms par appel ; pyopencl brut : 0,30 ms. Objectif M1 : ≤ 0,3 ms.
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


def peak_rss() -> int:
    """Pic de mémoire du processus, en octets : mémoire résidente (macOS, Linux), mémoire engagée
    (Windows, PeakPagefileUsage : le working set y est rogné par le système et ne compte pas les
    pages écrites par le GPU)."""
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
        k32.GetCurrentProcess.restype = wintypes.HANDLE  # sinon le handle 64 bits est tronqué
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            raise OSError("GetProcessMemoryInfo a échoué")
        return pmc.PeakPagefileUsage
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
    gbs = 2 * big.nbytes / (big_ms / 1e3) / 1e9  # lecture + écriture
    mark = "✓" if small <= 0.3 else "✗"
    print(f"  {dev:<10} 1 000 éléments : {small:7.3f} ms/appel {mark}   |   10 M éléments : {big_ms:8.1f} ms ({gbs:5.1f} Go/s)", flush=True)


def memory(dev: str) -> None:
    size = 1 << 30  # 1 Gio d'entrée
    info = next(d for d in gs.devices() if d.id == dev)
    if info.max_buffer_bytes is not None and info.max_buffer_bytes < size:
        print(f"  {dev:<10} 1 Gio dépasse la taille max d'un buffer ({info.max_buffer_bytes / 2**30:.2f} Gio) : "
              "erreur claire attendue (out-of-core en M3)", flush=True)
        try:
            normalize(np.zeros(size // 4, np.float32), 0.0, 1.0, device=dev)
        except gs.UnsupportedFeature as e:
            print(f"             → gs.UnsupportedFeature : {str(e)[:90]}…", flush=True)
        return
    x = np.empty(size // 4, dtype=np.float32)
    x[:] = 1.0  # pages réellement allouées
    base = peak_rss()
    y = normalize(x, 0.0, 2.0, device=dev)
    out = np.asarray(y)
    # Lire tout le résultat : les pages écrites par le GPU ne comptent dans la mémoire
    # résidente qu'une fois touchées par le CPU. Mesure honnête : la sortie compte pour 1 Gio.
    assert float(out.sum(dtype=np.float64)) == 0.5 * out.size
    extra = peak_rss() - base
    print(f"  {dev:<10} entrée 1,00 Gio → pic supplémentaire {extra / size:4.2f} Gio "
          f"(sortie seule = 1,00) | zéro-copie : {'oui' if y.zero_copy else 'non'}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tout"
    devs = [d.id for d in gs.devices()]
    if cmd == "memoire" and len(sys.argv) > 2:
        memory(sys.argv[2])
    elif cmd in ("latence", "tout"):
        print("Surcoût par appel (appel + lecture du résultat) et débit :")
        for d in devs:
            latency(d)
        if cmd == "tout":
            print("\nPic mémoire (un processus par périphérique) :", flush=True)
            for d in devs:
                if d != "cpu:0":
                    subprocess.run([sys.executable, __file__, "memoire", d], check=False)
    else:
        sys.exit(__doc__)
