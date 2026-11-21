"""Un processus de mesure pour une version de GPUsync (v1 = prototype Python, v2 = Rust).

    python worker.py v1|v2 BACKEND TÂCHE...     (sortie : une ligne JSON par tâche)

Les deux versions s'appellent `gpusync` : chacune tourne dans son propre processus (et son propre
environnement Python pour v1). Mêmes données (graine fixe), mêmes calculs, même GPU.
"""

import asyncio
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMPL, BACKEND, TASKS = sys.argv[1], sys.argv[2], sys.argv[3:]


def peak_rss() -> int:
    """Pic de mémoire physique du processus (octets)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE  # sinon le handle 64 bits est tronqué
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = PMC()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise OSError("GetProcessMemoryInfo a échoué")
        return c.PeakWorkingSetSize
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


# ───────────────────────────── v1 : prototype ─────────────────────────────
if IMPL == "v1":
    sys.path[:0] = [os.path.join(ROOT, "GPUsync-python", "src"), os.path.join(ROOT, "GPUsync-python")]
    t_import = time.perf_counter()
    from gpusync import Scheduler, gpusync_kernel, list_gpus
    from examples.algorithms.dtw.dtw_gpu_engine import compute_dtw_matrix
    t_import = time.perf_counter() - t_import

    backend, _, name_part = BACKEND.partition("/")  # ex. "opencl/NVIDIA" : GPU dont le nom contient NVIDIA
    gpus = [g for g in list_gpus() if g.backend == backend and name_part.lower() in g.name.lower()]
    if not gpus:
        print(json.dumps({"error": f"v1 : aucun GPU {BACKEND}"}))
        sys.exit(0)
    SCHED = Scheduler(gpu=gpus[0])
    SCHED.initialize()
    DEVICE_NAME = gpus[0].name

    @gpusync_kernel
    def scale(ctx: int, data: list) -> tuple:
        return data[ctx] * 3.0 + 5.0

    def run_scale(x):
        return np.asarray(scale(x, global_size=(len(x),), scheduler=SCHED))

    def run_dtw(series, w):
        async def go():
            rows = []
            for i in range(0, len(series), 16):  # comme tests/run_xlsx_dtw.py du prototype
                rows.append(await compute_dtw_matrix(series, series[i:i + 16], scheduler=SCHED, w=w))
            return np.vstack(rows)
        return asyncio.run(go())

# ───────────────────────────── v2 : Rust ─────────────────────────────
else:
    t_import = time.perf_counter()
    import gpusync as gs
    t_import = time.perf_counter() - t_import
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_index import dtw as dtw_kernel

    DEVICE = BACKEND
    DEVICE_NAME = next(d.name for d in gs.devices() if d.id == DEVICE)

    @gs.kernel
    def scale(x: gs.f32[:]) -> gs.f32[:]:
        return x * 3.0 + 5.0

    def run_scale(x):
        return np.asarray(scale(x, device=DEVICE))

    def run_dtw(series, w):
        return np.asarray(dtw_kernel(series, w, device=DEVICE))


def best(f, n):
    t = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        r = f()
        t = min(t, time.perf_counter() - t0)
    return t, r


rng = np.random.default_rng(42)
out = {"impl": IMPL, "backend": BACKEND, "import_s": t_import}
for task in TASKS:
    try:
        if task == "lat":
            x = rng.random(1000, dtype=np.float32)
            t0 = time.perf_counter()
            run_scale(x)  # premier appel : compilation comprise
            out["first_call_ms"] = (time.perf_counter() - t0) * 1e3
            for _ in range(20):
                run_scale(x)
            t0 = time.perf_counter()
            for _ in range(200):
                r = run_scale(x)
            out["call_ms"] = (time.perf_counter() - t0) / 200 * 1e3
            out["lat_ok"] = bool(np.allclose(r, x * 3 + 5))
        elif task == "big":
            x = rng.random(10_000_000, dtype=np.float32)
            t, r = best(lambda: run_scale(x), 5)
            out["big_ms"] = t * 1e3
            out["big_ok"] = bool(np.allclose(r, x * 3 + 5))
        elif task == "huge":
            n = (1 << 30) // 4
            base = peak_rss()  # avant l'entrée : surcoût = pic − départ − entrée − sortie
            x = np.empty(n, dtype=np.float32)
            x[:] = 1.5
            t0 = time.perf_counter()
            r = run_scale(x)
            out["huge_s"] = time.perf_counter() - t0
            float(r[::1024].sum())  # toucher les pages du résultat
            out["huge_extra_gib"] = (peak_rss() - base - x.nbytes - r.nbytes) / 2**30
            out["huge_ok"] = bool((r[:: 4096] == 9.5).all())
        elif task == "dtw391":
            s = np.ascontiguousarray(np.load(os.path.join(ROOT, "tests", "data", "oxygene_series.npy")))
            ref = np.load(os.path.join(ROOT, "tests", "data", "oxygene_dtw_w5_prototype.npy"))
            run_dtw(s, 5)
            t, r = best(lambda: run_dtw(s, 5), 3)
            out["dtw391_ms"] = t * 1e3
            out["dtw391_err"] = float(np.abs(r - ref).max())
        elif task == "dtw2000":
            s = rng.random((2000, 64), dtype=np.float32)
            run_dtw(s[:16], 5)
            t, r = best(lambda: run_dtw(s, 5), 2)
            out["dtw2000_ms"] = t * 1e3
            out["dtw2000_sum"] = float(r.astype(np.float64).sum())
            out["dtw2000_corner"] = r[:4, :4].astype(float).tolist()
    except Exception as e:  # une tâche en échec n'empêche pas les autres
        out[f"{task}_error"] = f"{type(e).__name__}: {e}"[:300]
out["device"] = str(DEVICE_NAME)
print("RESULT " + json.dumps(out), flush=True)
