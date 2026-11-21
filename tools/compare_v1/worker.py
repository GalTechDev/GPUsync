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

    # DTW 1 contre N : examples/algorithms/dtw/benchmark.py du prototype (recopié : ce module importe
    # numba pour sa référence CPU, absent ici).
    from gpusync import AsyncBuffer
    from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel

    BATCH_KERNEL = get_batch_dtw_kernel(32, 1)

    async def _chunk_manual(offset, chunk_size, gpu_refs, gpu_targets, pool, num_refs, out_matrix):
        gpu_out = await pool.get()
        await BATCH_KERNEL(gpu_refs, gpu_targets, 0, 32, gpu_out, num_refs, chunk_size,
                           global_size=(chunk_size,), scheduler=SCHED)
        res = await gpu_out.to_host()
        np.copyto(out_matrix[:, offset:offset + chunk_size],
                  np.sqrt(np.maximum(res[:num_refs * chunk_size].reshape(num_refs, chunk_size), 0.0)))
        pool.put_nowait(gpu_out)

    async def _manual_ooc(targets, refs, chunk_size=20000, in_flight=2):
        N, K, M = len(targets), len(refs), targets.shape[1]
        gpu_refs = await AsyncBuffer.to_device(SCHED, refs.flatten())
        pool = asyncio.Queue(maxsize=in_flight)
        for _ in range(in_flight):
            pool.put_nowait(AsyncBuffer.empty_like(SCHED, np.zeros(K * chunk_size, dtype=np.float32)))
        out_m = np.zeros((K, N), dtype=np.float32)
        active = set()
        flat = targets.flatten()
        for c in range((N + chunk_size - 1) // chunk_size):
            off = c * chunk_size
            size = min(chunk_size, N - off)
            gpu_t = await AsyncBuffer.to_device(SCHED, flat[off * M:(off + size) * M])
            active.add(asyncio.create_task(_chunk_manual(off, size, gpu_refs, gpu_t, pool, K, out_m)))
            while len(active) >= in_flight:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    t.result()
        while active:
            done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                t.result()
        return out_m

    def run_dtw1n(ref_pad, targets_pad):
        return {
            "OOC Manuel": lambda: asyncio.run(_manual_ooc(targets_pad, ref_pad.reshape(1, 32)))[0],
            "OOC Auto": lambda: asyncio.run(compute_dtw_matrix(targets_pad, ref_pad.reshape(1, 32), scheduler=SCHED))[0],
        }

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

    @gs.kernel
    def dtw_1n(ref: gs.f32["t"], x: gs.f32["n", "t"]) -> gs.f32["n"]:
        """DTW complet (sans fenêtre) de la série x[i] contre ref, ligne de la matrice en local."""
        i = gs.index()
        t = len(ref)
        row = gs.local(gs.f32, 64)
        d = ref[0] - x[i, 0]
        row[0] = d * d
        for j in range(1, t):
            d = ref[0] - x[i, j]
            row[j] = row[j - 1] + d * d
        for a in range(1, t):
            d = ref[a] - x[i, 0]
            diag = row[0]
            row[0] = row[0] + d * d
            for j in range(1, t):
                d = ref[a] - x[i, j]
                up = row[j]
                row[j] = gs.min(up, row[j - 1], diag) + d * d
                diag = up
        return gs.sqrt(gs.max(row[t - 1], 0.0))

    def run_dtw1n(ref_pad, targets_pad):
        return {"Rust": lambda: np.asarray(dtw_1n(ref_pad, targets_pad, device=DEVICE))}


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
        elif task == "dtw1n":
            # Comme le benchmark du prototype : 1 référence, séries de 30 points complétées à 32 par des
            # zéros (le DTW porte donc sur 32 points), N cibles.
            g = np.random.default_rng(7)
            ref_pad = np.zeros(32, dtype=np.float32)
            ref_pad[:30] = g.standard_normal(30)
            sizes = [int(v) for v in os.environ.get("DTW1N_SIZES", "100000,500000,1000000,2000000").split(",")]
            res = {}
            for n in sizes:
                tp = np.zeros((n, 32), dtype=np.float32)
                tp[:, :30] = g.standard_normal((n, 30), dtype=np.float32)
                for variant, f in run_dtw1n(ref_pad, tp).items():
                    f()  # préchauffage (compilation, buffers)
                    t, r = best(f, 3)
                    res.setdefault(variant, {})[n] = {"ms": t * 1e3, "sum": float(r.astype(np.float64).sum()),
                                                       "head": r[:3].astype(float).tolist()}
            out["dtw1n"] = res
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
