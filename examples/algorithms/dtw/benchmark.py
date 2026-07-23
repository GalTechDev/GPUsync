import sys
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from numba import njit, prange
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel
from gpusync import AsyncBuffer, Scheduler, list_gpus

# =========================================================================
# 1. NOYAUX ET WRAPPERS POUR OOC MANUEL
# =========================================================================
async def process_chunk_manual(chunk_id, offset, chunk_size, gpu_refs_all, gpu_targets_chunk, 
                              scheduler, pool, num_refs, out_matrix, batch_kernel):
    gpu_outputs_batch = await pool.get()
    await batch_kernel(
        gpu_refs_all, gpu_targets_chunk, 0, 32, gpu_outputs_batch,
        num_refs, chunk_size, global_size=(chunk_size,), scheduler=scheduler
    )
    res_flat = await gpu_outputs_batch.to_host()
    res_matrix = np.sqrt(np.maximum(res_flat[:num_refs * chunk_size].reshape(num_refs, chunk_size), 0.0))
    np.copyto(out_matrix[:, offset : offset + chunk_size], res_matrix)
    pool.put_nowait(gpu_outputs_batch)
    
    # Libération explicite immédiate du buffer OpenCL
    if hasattr(gpu_targets_chunk.cl_buffer, "release"):
        gpu_targets_chunk.cl_buffer.release()
        
    return chunk_size

async def run_manual_ooc(sched, targets_np, refs_np, chunk_size, in_flight_limit, batch_kernel):
    N = len(targets_np)
    K = len(refs_np)
    M = targets_np.shape[1]
    num_chunks = (N + chunk_size - 1) // chunk_size
    gpu_refs_all = await AsyncBuffer.to_device(sched, refs_np.flatten())
    pool = asyncio.Queue(maxsize=in_flight_limit)
    dummy_out = np.zeros(K * chunk_size, dtype=np.float32)
    for _ in range(in_flight_limit):
        pool.put_nowait(AsyncBuffer.empty_like(sched, dummy_out))
    matrice_distances = np.zeros((K, N), dtype=np.float32)
    active_tasks = set()
    targets_flat = targets_np.flatten()
    for chunk_id in range(num_chunks):
        offset = chunk_id * chunk_size
        curr_size = min(chunk_size, N - offset)
        chunk_data = targets_flat[offset * M : (offset + curr_size) * M]
        gpu_targets_chunk = await AsyncBuffer.to_device(sched, chunk_data)
        task = asyncio.create_task(process_chunk_manual(
            chunk_id, offset, curr_size,
            gpu_refs_all, gpu_targets_chunk,
            sched, pool, K, matrice_distances, batch_kernel
        ))
        active_tasks.add(task)
        while len(active_tasks) >= in_flight_limit:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for coro in done:
                coro.result()
    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            coro.result()
            
    # Nettoyage explicite des buffers du pool d'outputs
    while not pool.empty():
        buf = pool.get_nowait()
        if hasattr(buf.cl_buffer, "release"):
            buf.cl_buffer.release()
            
    if hasattr(gpu_refs_all.cl_buffer, "release"):
        gpu_refs_all.cl_buffer.release()
        
    return matrice_distances

# =========================================================================
# 2. LOGIQUE CPU
# =========================================================================
@njit(fastmath=True)
def cpu_dtw_single(ref, tgt):
    n = len(ref)
    dtw = np.zeros((32, 32), dtype=np.float32)
    dtw[0, 0] = (ref[0] - tgt[0]) ** 2
    for i in range(1, n):
        dtw[i, 0] = dtw[i - 1, 0] + (ref[i] - tgt[0]) ** 2
    for j in range(1, n):
        dtw[0, j] = dtw[0, j - 1] + (ref[0] - tgt[j]) ** 2
    for i in range(1, n):
        for j in range(1, n):
            cost = (ref[i] - tgt[j]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return np.sqrt(dtw[n - 1, n - 1])

@njit(parallel=True)
def cpu_dtw_batch(ref, targets, results):
    N = len(targets)
    for i in prange(N):
        results[i] = cpu_dtw_single(ref, targets[i])

# =========================================================================
# 3. BENCHMARK
# =========================================================================
class DTWBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 5):
        super().__init__(
            name="DTW",
            description="Dynamic Time Warping 1-to-N distance computation using dtw_gpu_engine",
            num_runs=num_runs,
            variants=["OOC Manuel", "OOC Auto"]
        )
        self.kernel = get_batch_dtw_kernel(32, 1)

    def get_plot_metadata(self):
        return {
            "n (longueur)": 30,
            "t (références)": 1,
            "s (cibles)": "N (10k à 2M)"
        }

    def setup_data(self, size: int):
        n = 30
        ref = np.random.randn(n).astype(np.float32)
        targets = np.random.randn(size, n).astype(np.float32)

        ref_pad = np.zeros(32, dtype=np.float32)
        ref_pad[:n] = ref

        targets_pad = np.zeros((size, 32), dtype=np.float32)
        targets_pad[:, :n] = targets

        return {
            "n": n,
            "ref": ref,
            "targets": targets,
            "ref_pad": ref_pad,
            "targets_pad": targets_pad,
            "num_targets": size
        }

    def run_cpu(self, data):
        ref = data["ref"]
        targets = data["targets"]
        N = len(targets)
        
        sample_size = min(N, 100_000)
        targets_sample = targets[:sample_size]
        
        results = np.zeros(sample_size, dtype=np.float32)
        cpu_dtw_batch(ref, targets_sample, results)
        return sample_size

    def run_gpusync(self, data, scheduler, variant: str = "OOC Manuel"):
        ref_pad = data["ref_pad"]
        targets_pad = data["targets_pad"]
        N = data["num_targets"]

        ref_pad_2d = ref_pad.reshape(1, 32)
        targets_pad_2d = targets_pad

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if variant == "OOC Auto":
            from examples.algorithms.dtw.dtw_gpu_engine import compute_dtw_matrix
            _ = loop.run_until_complete(compute_dtw_matrix(targets_pad_2d, ref_pad_2d, scheduler=scheduler))
            return N
        else:
            # Calibrage dynamique pour l'OOC Manuel de la même façon
            chunk_size = 50000 if N >= 100000 else 20000
            concurrency = 4 if N >= 100000 else 2
            _ = loop.run_until_complete(run_manual_ooc(scheduler, targets_pad_2d, ref_pad_2d, chunk_size, concurrency, self.kernel))
            return N

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = DTWBenchmark(num_runs=5)
    bm.execute(sizes=[100_000, 200_000, 300_000], scheduler=sched)

if __name__ == "__main__":
    main()
