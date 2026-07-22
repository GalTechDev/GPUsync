import sys
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from numba import njit, prange

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel
from gpusync import AsyncBuffer, Scheduler, list_gpus

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

class DTWBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 5):
        super().__init__(
            name="DTW",
            description="Dynamic Time Warping 1-to-N distance computation using dtw_gpu_engine",
            num_runs=num_runs
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
            "targets_pad": targets_pad.flatten(),
            "num_targets": size
        }

    def run_cpu(self, data):
        ref = data["ref"]
        targets = data["targets"]
        N = len(targets)
        
        # Vrai calcul CPU jusqu'à 100 000. Capping au-delà pour projection.
        sample_size = min(N, 100_000)
        targets_sample = targets[:sample_size]
        
        results = np.zeros(sample_size, dtype=np.float32)
        cpu_dtw_batch(ref, targets_sample, results)
        return sample_size

    def run_gpusync(self, data, scheduler, variant: str = "GPUsync"):
        ref_pad = data["ref_pad"]
        targets_pad = data["targets_pad"]
        N = data["num_targets"]

        buf_ref = AsyncBuffer.to_device(scheduler, ref_pad)
        buf_tgt = AsyncBuffer.to_device(scheduler, targets_pad)
        buf_out = AsyncBuffer.empty_like(scheduler, np.zeros(1 * N, dtype=np.float32))

        self.kernel(
            buf_ref, buf_tgt, 0, 32, buf_out,
            1, N, global_size=(N,), scheduler=scheduler
        )
        return N

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = DTWBenchmark(num_runs=5)
    bm.execute(sizes=[10_000, 100_000, 200_000], scheduler=sched)

if __name__ == "__main__":
    main()
