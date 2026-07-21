import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.dtw.dtw_gpu import batch_dtw_kernel, cpu_dtw_single
from gpusync import AsyncBuffer, Scheduler, list_gpus

class DTWBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 5):
        super().__init__(
            name="DTW",
            description="Dynamic Time Warping 1-to-N distance computation",
            num_runs=num_runs
        )

    def setup_data(self, size: int):
        seq_length = 30
        ref = np.random.randn(seq_length).astype(np.float32)
        targets = np.random.randn(size, seq_length).astype(np.float32).flatten()
        return {
            "seq_length": seq_length,
            "ref": ref,
            "targets": targets,
            "num_targets": size
        }

    def run_cpu(self, data):
        seq_len = data["seq_length"]
        ref = data["ref"]
        targets_2d = data["targets"].reshape(-1, seq_len)
        # Échantillonne max 500 séries pour le CPU pour éviter un blocage prolongé
        sample_size = min(len(targets_2d), 200)
        res = [cpu_dtw_single(ref, targets_2d[i]) for i in range(sample_size)]
        return len(res)

    def run_gpusync(self, data, scheduler):
        ref = data["ref"]
        targets = data["targets"]
        seq_len = data["seq_length"]
        N = data["num_targets"]

        buf_ref = AsyncBuffer.to_device(scheduler, ref)
        buf_tgt = AsyncBuffer.to_device(scheduler, targets)

        _ = batch_dtw_kernel(buf_ref, buf_tgt, 0, seq_len, global_size=(N,), scheduler=scheduler)
        return N

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = DTWBenchmark(num_runs=5)
    bm.execute(sizes=[10_000, 50_000, 100_000, 250_000], scheduler=sched)

if __name__ == "__main__":
    main()
