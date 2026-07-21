import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.matrix.matrix_gpu import matrix_mult_kernel
from gpusync import AsyncBuffer, Scheduler, list_gpus

class MatrixBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 5):
        super().__init__(
            name="MatrixMult",
            description="2D Dense Matrix Multiplication (N x N)",
            num_runs=num_runs
        )

    def setup_data(self, size: int):
        # size représente la dimension N de la matrice carrée N x N
        N = size
        A = np.random.randn(N, N).astype(np.float32).flatten()
        B = np.random.randn(N, N).astype(np.float32).flatten()
        return {
            "N": N,
            "A": A,
            "B": B
        }

    def run_cpu(self, data):
        N = data["N"]
        A_2d = data["A"].reshape(N, N)
        B_2d = data["B"].reshape(N, N)
        # Multiplication matricielle pure Python/Loop pour référence explicite
        C = np.dot(A_2d, B_2d)
        return C.shape[0]

    def run_gpusync(self, data, scheduler):
        N = data["N"]
        A = data["A"]
        B = data["B"]

        buf_a = AsyncBuffer.to_device(scheduler, A)
        buf_b = AsyncBuffer.to_device(scheduler, B)

        res = matrix_mult_kernel(buf_a, buf_b, N, global_size=(N * N,), scheduler=scheduler)
        return len(res)

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = MatrixBenchmark(num_runs=5)
    bm.execute(sizes=[128, 256, 512, 1024], scheduler=sched)

if __name__ == "__main__":
    main()
