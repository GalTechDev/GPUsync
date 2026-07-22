import sys
import os
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import math
from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.kmeans.kmeans_engine import get_kmeans_multi_kernel
from gpusync import AsyncBuffer, Scheduler, list_gpus

class KMeansBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 5):
        super().__init__(
            name="KMeans",
            description="K-Means Cluster Assignment Phase (N points, K=16, D=8) using kmeans_engine",
            num_runs=num_runs
        )
        self.p = 8
        self.k = 16
        self.dim = 8
        self.kernel = get_kmeans_multi_kernel(self.p, self.k, self.dim)

    def get_plot_metadata(self):
        return {
            "k (clusters)": 16,
            "dim": 8
        }

    def setup_data(self, size: int):
        X = np.random.randn(size, self.dim).astype(np.float32)
        X_soa = np.ascontiguousarray(X.T).ravel()
        centroids = np.random.randn(self.p, self.k, self.dim).astype(np.float32).flatten()
        return {
            "size": size,
            "X": X,
            "X_soa": X_soa,
            "centroids": centroids
        }

    def run_cpu(self, data):
        X = data["X"]
        centroids_all = data["centroids"].reshape(self.p, self.k, self.dim)
        centroids_2d = centroids_all[0]
        dists = np.linalg.norm(X[:, np.newaxis, :] - centroids_2d[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return len(labels)

    def run_gpusync(self, data, scheduler, variant: str = "GPUsync"):
        X_soa = data["X_soa"]
        centroids = data["centroids"]
        N = data["size"]

        buf_x = AsyncBuffer.to_device(scheduler, X_soa)
        buf_c = AsyncBuffer.to_device(scheduler, centroids)
        
        buf_sums = AsyncBuffer.empty_like(scheduler, np.zeros(self.p * self.k * self.dim, dtype=np.float32))
        buf_counts = AsyncBuffer.empty_like(scheduler, np.zeros(self.p * self.k, dtype=np.float32))
        buf_inertia = AsyncBuffer.empty_like(scheduler, np.zeros(self.p, dtype=np.float32))

        COARSE_FACTOR = 4
        local_sz_val = 256
        num_threads_needed = math.ceil(N / COARSE_FACTOR)
        global_aligned = math.ceil(num_threads_needed / local_sz_val) * local_sz_val

        self.kernel(
            buf_x, buf_c, buf_sums, buf_counts, buf_inertia,
            N, N, self.dim, self.k, self.p,
            global_size=(global_aligned,), local_size=(local_sz_val,),
            scheduler=scheduler
        )
        return N

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = KMeansBenchmark(num_runs=5)
    bm.execute(sizes=[50_000, 250_000, 500_000, 1_000_000], scheduler=sched)

if __name__ == "__main__":
    main()
