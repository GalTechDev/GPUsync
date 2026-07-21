import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.kmeans.kmeans_gpu import assign_clusters_kernel
from gpusync import AsyncBuffer, Scheduler, list_gpus

class KMeansBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 5):
        super().__init__(
            name="KMeans",
            description="K-Means Cluster Assignment Phase (N points, K=16, D=8)",
            num_runs=num_runs
        )

    def setup_data(self, size: int):
        k = 16
        dim = 8
        X = np.random.randn(size, dim).astype(np.float32).flatten()
        centroids = np.random.randn(k, dim).astype(np.float32).flatten()
        return {
            "size": size,
            "k": k,
            "dim": dim,
            "X": X,
            "centroids": centroids
        }

    def run_cpu(self, data):
        X_2d = data["X"].reshape(-1, data["dim"])
        centroids_2d = data["centroids"].reshape(-1, data["dim"])
        # Distances euclidiennes broadcastées NumPy
        dists = np.linalg.norm(X_2d[:, np.newaxis, :] - centroids_2d[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return len(labels)

    def run_gpusync(self, data, scheduler):
        X = data["X"]
        centroids = data["centroids"]
        k = data["k"]
        dim = data["dim"]
        N = data["size"]

        buf_x = AsyncBuffer.to_device(scheduler, X)
        buf_c = AsyncBuffer.to_device(scheduler, centroids)

        labels = assign_clusters_kernel(buf_x, buf_c, k, dim, global_size=(N,), scheduler=scheduler)
        return len(labels)

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
