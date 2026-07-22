import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.xmeans.xmeans_engine import xmeans_cpu, XMeansEngine
from gpusync import Scheduler, list_gpus

class XMeansBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 3):
        super().__init__(
            name="XMeans",
            description="Standard X-Means Multidimensional Clustering (N points, D=8, K_min=2, K_max=16)",
            num_runs=num_runs,
            variants=["GPUsync", "GPUsync-Genetic"]
        )
        self.dim = 8
        self.k_min = 2
        self.k_max = 16
        self.engine = XMeansEngine(dim=self.dim, max_k=self.k_max)

    def get_plot_metadata(self):
        return {
            "dim": self.dim,
            "k_min": self.k_min,
            "k_max": self.k_max
        }

    def setup_data(self, size: int):
        np.random.seed(42)
        X = np.random.randn(size, self.dim).astype(np.float32)
        return {
            "size": size,
            "X": X
        }

    def run_cpu(self, data):
        X = data["X"]
        labels, centroids = xmeans_cpu(X, k_min=self.k_min, k_max=self.k_max)
        return len(labels)

    def run_gpusync(self, data, scheduler, variant: str = "GPUsync"):
        X = data["X"]
        if variant == "GPUsync-Genetic":
            labels, centroids = self.engine.fit_genetic(scheduler, X, k_min=self.k_min, k_max=self.k_max, P=8)
        else:
            labels, centroids = self.engine.fit(scheduler, X, k_min=self.k_min, k_max=self.k_max)
        return len(labels)

def main():
    gpus = list_gpus()
    if not gpus:
        print("[!] Aucun GPU disponible pour le benchmark XMeans.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = XMeansBenchmark(num_runs=3)
    bm.execute(sizes=[10_000, 50_000, 100_000], scheduler=sched)

if __name__ == "__main__":
    main()
