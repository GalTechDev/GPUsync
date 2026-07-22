import sys
import os
import time
import asyncio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.kshape.kshape_engine import kshape_gpu_fit
from examples.algorithms.kshape.demo_kshape import generate_kshape_dataset
from gpusync import Scheduler, list_gpus

try:
    from tslearn.clustering import KShape
    TSLEARN_AVAILABLE = True
except ImportError:
    TSLEARN_AVAILABLE = False

def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
    return loop.run_until_complete(coro)

class KShapeBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 3):
        super().__init__(
            name="KShape",
            description="KShape Time Series Clustering",
            num_runs=num_runs,
            variants=["GPUsync Hybrid Fit"]
        )

    def get_plot_metadata(self):
        return {
            "K (clusters)": 3,
            "M (longueur)": 128
        }

    def setup_data(self, size: int):
        M = 128
        K = 3
        X, true_labels = generate_kshape_dataset(size, M)
        return {
            "X": X.astype(np.float32),
            "K": K,
            "size": size,
            "M": M
        }

    def run_cpu(self, data):
        if not TSLEARN_AVAILABLE:
            raise RuntimeError("tslearn is not installed.")
        X = data["X"]
        K = data["K"]
        
        # Limite de taille pour le benchmark CPU
        sample_size = min(len(X), 500)
        X_sample = X[:sample_size]
        
        ts_ks = KShape(n_clusters=K, max_iter=5, n_init=1, random_state=42)
        _ = ts_ks.fit_predict(X_sample)
        return sample_size

    def run_gpusync(self, data, scheduler, variant: str):
        X = data["X"]
        K = data["K"]
        centroids_gpu, labels_gpu, shifts_gpu = run_async(kshape_gpu_fit(X, K, scheduler, max_iter=5))
        return len(labels_gpu)

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = KShapeBenchmark(num_runs=2)
    bm.execute(sizes=[100, 500], scheduler=sched)

if __name__ == "__main__":
    main()
