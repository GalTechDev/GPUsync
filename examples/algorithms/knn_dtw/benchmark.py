import sys
import os
import time
import asyncio
import numpy as np
from numba import njit, prange

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from gpusync import AsyncBuffer, Scheduler, list_gpus

# --- CPU BASELINE (Numba JIT) ---
@njit(parallel=True, fastmath=True)
def dtw_cpu_numba(query, timeseries, distances):
    M = len(query)
    N_windows = len(timeseries) - M + 1
    
    for i in prange(N_windows):
        row = np.zeros(M, dtype=np.float32)
        
        t_val_0 = timeseries[i]
        d = query[0] - t_val_0
        row[0] = d * d
        
        for j in range(1, M):
            d = query[j] - t_val_0
            row[j] = row[j - 1] + d * d
            
        for k in range(1, M):
            t_val = timeseries[i + k]
            d = query[0] - t_val
            
            m3 = row[0]
            row[0] = row[0] + d * d
            m2 = row[0]
            
            for j in range(1, M):
                d = query[j] - t_val
                m1 = row[j]
                
                min_val = min(m1, m2, m3)
                m3 = row[j]
                row[j] = min_val + d * d
                m2 = row[j]
                
        distances[i] = row[M - 1]

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

class KNNDTWBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 3):
        super().__init__(
            name="KNN-DTW",
            description="K-Nearest Neighbors Dynamic Time Warping window search",
            num_runs=num_runs,
            variants=["Single Query", "Multi-Query (Q=4)"]
        )

    def get_plot_metadata(self):
        return {
            "M (query)": 128,
            "K (voisins)": 5
        }

    def setup_data(self, size: int):
        M = 128
        query = np.random.randn(M).astype(np.float32)
        queries_multi = np.random.randn(4, M).astype(np.float32)
        time_series = np.random.randn(size).astype(np.float32)
        return {
            "query": query,
            "queries_multi": queries_multi,
            "time_series": time_series,
            "M": M,
            "size": size
        }

    def run_cpu(self, data):
        ts = data["time_series"]
        query = data["query"]
        sample_len = min(len(ts), 10_000)
        ts_sample = ts[:sample_len]
        
        distances = np.zeros(len(ts_sample) - len(query) + 1, dtype=np.float32)
        dtw_cpu_numba(query, ts_sample, distances)
        return len(distances)

    def run_gpusync(self, data, scheduler, variant: str):
        query = data["query"]
        queries_multi = data["queries_multi"]
        ts = data["time_series"]
        num_windows = len(ts) - len(query) + 1

        if variant == "Single Query":
            from examples.algorithms.knn_dtw.knn_dtw_engine import knn_dtw_search
            _ = run_async(knn_dtw_search(query, ts, K=5, scheduler=scheduler, chunk_size=100_000))
            return num_windows
        elif variant == "Multi-Query (Q=4)":
            from examples.algorithms.knn_dtw.knn_dtw_multi_engine import knn_dtw_multi_search
            _ = run_async(knn_dtw_multi_search(queries_multi, ts, K=5, scheduler=scheduler, chunk_size=100_000))
            return num_windows
        else:
            raise ValueError(f"Unknown variant: {variant}")

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = KNNDTWBenchmark(num_runs=2)
    bm.execute(sizes=[10_000, 50_000], scheduler=sched)

if __name__ == "__main__":
    main()
