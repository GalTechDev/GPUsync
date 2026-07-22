import sys
import os
import time
import asyncio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.matrix_profile.stomp_engine import stomp_matrix_profile
from gpusync import Scheduler, list_gpus

try:
    import stumpy
    STUMPY_AVAILABLE = True
except ImportError:
    STUMPY_AVAILABLE = False

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

def generate_mp_data(N=50000, M=200):
    np.random.seed(42)
    T = np.cumsum(np.random.randn(N))
    motif = np.sin(np.linspace(0, 4*np.pi, M)) * 20
    if N > 40000:
        T[10000:10000+M] = motif
        T[35000:35000+M] = motif
        discord = np.exp(-np.linspace(-2, 2, M)**2) * 50
        T[25000:25000+M] = discord
    return T

class STOMPBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 3):
        super().__init__(
            name="STOMP",
            description="Matrix Profile Computation using STOMP",
            num_runs=num_runs,
            variants=["GPUsync STOMP Fit"]
        )

    def get_plot_metadata(self):
        return {
            "M (motif)": 200
        }

    def setup_data(self, size: int):
        M = 200
        T = generate_mp_data(size, M)
        return {
            "T": T,
            "M": M
        }

    def run_cpu(self, data):
        if not STUMPY_AVAILABLE:
            raise RuntimeError("stumpy is not installed.")
        T = data["T"]
        M = data["M"]
        
        sample_size = min(len(T), 5000)
        T_sample = T[:sample_size]
        
        stumpy.stump(T_sample, m=M)
        return sample_size

    def run_gpusync(self, data, scheduler, variant: str):
        T = data["T"]
        M = data["M"]
        res = run_async(stomp_matrix_profile(T, M, scheduler))
        return len(res)

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = STOMPBenchmark(num_runs=2)
    bm.execute(sizes=[10_000, 25_000], scheduler=sched)

if __name__ == "__main__":
    main()
