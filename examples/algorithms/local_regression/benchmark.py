import sys
import os
import time
import asyncio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.local_regression.loess_engine import loess_gpu_fit
from examples.algorithms.local_regression.demo_loess import generate_noisy_timeseries
from gpusync import Scheduler, list_gpus

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

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

class LOESSBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 3):
        super().__init__(
            name="LOESS",
            description="Local Regression (LOWESS) Smoothing",
            num_runs=num_runs,
            variants=["GPUsync LOESS Fit"]
        )

    def get_plot_metadata(self):
        return {
            "W (fenêtre)": 500
        }

    def setup_data(self, size: int):
        W = 500
        t, Y, _, _ = generate_noisy_timeseries(size)
        return {
            "t": t,
            "Y": Y.astype(np.float32),
            "W": W,
            "frac": (2 * W) / size
        }

    def run_cpu(self, data):
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels is not installed.")
        t = data["t"]
        Y = data["Y"]
        
        sample_size = min(len(t), 10_000)
        t_sample = t[:sample_size]
        Y_sample = Y[:sample_size]
        frac_sample = min(1.0, (2 * 500) / sample_size)
        
        sm.nonparametric.lowess(Y_sample, t_sample, frac=frac_sample, it=0)
        return sample_size

    def run_gpusync(self, data, scheduler, variant: str):
        Y = data["Y"]
        W = data["W"]
        res = run_async(loess_gpu_fit(Y, W=W, scheduler=scheduler))
        return len(res)

def main():
    gpus = list_gpus()
    if not gpus:
        print("Aucun GPU disponible.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = LOESSBenchmark(num_runs=2)
    bm.execute(sizes=[10_000, 25_000], scheduler=sched)

if __name__ == "__main__":
    main()
