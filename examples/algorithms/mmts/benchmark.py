# -*- coding: utf-8 -*-
"""
Benchmark standardisé pour MMTS (XmeansTS).
Compare l'exécution CPU séquentielle en Python Pur avec l'accélération GPU GPUsync.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from examples.common.benchmark_base import BaseBenchmark
from examples.algorithms.mmts.mmts_engine import XmeansTS
from gpusync import Scheduler, list_gpus
from config import ID_COLUMN

class MMTSBenchmark(BaseBenchmark):
    def __init__(self, num_runs: int = 2):
        super().__init__(
            name="MMTS (Time Series X-Means)",
            description="Clustering de séries temporelles avec X-Means et DTW contraint (W=5, K=3)",
            num_runs=num_runs,
            variants=["GPUsync"]
        )
        self.w = 5
        self.k = 3
        self.nb_min = 2

    def get_plot_metadata(self):
        return {
            "w": self.w,
            "k": self.k,
            "nb_min": self.nb_min
        }

    def setup_data(self, size: int):
        """
        Génère des séries temporelles synthétiques (ondes sinusoïdales avec bruit).
        """
        np.random.seed(42)
        M = 32  # Longueur de la série temporelle
        data = []
        for i in range(size):
            phase = np.random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4])
            t_seq = np.arange(M)
            sig = 20.0 + 5.0 * np.sin(t_seq * (2 * np.pi / 16) + phase) + np.random.randn(M) * 0.3
            data.append(sig.astype(np.float32))
            
        df_ids = pd.DataFrame({
            ID_COLUMN: [i for i in range(size)]
        })
        return {
            "data": data,
            "df_ids": df_ids
        }

    def run_cpu(self, data):
        """
        Référence CPU : Version séquentielle classique d'origine en Python Pur.
        """
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        sys.path.insert(0, os.path.join(root_dir, 'tests', 'mmts'))
        try:
            import xmeants_module as og_module
            XmeansTS_CPU = og_module.XmeansTS
        finally:
            sys.path.pop(0)

        ts_data = data["data"]
        # L'original requiert self.data comme numpy array (indexation booléenne interne)
        ts_data_np = np.array(ts_data, dtype=np.float32)
        df_ids = data["df_ids"]
        
        engine = XmeansTS_CPU(ts_data_np, self.k, "benchmark_run", df_ids.copy(), 1.0)
        result = engine.k_means_clust(
            ts_data_np, self.k, 1, df_ids.copy(), 0, "Set_TEMP", {}, [], self.nb_min, self.w
        )
        return len(result)

    def run_gpusync(self, data, scheduler, variant: str = "GPUsync"):
        """
        Version GPU : Version optimisée GPUsync OpenCL (sans fallback).
        """
        ts_data = data["data"]
        df_ids = data["df_ids"]
        
        # Initialisation du moteur MMTS en mode GPU exclusif
        engine = XmeansTS(ts_data, self.k, "benchmark_run", df_ids.copy(), 1.0, scheduler=scheduler)
        result = engine.k_means_clust(
            ts_data, self.k, 1, df_ids.copy(), 0, "Set_TEMP", {}, [], self.nb_min, self.w
        )
        return len(result)

def main():
    gpus = list_gpus()
    if not gpus:
        print("[!] Aucun GPU disponible pour le benchmark MMTS.")
        return

    sched = Scheduler(gpu=gpus[0])
    sched.initialize()

    bm = MMTSBenchmark(num_runs=2)
    # Tailles plus réduites adaptées à la lenteur du CPU Python Pur, montrant un speedup phénoménal
    bm.execute(sizes=[50, 100, 150], scheduler=sched)

if __name__ == "__main__":
    main()

