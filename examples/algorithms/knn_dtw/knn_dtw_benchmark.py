import time
import asyncio
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from numba import njit, prange

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dtaidistance.dtw import distance_fast
    DTAIDISTANCE_AVAILABLE = True
except ImportError:
    DTAIDISTANCE_AVAILABLE = False

from knn_dtw_engine import knn_dtw_search
from gpusync.scheduler import Scheduler

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

def cpu_knn_dtw_search(query, timeseries):
    distances = np.zeros(len(timeseries) - len(query) + 1, dtype=np.float32)
    t0 = time.perf_counter()
    dtw_cpu_numba(query, timeseries, distances)
    t1 = time.perf_counter()
    return t1 - t0

def cpu_dtaidistance_search(query, timeseries):
    # Boucle en Python + C (Très lent à cause de l'overhead Python, mais c'est pour comparer)
    M = len(query)
    N_windows = len(timeseries) - M + 1
    
    # dtaidistance exige du float64 (double)
    q_double = query.astype(np.float64)
    ts_double = timeseries.astype(np.float64)
    
    t0 = time.perf_counter()
    for i in range(N_windows):
        distance_fast(q_double, ts_double[i:i+M])
    t1 = time.perf_counter()
    return t1 - t0

# --- BENCHMARK ---
async def main():
    print("=== Benchmark Comparatif : KNN-DTW GPU (gpusync) vs CPU (Numba JIT) ===")
    
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    
    M = 128
    query = np.random.randn(M).astype(np.float32)
    
    # Tailles de la série temporelle à tester
    sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000]
    
    gpu_times = []
    cpu_numba_times = []
    cpu_dtai_times = []
    
    # Warmup Numba compiler
    print("\n[Warm-up Numba compilation...]")
    dummy_ts = np.random.randn(200).astype(np.float32)
    cpu_knn_dtw_search(query, dummy_ts)
    
    for size in sizes:
        print(f"\n--- Test avec N = {size:,} points ---")
        timeseries = np.random.randn(size).astype(np.float32)
        
        # Test GPU
        t0_gpu = time.perf_counter()
        await knn_dtw_search(query, timeseries, K=5, scheduler=scheduler, chunk_size=1_000_000)
        t_gpu = time.perf_counter() - t0_gpu
        gpu_times.append(t_gpu)
        print(f"[GPU] Temps : {t_gpu:.4f} s | Débit : {(size/1e6)/t_gpu:.2f} M pts/s")
        
        # Test CPU Numba
        if size <= 1_000_000:
            t_cpu = cpu_knn_dtw_search(query, timeseries)
            cpu_numba_times.append(t_cpu)
            print(f"[CPU Numba] Temps : {t_cpu:.4f} s | Débit : {(size/1e6)/t_cpu:.2f} M pts/s")
            print(f"-> Speedup GPU vs CPU Numba : x{t_cpu/t_gpu:.1f}")
        else:
            print("[CPU Numba] Ignoré (Trop long pour CPU)")
            cpu_numba_times.append(None)
            
        # Test CPU dtaidistance
        if DTAIDISTANCE_AVAILABLE and size <= 100_000:
            t_dtai = cpu_dtaidistance_search(query, timeseries)
            cpu_dtai_times.append(t_dtai)
            print(f"[CPU DTAIDistance] Temps : {t_dtai:.4f} s | Débit : {(size/1e6)/t_dtai:.2f} M pts/s")
            print(f"-> Speedup GPU vs DTAIDistance : x{t_dtai/t_gpu:.1f}")
        else:
            if DTAIDISTANCE_AVAILABLE:
                print("[CPU DTAIDistance] Ignoré (Trop long / Explosion de l'overhead)")
            cpu_dtai_times.append(None)
            
    # Génération du graphe
    plt.figure(figsize=(10, 6))
    
    valid_sizes_numba = [s for s, t in zip(sizes, cpu_numba_times) if t is not None]
    valid_times_numba = [t for t in cpu_numba_times if t is not None]
    
    valid_sizes_dtai = [s for s, t in zip(sizes, cpu_dtai_times) if t is not None]
    valid_times_dtai = [t for t in cpu_dtai_times if t is not None]
    
    plt.plot(sizes, gpu_times, 'b-o', linewidth=2, label='GPU (gpusync + Unrolling)')
    plt.plot(valid_sizes_numba, valid_times_numba, 'r-s', linewidth=2, label='CPU (Numba JIT Multi-Core)')
    
    if DTAIDISTANCE_AVAILABLE:
        plt.plot(valid_sizes_dtai, valid_times_dtai, 'g-^', linewidth=2, label='CPU (DTAIDistance C-Fast)')
    
    plt.title('Performance KNN-DTW : GPU vs Numba vs DTAIDistance', fontsize=14)
    plt.xlabel('Taille de la série temporelle (Nombre de points)', fontsize=12)
    plt.ylabel('Temps d\'exécution (secondes)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=12)
    
    out_path = os.path.join(os.path.dirname(__file__), 'benchmark_plot.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n[SUCCES] Graphique de benchmark sauvegardé sous : {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
