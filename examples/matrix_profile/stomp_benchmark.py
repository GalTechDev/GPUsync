import os
import sys
import time
import numpy as np
import asyncio
import matplotlib.pyplot as plt

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stomp_engine import stomp_matrix_profile
from gpusync.scheduler import Scheduler

try:
    import stumpy
except ImportError:
    print("Veuillez installer stumpy pour ce benchmark (pip install stumpy)")
    sys.exit(1)

def generate_mp_data(N=50000, M=200):
    np.random.seed(42)
    T = np.cumsum(np.random.randn(N))
    # Inject motifs and discords safely (only if N is large enough)
    motif = np.sin(np.linspace(0, 4*np.pi, M)) * 20
    if N > 40000:
        T[10000:10000+M] = motif
        T[35000:35000+M] = motif
        discord = np.exp(-np.linspace(-2, 2, M)**2) * 50
        T[25000:25000+M] = discord
    return T

async def run_stomp_benchmark_sizes():
    np.random.seed(42)
    print("=== BENCHMARK SCALING MATRIX PROFILE (STOMP) : CPU vs GPU ===")
    
    sizes = [10_000, 25_000, 50_000, 100_000, 150_000]
    M = 200  # Motif size constant
    
    cpu_times = []
    gpu_times = []
    
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    
    print("\n[0] Phase de 'Warmup' (Chauffe JIT Numba et OpenCL) ...")
    # Tour non enregistré pour compiler Stumpy (Numba) et GPUsync (OpenCL)
    T_warmup = np.random.randn(2000)
    stumpy.stump(T_warmup, m=200)
    await stomp_matrix_profile(T_warmup, 200, scheduler)
    print("    -> Compilation terminée.\n")
    
    for N in sizes:
        print(f"\n[Test N={N:,}] Génération de la Série Temporelle ...")
        T = generate_mp_data(N, M)
        
        # --- STUMPY (CPU) ---
        t0_cpu = time.perf_counter()
        # stumpy.stump is heavily multi-threaded with numba
        stumpy.stump(T, m=M)
        t1_cpu = time.perf_counter()
        time_cpu = t1_cpu - t0_cpu
        cpu_times.append(time_cpu)
        
        # --- GPUsync (GPU) ---
        t0_gpu = time.perf_counter()
        await stomp_matrix_profile(T, M, scheduler)
        t1_gpu = time.perf_counter()
        time_gpu = t1_gpu - t0_gpu
        gpu_times.append(time_gpu)
        
        print(f"-> CPU: {time_cpu:.3f}s | GPU: {time_gpu:.3f}s | Speedup: {time_cpu/time_gpu:.2f}x")
        
    print("\n=== BILAN DES PERFORMANCES ===")
    for i, N in enumerate(sizes):
        print(f"N={N:<8} : Speedup = {cpu_times[i]/gpu_times[i]:.2f}x")
        
    # --- GRAPHIQUE ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(sizes, cpu_times, marker='o', color='#e74c3c', linewidth=2, label='Stumpy (CPU Multi-thread)')
    plt.plot(sizes, gpu_times, marker='s', color='#2ecc71', linewidth=2, label='GPUsync (GPU)')
    
    plt.fill_between(sizes, cpu_times, gpu_times, color='#e74c3c', alpha=0.1)
    
    plt.title('Performance Matrix Profile STOMP : Scalabilité CPU vs GPU (M=200)', fontsize=14)
    plt.xlabel('Taille du dataset (N)', fontsize=12)
    plt.ylabel('Temps d\'exécution (secondes)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Annoter les speedups
    for i, N in enumerate(sizes):
        speedup = cpu_times[i] / gpu_times[i]
        plt.annotate(f"{speedup:.1f}x", 
                     (sizes[i], gpu_times[i]), 
                     textcoords="offset points", 
                     xytext=(0, 15), 
                     ha='center', 
                     color='green',
                     fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stomp_scaling_benchmark.png')
    plt.savefig(plot_path)
    print(f"\nGraphique détaillé sauvegardé sous : {plot_path}")

if __name__ == "__main__":
    asyncio.run(run_stomp_benchmark_sizes())
