import os
import sys
import time
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_row, ind_col = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kshape_engine import kshape_gpu_fit
from demo_kshape import generate_kshape_dataset
from gpusync.scheduler import Scheduler

try:
    from tslearn.clustering import KShape
except ImportError:
    print("Veuillez installer tslearn pour ce benchmark (pip install tslearn)")
    sys.exit(1)

async def run_kshape_benchmark():
    np.random.seed(42)
    print("=== BENCHMARK K-SHAPE : CPU (Tslearn) vs GPU (GPUsync) ===")
    
    # Paramètres massifs
    N = 3000
    M = 128
    K = 3
    
    print(f"\n[1] Génération du Dataset (N={N:,}, M={M}, K={K}) ...")
    X, true_labels = generate_kshape_dataset(N, M)
    
    # --- TSLEARN K-SHAPE (CPU) ---
    print("\n--- TSLEARN K-SHAPE (CPU) ---")
    print("Tslearn exécute KShape classique (basé sur numba/cython).")
    
    t0_cpu = time.perf_counter()
    ts_ks = KShape(n_clusters=K, max_iter=20, n_init=1, random_state=42)
    labels_cpu = ts_ks.fit_predict(X)
    t1_cpu = time.perf_counter()
    time_cpu = t1_cpu - t0_cpu
    print(f"Tslearn KShape terminé en {time_cpu:.3f}s")
    
    # --- GPUSYNC K-SHAPE (GPU) ---
    print("\n--- GPUSYNC K-SHAPE HYBRID (GPU + CPU Eigen) ---")
    print("Exécution hybride : Assignation SBD (GPU) + Extraction Eigen (CPU).")
    
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    
    t0_gpu = time.perf_counter()
    centroids_gpu, labels_gpu, shifts_gpu = await kshape_gpu_fit(X.astype(np.float32), K, scheduler, max_iter=20)
    t1_gpu = time.perf_counter()
    time_gpu = t1_gpu - t0_gpu
    print(f"GPUsync KShape terminé en {time_gpu:.3f}s")
    
    # --- BILAN ---
    print("\n=== BILAN DES PERFORMANCES ===")
    speedup = time_cpu / time_gpu
    print(f"Speedup GPUsync vs Tslearn : {speedup:.2f}x")
    
    # --- PRECISION ---
    acc_cpu = clustering_accuracy(true_labels, labels_cpu)
    acc_gpu = clustering_accuracy(true_labels, labels_gpu)
    print(f"Précision Tslearn : {acc_cpu:.2%}")
    print(f"Précision GPUsync : {acc_gpu:.2%}")
    
    # --- GRAPHIQUE ---
    labels_plot = ['Tslearn (CPU)', 'GPUsync (GPU)']
    times = [time_cpu, time_gpu]
    accs = [acc_cpu * 100, acc_gpu * 100]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Temps d'exécution
    bars1 = ax1.bar(labels_plot, times, color=['#e74c3c', '#2ecc71'])
    ax1.set_ylabel('Temps d\'exécution (secondes)')
    ax1.set_title(f'Temps (plus c\'est bas, mieux c\'est)\nSpeedup: {speedup:.1f}x')
    
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + (max(times)*0.01), f'{yval:.3f}s', ha='center', va='bottom', fontweight='bold')
        
    # Subplot 2: Précision
    bars2 = ax2.bar(labels_plot, accs, color=['#3498db', '#9b59b6'])
    ax2.set_ylabel('Précision (%)')
    ax2.set_title('Précision du Clustering (plus c\'est haut, mieux c\'est)')
    ax2.set_ylim(0, 110)
    
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.suptitle(f'K-Shape Benchmark (N={N:,}, M={M}, K={K})', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kshape_benchmark.png')
    plt.savefig(plot_path)
    print(f"\nGraphique détaillé sauvegardé sous : {plot_path}")

if __name__ == "__main__":
    asyncio.run(run_kshape_benchmark())
