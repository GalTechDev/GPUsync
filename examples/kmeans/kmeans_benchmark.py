import time
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure sklearn is available
try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
except ImportError:
    print("scikit-learn n'est pas installé. Veuillez installer scikit-learn pour exécuter ce benchmark.")
    sys.exit(1)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from kmeans_engine import kmeans_classic, kmeans_mini, kmeans_genetic

async def run_benchmark():
    np.random.seed(1)
    print("=== BENCHMARK K-MEANS : CPU (Scikit-Learn) vs GPU (GPUsync) ===")
    
    # Paramètres de test
    N = 500_000
    K = 16
    D = 16
    P = 16 # Nombre de populations / d'initialisations
    
    print(f"\n[1] Génération du Dataset (N={N:,}, D={D}, K={K}, P={P}) ...")
    X = np.random.randn(N, D).astype(np.float32)
    
    print("\n--- SKLEARN K-MEANS (CPU) ---")
    print("Scikit-Learn exécute 'n_init' initialisations pour trouver la meilleure inertie.")
    t0_sk = time.perf_counter()
    kmeans = KMeans(n_clusters=K, n_init=P, max_iter=20, tol=1e-4, algorithm="lloyd")
    kmeans.fit(X)
    t1_sk = time.perf_counter()
    time_sk = t1_sk - t0_sk
    print(f"Scikit-Learn KMeans ({P} inits) terminé en {time_sk:.3f}s")
    print(f"Inertie Sklearn : {kmeans.inertia_:.2f}")

    print("\n--- SKLEARN MINIBATCH (CPU) ---")
    t0_mb = time.perf_counter()
    mb_kmeans = MiniBatchKMeans(n_clusters=K, n_init=P, max_iter=20, batch_size=10000, tol=1e-4)
    mb_kmeans.fit(X)
    t1_mb = time.perf_counter()
    time_mb = t1_mb - t0_mb
    inertia_mb = mb_kmeans.inertia_
    print(f"Scikit-Learn MiniBatch ({P} inits) terminé en {time_mb:.3f}s")
    print(f"Inertie MiniBatch : {inertia_mb:.2f}")

    print("\n--- GPUSYNC K-MEANS V3 (GPU) ---")
    print("Exécution simultanée de 16 populations en 1 seule passe mémoire Out-Of-Core.")
    t0 = time.time()
    best_C_v3, inertias_v3 = await kmeans_classic(X, K=K, max_iters=20, P=P)
    t_v3 = time.time() - t0
    best_inertia_v3 = np.min(inertias_v3)
    print(f"GPUsync K-Means V3 terminé en {t_v3:.3f}s")
    
    print("\n--- GPUSYNC K-MEANS V5 (GPU - GENETIC LLOYD) ---")
    print("Même exécution que V3 mais avec l'hybridation génétique (La Totale).")
    t0 = time.time()
    best_C_gen, inertias_gen = await kmeans_genetic(X, K=K, max_iters=20, P=P, mutation_interval=3)
    t_v5 = time.time() - t0
    best_inertia_v5 = np.min(inertias_gen)
    print(f"GPUsync K-Means V5 terminé en {t_v5:.3f}s")
    print(f"Inertie GPUsync V5 : {best_inertia_v5:.2f}")

    print("\n--- GPUSYNC MINIBATCH (V4 - GENETIC) ---")
    print("Exécution stochastique par lots de 10 000 points avec sélection naturelle.")
    t0_gpu_mb = time.perf_counter()
    best_C_mb, inertias_mb, time_mb = await kmeans_mini(X, K=K, batch_size=10000, max_iters=100, P=P, mutation_interval=10, mutation_rate=0.01)
    t1_gpu_mb = time.perf_counter()
    time_gpu_mb = t1_gpu_mb - t0_gpu_mb
    t_v4 = time_gpu_mb
    best_inertia_v4 = np.min(inertias_mb)
    print(f"GPUsync MiniBatch V4 terminé en {time_gpu_mb:.3f}s (Temps boucle GPU: {t_v4:.3f}s)")
    print(f"Inertie GPUsync MiniBatch V4 : {best_inertia_v4:.2f}")
    
    print("\n=== BILAN DES PERFORMANCES ===")
    
    labels = ['Sklearn\nLloyd', 'GPUsync\nV3 Lloyd', 'GPUsync\nV5 Genetic', 'Sklearn\nMiniBatch', 'GPUsync\nV4 Genetic MB']
    times = [time_sk, t_v3, t_v5, time_mb, t_v4]
    inertias_list = [kmeans.inertia_, best_inertia_v3, best_inertia_v5, inertia_mb, best_inertia_v4]
    
    colors_time = ['#ff9999', '#ffcc99', '#99ccff', '#cc99ff', '#ffccff']
    colors_prec = ['#66b3ff', '#ffb3e6', '#99ff99', '#e6b3ff', '#ffff99']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- PLOT 1 : TEMPS D'EXECUTION (Plus bas = Mieux) ---
    axes[0].bar(labels, times, color=colors_time, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Temps d\'exécution (secondes)', fontsize=12)
    axes[0].set_title(f'Vitesse d\'exécution (Plus bas = Mieux)', fontsize=14, pad=15)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(times):
        axes[0].text(i, v + (max(times)*0.02), f"{v:.2f}s", ha='center', fontweight='bold', fontsize=11)
        
    # --- PLOT 2 : INERTIE FINALE (Plus bas = Mieux) ---
    # On soustrait un offset pour mieux voir la différence car les valeurs sont très grandes (6.33M)
    min_inertia = min(inertias_list)
    offset = min_inertia * 0.999
    
    diff_inertias = [val - offset for val in inertias_list]
    
    axes[1].bar(labels, diff_inertias, color=colors_prec, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel(f'Inertie finale (- {offset:.0f})', fontsize=12)
    axes[1].set_title(f'Précision Mathématique (Inertie : Plus bas = Mieux)', fontsize=14, pad=15)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(inertias_list):
        if v == min_inertia:
            axes[1].text(i, diff_inertias[i] + (max(diff_inertias)*0.02), f"★ {v:.0f}", ha='center', fontweight='bold', color='green', fontsize=11)
        else:
            axes[1].text(i, diff_inertias[i] + (max(diff_inertias)*0.02), f"{v:.0f}", ha='center', fontweight='bold', fontsize=11)
            
    fig.suptitle(f'Benchmark K-Means (N={N:,} points | K={K} | D={D} | {P} initialisations)', fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), 'kmeans_benchmark.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\nGraphique détaillé sauvegardé sous : {out_path}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
