import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize

# Configure PyOpenCL to use the first available platform/device automatically if possible
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kmeans_engine import kmeans_spherical_classic, kmeans_spherical_mini

def generate_spherical_data(N, D, K):
    print(f"[1] Génération du Dataset Sphérique (N={N:,}, D={D}, K={K}) ...")
    
    # 1. Générer K centres aléatoires sur la sphère
    true_centers = np.random.randn(K, D)
    true_centers = normalize(true_centers)
    
    # 2. Assigner chaque point à un cluster aléatoire
    labels = np.random.randint(0, K, size=N)
    
    # 3. Ajouter du bruit gaussien (concentration)
    noise_level = 0.2
    X = true_centers[labels] + np.random.randn(N, D) * noise_level
    
    # 4. Normaliser chaque point sur la sphère
    X = normalize(X).astype(np.float32)
    
    return X

def cosine_inertia(X, centroids, labels):
    # L'inertie Cosinus est la somme des (1 - Cosinus(x, C)) pour chaque point x et son centroïde C
    # dist = 1 - dot(x, C)
    # centroids doivent être normalisés
    normalized_centroids = normalize(centroids)
    assigned_centroids = normalized_centroids[labels]
    
    # Produit scalaire point par point
    dot_products = np.sum(X * assigned_centroids, axis=1)
    
    # Somme des distances
    return np.sum(1.0 - dot_products)

async def run_spherical_benchmark():
    np.random.seed(1)
    print("=== BENCHMARK SPHERICAL K-MEANS : CPU (Sklearn) vs GPU (GPUsync V6) ===")
    
    # Paramètres massifs
    N = 500_000
    D = 16
    K = 16
    P = 16
    
    X = generate_spherical_data(N, D, K)
    
    print("\n--- SKLEARN L2-NORMALIZED (CPU) ---")
    print("Sklearn exécute le KMeans euclidien classique sur les données normalisées.")
    t0_sk = time.perf_counter()
    kmeans = KMeans(n_clusters=K, n_init=P, max_iter=20, random_state=42)
    kmeans.fit(X)
    t1_sk = time.perf_counter()
    time_sk = t1_sk - t0_sk
    
    # Évaluation équitable avec la vraie Inertie Cosinus
    sk_inertia = cosine_inertia(X, kmeans.cluster_centers_, kmeans.labels_)
    
    print(f"Scikit-Learn ({P} inits) terminé en {time_sk:.3f}s")
    print(f"Inertie Sklearn (Cosinus) : {sk_inertia:.2f}")

    print("\n--- GPUSYNC SPHERICAL V6 (GPU) ---")
    print("Exécution avec distance Cosinus et Algorithme Génétique.")
    t0_gpu = time.perf_counter()
    
    # Initialisation de C (P, K, D) normalisé
    initial_C = np.random.randn(P, K, D).astype(np.float32)
    for p in range(P):
        initial_C[p] = normalize(initial_C[p])
        
    best_C, inertias = await kmeans_spherical_classic(X, K=K, initial_C=initial_C, max_iters=20, P=P, mutation_interval=3)
    t1_gpu = time.perf_counter()
    time_gpu = t1_gpu - t0_gpu
    
    best_inertia_gpu = np.min(inertias)
    print(f"GPUsync Spherical V6 terminé en {time_gpu:.3f}s")
    print(f"Inertie GPUsync Spherical V6 : {best_inertia_gpu:.2f}")
    
    print("\n--- GPUSYNC SPHERICAL MINIBATCH V7 (GPU) ---")
    print("Exécution avec distance Cosinus et Algorithme Génétique sur des MiniBatchs de 10 000 points.")
    t0_mb = time.perf_counter()
    best_C_mb, inertias_mb, _ = await kmeans_spherical_mini(X, K=K, initial_C=initial_C, batch_size=10000, max_iters=100, P=P, mutation_interval=10)
    t1_mb = time.perf_counter()
    time_mb = t1_mb - t0_mb
    
    best_inertia_mb = np.min(inertias_mb)
    print(f"GPUsync Spherical MiniBatch V7 terminé en {time_mb:.3f}s")
    print(f"Inertie GPUsync Spherical MiniBatch V7 : {best_inertia_mb:.2f}")
    
    print("\n=== BILAN DES PERFORMANCES ===")
    print(f"Speedup V6 vs Sklearn : {time_sk / time_gpu:.2f}x")
    print(f"Speedup V7 vs Sklearn : {time_sk / time_mb:.2f}x")
    
    # --- PLOT ---
    labels = ['Sklearn\n(L2 CPU)', 'GPUsync V6\n(Spherical GPU)', 'GPUsync V7\n(Spherical MB GPU)']
    times = [time_sk, time_gpu, time_mb]
    inertias_list = [sk_inertia, best_inertia_gpu, best_inertia_mb]
    
    colors_time = ['#ff9999', '#99ccff', '#cc99ff']
    colors_prec = ['#66b3ff', '#99ff99', '#e6b3ff']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphe Vitesse
    axes[0].bar(x=np.arange(len(labels)), height=times, color=colors_time, edgecolor='black', zorder=3)
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Temps (secondes)")
    axes[0].set_title(f"Temps de Calcul Sphérique (N={N}, K={K}, D={D}, P={P})")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    for i, v in enumerate(times):
        axes[0].text(i, v + (max(times)*0.02), f"{v:.2f}s", ha='center', fontweight='bold')
        
    # Graphe Précision (Inertie Cosinus)
    # L'inertie Cosinus est très petite comparée à l'Euclidien, donc on affiche un bar plot en centrant sur la variance
    axes[1].bar(x=np.arange(len(labels)), height=inertias_list, color=colors_prec, edgecolor='black', zorder=3)
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Inertie Cosinus (Plus bas = Meilleur)")
    axes[1].set_title("Qualité de Convergence (Distance Angulaire)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Zoom pour mieux voir la différence
    min_inertia = min(inertias_list)
    max_inertia = max(inertias_list)
    margin = (max_inertia - min_inertia) * 0.5
    if margin == 0: margin = max_inertia * 0.05
    axes[1].set_ylim(max(0, min_inertia - margin), max_inertia + margin)
    
    for i, v in enumerate(inertias_list):
        axes[1].text(i, v + margin*0.1, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spherical_benchmark.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nGraphique détaillé sauvegardé sous : {plot_path}")

if __name__ == "__main__":
    asyncio.run(run_spherical_benchmark())
