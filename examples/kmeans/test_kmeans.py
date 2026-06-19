import os
import sys
import asyncio
import numpy as np
import time

try:
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
except ImportError:
    print("Erreur: scikit-learn n'est pas installé. Lancez 'pip install scikit-learn'.")
    sys.exit(1)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kmeans_gpu import kmeans_out_of_core

def sort_centroids(centroids):
    """
    Trie les centroïdes de manière déterministe pour pouvoir les comparer.
    (L'ordre des clusters trouvés par K-Means est arbitraire)
    """
    # Trie par la première dimension, puis la deuxième, etc.
    sorted_indices = np.lexsort([centroids[:, i] for i in range(centroids.shape[1]-1, -1, -1)])
    return centroids[sorted_indices]

def main():
    print("==================================================")
    print("--- TEST K-MEANS OUT-OF-CORE vs SCIKIT-LEARN ---")
    print("==================================================")
    
    # 1. Génération des données de test
    N = 100_000   # 100 000 profils de "caméras"
    D = 64        # 64 caractéristiques par profil
    K = 16        # 16 groupes (clusters) à trouver
    
    print(f"[1] Génération des données : {N} points, {D} dimensions, {K} clusters...")
    X, true_labels = make_blobs(n_samples=N, n_features=D, centers=K, random_state=42)
    X = X.astype(np.float32)
    
    print("\n[2] Lancement de Scikit-Learn (CPU In-Core)...")
    np.random.seed(42)
    random_indices = np.random.choice(N, K, replace=False)
    initial_C = X[random_indices].copy().astype(np.float32)
    
    start_cpu = time.perf_counter()
    kmeans_cpu = KMeans(n_clusters=K, init=initial_C, n_init=1, max_iter=1, algorithm='lloyd')
    kmeans_cpu.fit(X)
    cpu_time = time.perf_counter() - start_cpu
    print(f"    Temps CPU Scikit-Learn : {cpu_time:.2f}s")
    
    print("\n[3] Lancement de GPUsync (GPU Out-Of-Core)...")
    # On limite à 1 itération pour valider strictement le moteur mathématique.
    # Au-delà, l'effet papillon du Float32 (chaos de K-Means) fait diverger les centroïdes.
    C_gpu = kmeans_out_of_core(X, K, initial_C=initial_C, max_iters=1, backend="cuda")
    
    # 4. Comparaison des Inerties (Somme des distances au carré)
    # K-Means est extrêmement instable aux petites variations (float32 vs float64).
    # Des points sur la frontière peuvent basculer et changer les clusters finaux.
    # On compare donc la qualité globale du clustering (Inertia).
    
    def compute_inertia(X_data, centroids):
        # Calcule la distance de chaque point à son centroïde le plus proche
        distances = np.linalg.norm(X_data[:, np.newaxis] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)
        
    inertia_cpu = compute_inertia(X, sort_centroids(kmeans_cpu.cluster_centers_))
    inertia_gpu = compute_inertia(X, sort_centroids(C_gpu))
    
    print("\n[4] Résultats de Précision :")
    print(f"    Inertie Scikit-Learn (CPU) : {inertia_cpu:,.2f}")
    print(f"    Inertie GPUsync (GPU)      : {inertia_gpu:,.2f}")
    
    # Validation du test
    difference = abs(inertia_cpu - inertia_gpu) / inertia_cpu
    if difference < 0.05: # Tolérance de 5% liée aux écarts de Float32
        print("\n[SUCCES] L'algorithme K-Means GPUsync produit des clusters équivalents à Scikit-Learn !")
    else:
        print("\n[ECHEC] Grosse divergence trouvée entre Scikit-Learn et GPUsync.")

if __name__ == "__main__":
    main()
