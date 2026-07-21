import os
import sys
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs

# Ajouter le chemin racine au path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from examples.kmeans.kmeans_gpu import kmeans_out_of_core

async def main():
    print("Génération du dataset 2D...")
    N = 100_000
    D = 2
    K = 5
    
    X, true_labels = make_blobs(n_samples=N, n_features=D, centers=K, random_state=42, cluster_std=1.5)
    X = X.astype(np.float32)
    
    np.random.seed(42)
    random_indices = np.random.choice(N, K, replace=False)
    initial_C = X[random_indices].copy().astype(np.float32)
    
    print("Lancement K-Means GPUsync...")
    C_gpu = await kmeans_out_of_core(X, K, initial_C=initial_C, max_iters=20, backend="cuda")
    
    print("Assignation finale pour la couleur des points...")
    # Calcul des distances (en CPU vu que c'est juste 100k points en 2D pour le plot)
    dists = np.sum((X[:, np.newaxis, :] - C_gpu[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    
    print("Génération du graphique...")
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8), facecolor='#121212')
    ax = plt.gca()
    ax.set_facecolor('#1e1e1e')
    
    # Points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=1, alpha=0.5)
    
    # Centroïdes
    plt.scatter(C_gpu[:, 0], C_gpu[:, 1], c='red', marker='X', s=200, edgecolors='white', linewidths=2, label='Centroïdes trouvés')
    
    plt.title('Résultat du Clustering K-Means 2D (GPUsync Out-Of-Core)', color='white', fontsize=14)
    plt.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
    
    output_path = os.path.join(os.path.dirname(__file__), 'cluster_plot_2d.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#121212')
    print(f"Graphique sauvegardé sous : {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
