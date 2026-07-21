import asyncio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kshape_engine import kshape_gpu_fit
from gpusync.scheduler import Scheduler

def generate_kshape_dataset(N=3000, M=128):
    np.random.seed(42)
    X = np.zeros((N, M), dtype=np.float32)
    true_labels = np.zeros(N, dtype=int)
    
    # 3 Motifs de base
    t = np.linspace(0, 1, M)
    
    # Motif 0: Sinusoide avec bosse
    m0 = np.sin(2 * np.pi * t * 2) + np.exp(-((t - 0.5) ** 2) / 0.02)
    
    # Motif 1: Double pic (Gaussian)
    m1 = np.exp(-((t - 0.3) ** 2) / 0.01) + 0.8 * np.exp(-((t - 0.7) ** 2) / 0.01)
    
    # Motif 2: Plateau (Square-like)
    m2 = np.where((t > 0.2) & (t < 0.6), 1.0, 0.0) + np.sin(10 * np.pi * t) * 0.1
    
    motifs = [m0, m1, m2]
    
    for i in range(N):
        label = np.random.randint(0, 3)
        true_labels[i] = label
        base = motifs[label].copy()
        
        # Ajout de bruit
        base += np.random.randn(M) * 0.2
        
        # Décalage aléatoire (Time Shift Invariance)
        # On décale la série vers la gauche ou la droite
        shift = np.random.randint(-M // 4, M // 4)
        shifted = np.zeros(M)
        if shift > 0:
            shifted[shift:] = base[:-shift]
        elif shift < 0:
            shifted[:shift] = base[-shift:]
        else:
            shifted = base
            
        # Z-Normalisation stricte requise par K-Shape
        shifted = (shifted - np.mean(shifted)) / (np.std(shifted) + 1e-8)
        X[i] = shifted
        
    return X, true_labels

async def main():
    print("=== Démonstration de K-Shape GPU (Time-Series Clustering) ===")
    N = 3000
    M = 128
    K = 3
    
    print(f"[1] Génération du Dataset Synthétique (N={N}, M={M}, K={K})...")
    X, true_labels = generate_kshape_dataset(N, M)
    print("    -> 3 motifs temporels différents (Sinus, Double Pic, Plateau) décalés aléatoirement dans le temps.")
    
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()
    
    try:
        print("\n[2] Lancement de K-Shape Hybride (SBD sur GPU, Eigen-Extraction sur CPU)...")
        # On force la conversion en float32 pour l'OpenCL
        centroids, labels, shifts = await kshape_gpu_fit(X.astype(np.float32), K, scheduler, max_iter=20)
        
        print("\n[3] Génération du graphique visuel...")
        fig, axes = plt.subplots(K, 1, figsize=(10, 8), sharex=True)
        
        for k in range(K):
            ax = axes[k]
            mask = (labels == k)
            count = np.sum(mask)
            
            # Afficher 50 séries aléatoires du cluster
            indices = np.where(mask)[0]
            if len(indices) > 50:
                indices = np.random.choice(indices, 50, replace=False)
                
            for idx in indices:
                # Afficher la série NON-DECALEE (telle qu'elle est dans le dataset)
                # Mais pour que ça soit joli, on pourrait afficher la série décalée.
                # Affichons d'abord la série originale en gris très clair
                ax.plot(X[idx], color='gray', alpha=0.1)
                
            # Afficher le Centroïde K-Shape
            ax.plot(centroids[k], color='red', linewidth=3, label='K-Shape Centroid')
            ax.set_title(f"Cluster {k} ({count} séries)")
            ax.legend()
            
        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kshape_clusters.png')
        plt.savefig(save_path)
        print(f"Graphique sauvegardé sous : {save_path}")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())
