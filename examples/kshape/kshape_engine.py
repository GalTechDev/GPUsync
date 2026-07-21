import time
import asyncio
import numpy as np
import scipy.linalg
import sys
import os
import math

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.compiler import gpusync_kernel

class constant_list(list): pass

def unrolled_range(*args):
    return range(*args)

def get_kshape_kernel(M, K):
    @gpusync_kernel(constants={"M_CONST": M, "K_CONST": K})
    def sbd_assignment_kernel(ctx: int, X: list, C: constant_list, labels: list, shifts: list):
        # Index de départ de la série X pour ce thread
        start_idx: int = ctx * M_CONST
        
        # Charger la série X dans les registres privés
        x_priv = [0.0] * M_CONST
        for m in unrolled_range(M_CONST):
            x_priv[m] = X[start_idx + m]
            
        best_k = 0
        best_shift = 0
        max_ncc = -9999999.0
        
        # Pour chaque centroid
        for k in range(K_CONST):
            # Charger le centroid dans les registres privés depuis la mémoire constante
            c_start: int = k * M_CONST
            c_priv = [0.0] * M_CONST
            for m in unrolled_range(M_CONST):
                c_priv[m] = C[c_start + m]
                
            # Calcul de la Cross-Correlation pour tous les shifts possibles
            # shift s de -(M-1) à +(M-1)
            # s > 0 : x est en avance sur c (ou c en retard)
            for s in range(-(M_CONST - 1), M_CONST):
                cc = 0.0
                
                # Zone de recouvrement
                m_start = 0
                if s < 0:
                    m_start = -s
                
                m_end = M_CONST
                if s > 0:
                    m_end = M_CONST - s
                    
                # On utilise une boucle classique car les bornes dépendent de s
                # Unrolling n'est pas possible si m_start et m_end sont variables.
                for m in range(m_start, m_end):
                    cc += x_priv[m] * c_priv[m + s]
                    
                # Puisque les séries sont Z-normalisées, ||x|| = ||c|| = sqrt(M)
                # Le dénominateur de NCC est M
                ncc = cc / M_CONST
                
                if ncc > max_ncc:
                    max_ncc = ncc
                    best_k = k
                    best_shift = s
                    
        labels[ctx] = best_k
        shifts[ctx] = best_shift
        
    return sbd_assignment_kernel


def extract_shape(X_cluster, shift_cluster, M):
    """
    K-Shape "Shape Extraction" (Centroid update).
    1. Décale toutes les séries du cluster selon `shift_cluster`.
    2. Construit la matrice S = sum(x_shifted * x_shifted^T).
    3. Calcule le premier vecteur propre.
    """
    if len(X_cluster) == 0:
        return np.random.randn(M)
        
    S = np.zeros((M, M), dtype=np.float64)
    
    for i in range(len(X_cluster)):
        x = X_cluster[i]
        s = shift_cluster[i]
        
        # Décaler la série. 
        # Si s > 0, on pad avec des 0.
        x_shifted = np.zeros(M, dtype=np.float64)
        m_start = max(0, s)
        m_end = min(M, M + s)
        x_start = max(0, -s)
        x_end = min(M, M - s)
        
        x_shifted[m_start:m_end] = x[x_start:x_end]
        
        # Additionner l'outer product à S
        # S += x_shifted @ x_shifted.T
        S += np.outer(x_shifted, x_shifted)
        
    # Extraire le vecteur propre principal
    # On utilise eigh car S est symétrique
    # np.linalg.eigh renvoie les valeurs propres triées, le dernier est le plus grand
    eigenvalues, eigenvectors = scipy.linalg.eigh(S)
    
    # Le vecteur propre correspondant à la plus grande valeur propre
    centroid = eigenvectors[:, -1]
    
    # K-shape stipule que la distance à l'origine n'importe pas,
    # mais le signe peut être inversé par la décomposition propre.
    # On choisit le signe qui maximise la somme des corrélations (ou a une somme positive)
    if np.sum(centroid) < 0:
        centroid = -centroid
        
    # Z-Normalisation du centroïde (très important dans K-Shape)
    centroid = (centroid - np.mean(centroid)) / (np.std(centroid) + 1e-8)
    return centroid.astype(np.float32)


async def kshape_gpu_fit(X: np.ndarray, K: int, scheduler: Scheduler, max_iter=20):
    """
    Exécute K-Shape Clustering sur le GPU/CPU.
    X: shape (N, M), Z-normalisé.
    """
    N, M = X.shape
    
    print(f"[K-Shape] Initialisation des centroïdes (K={K}, M={M})...")
    # Choix aléatoire de K séries parmi X comme centroïdes initiaux
    indices = np.random.choice(N, K, replace=False)
    centroids = X[indices].copy()
    
    X_gpu = await AsyncBuffer.to_device(scheduler, X.flatten())
    labels_gpu = AsyncBuffer.empty_like(scheduler, np.zeros(N, dtype=np.float32))
    shifts_gpu = AsyncBuffer.empty_like(scheduler, np.zeros(N, dtype=np.float32))
    
    kernel = get_kshape_kernel(M, K)
    
    best_labels = None
    
    for it in range(max_iter):
        t0 = time.perf_counter()
        
        # 1. Phase d'Assignation (GPU)
        C_gpu = await AsyncBuffer.to_device(scheduler, centroids.flatten())
        
        await kernel(
            X_gpu,
            C_gpu,
            labels_gpu,
            shifts_gpu,
            global_size=(N,),
            scheduler=scheduler
        )
        
        labels = await labels_gpu.to_host()
        shifts = await shifts_gpu.to_host()
        labels = labels.astype(int)
        shifts = shifts.astype(int)
        
        t1 = time.perf_counter()
        
        # Vérifier la convergence (labels inchangés)
        if best_labels is not None and np.array_equal(labels, best_labels):
            print(f"    -> Convergence atteinte à l'itération {it+1}")
            break
            
        best_labels = labels.copy()
        
        # 2. Phase de Mise à Jour (CPU)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = (labels == k)
            if not np.any(mask):
                # Cluster vide : on réinitialise avec une série aléatoire
                new_centroids[k] = X[np.random.choice(N)]
                continue
                
            X_k = X[mask]
            shifts_k = shifts[mask]
            
            # Extraction du centroïde K-Shape
            new_centroids[k] = extract_shape(X_k, shifts_k, M)
            
        centroids = new_centroids
        t2 = time.perf_counter()
        
        print(f"[Iter {it+1}/{max_iter}] GPU SBD: {t1-t0:.3f}s | CPU Update: {t2-t1:.3f}s")
        
    return centroids, best_labels, shifts

