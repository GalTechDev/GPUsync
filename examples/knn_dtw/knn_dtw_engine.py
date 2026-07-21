import time
import asyncio
import numpy as np
import sys
import os

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer

from examples.dtw.dtw_gpu_engine import get_sliding_dtw_kernel




async def knn_dtw_search(query_seq: np.ndarray, time_series: np.ndarray, K: int, scheduler: Scheduler, chunk_size: int = 1_000_000):
    """
    Moteur KNN-DTW : Recherche les K meilleures sous-séquences de `time_series` qui matchent `query_seq`.
    
    - query_seq : 1D array de taille M
    - time_series : 1D array de taille L
    - K : Nombre de voisins à retourner
    - chunk_size : Nombre de fenêtres traitées par le GPU en un batch
    """
    M = len(query_seq)
    L = len(time_series)
    num_windows = L - M + 1
    
    if num_windows <= 0:
        raise ValueError(f"La time_series (L={L}) doit être plus grande que la query (M={M}).")
        
    print(f"[KNN-DTW Engine] Recherche Top-{K} | Query: {M} pts | Base: {L} pts | Total Fenêtres: {num_windows:,}")
    
    gpu_query = await AsyncBuffer.to_device(scheduler, query_seq.astype(np.float32))
    
    top_k_global = [] # Liste de tuples (distance, index)
    
    num_chunks = (num_windows + chunk_size - 1) // chunk_size
    
    # 1. Compilation du Kernel
    kernel = get_sliding_dtw_kernel(M)
    
    total_compute_time = 0.0
    
    # Allocation du buffer de sortie (réutilisé pour chaque chunk)
    gpu_distances = AsyncBuffer.empty_like(scheduler, np.zeros(chunk_size, dtype=np.float32)) 
    
    for i in range(num_chunks):
        offset = i * chunk_size
        current_chunk_windows = min(chunk_size, num_windows - offset)
        
        # Le GPU a besoin de lire jusqu'à offset + current_chunk_windows - 1 + M
        # Donc la taille du chunk T envoyé au GPU est current_chunk_windows + M - 1
        points_to_read = current_chunk_windows + M - 1
        
        # Extraction du chunk (0 copie si memmap, sinon slice rapide)
        T_chunk = time_series[offset : offset + points_to_read].astype(np.float32)
        gpu_T_chunk = await AsyncBuffer.to_device(scheduler, T_chunk)
        
        start_compute = time.perf_counter()
        
        # Lancement du Kernel
        await kernel(
            gpu_query, 
            gpu_T_chunk, 
            gpu_distances, 
            offset, 
            global_size=(current_chunk_windows,),
            scheduler=scheduler
        )
        
        distances_host = await gpu_distances.to_host()
        distances_host = distances_host[:current_chunk_windows]
        
        total_compute_time += time.perf_counter() - start_compute
        
        # Côté CPU : Argpartition en O(N)
        local_K = min(K, current_chunk_windows)
        top_indices_local = np.argpartition(distances_host, local_K - 1)[:local_K]
        
        for local_idx in top_indices_local:
            dist = distances_host[local_idx]
            global_idx = offset + local_idx
            top_k_global.append((dist, global_idx))
            
        # On ne garde que les K meilleurs au fur et à mesure pour éviter que la liste enfle
        top_k_global.sort(key=lambda x: x[0])
        top_k_global = top_k_global[:K]
        
    print(f"[KNN-DTW Engine] Terminé en {total_compute_time:.3f}s (Compute GPU pur)")
    return top_k_global
