import time
import asyncio
import numpy as np
import sys
import os

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer

from examples.algorithms.dtw.dtw_gpu_engine import get_sliding_dtw_multi_kernel


async def knn_dtw_multi_search(queries_matrix: np.ndarray, time_series: np.ndarray, K: int, scheduler: Scheduler, chunk_size: int = 1_000_000):
    """
    Moteur KNN-DTW Multi-Query : Recherche les K meilleures sous-séquences pour *chaque* requête.
    
    - queries_matrix : 2D array de forme (Q, M)
    - time_series : 1D array de taille N
    - K : Nombre de correspondances à retourner par requête
    """
    Q, M = queries_matrix.shape
    N = len(time_series)
    num_windows = N - M + 1
    
    # 1. Compilation du Kernel
    kernel = get_sliding_dtw_multi_kernel(M, Q)
    
    total_compute_time = 0.0
    
    # Stockage de toutes les distances pour toutes les requêtes : (Q, num_windows)
    global_distances = np.zeros((Q, num_windows), dtype=np.float32)
    
    # 2. Upload des Queries en Constant Cache
    queries_flat = queries_matrix.flatten().astype(np.float32)
    gpu_queries = await AsyncBuffer.to_device(scheduler, queries_flat)
    
    num_chunks = (num_windows + chunk_size - 1) // chunk_size
    
    # Allocation du buffer de sortie (réutilisé pour chaque chunk)
    gpu_distances = AsyncBuffer.empty_like(scheduler, np.zeros(Q * chunk_size, dtype=np.float32)) 
    
    for i in range(num_chunks):
        offset = i * chunk_size
        current_chunk_windows = min(chunk_size, num_windows - offset)
        
        points_to_read = current_chunk_windows + M - 1
        T_chunk = time_series[offset : offset + points_to_read].astype(np.float32)
        gpu_T_chunk = await AsyncBuffer.to_device(scheduler, T_chunk)
        
        start_compute = time.perf_counter()
        
        # Lancement du Compute GPU
        await kernel(
            gpu_queries, 
            gpu_T_chunk, 
            gpu_distances, 
            chunk_size,  # chunk_offset = chunk_size (statique)
            global_size=(current_chunk_windows,), 
            scheduler=scheduler
        )
        
        # Rapatriement
        res_flat = await gpu_distances.to_host()
        
        total_compute_time += time.perf_counter() - start_compute
        
        # res_flat contient Q blocs successifs de taille chunk_size
        res_matrix = res_flat.reshape(Q, chunk_size)
        
        # On ne copie que les current_chunk_windows résultats
        global_distances[:, offset : offset + current_chunk_windows] = res_matrix[:, :current_chunk_windows]
    
    print(f"[KNN-DTW Multi Engine] Recherche Top-{K} | Queries: {Q}x{M} pts | Base: {N} pts")
    print(f"[KNN-DTW Multi Engine] Terminé en {total_compute_time:.3f}s (Compute GPU pur)")
    
    # 4. Filtrage NMS (Non-Maximum Suppression) indépendant pour chaque requête
    results_per_query = []
    
    for q in range(Q):
        distances_q = global_distances[q]
        top_k_indices = []
        
        # Tri des indices par distance croissante
        sorted_indices = np.argsort(distances_q)
        
        for idx in sorted_indices:
            if len(top_k_indices) >= K:
                break
                
            # Vérifier que ce match ne chevauche pas un match déjà sélectionné
            overlap = False
            for selected_idx in top_k_indices:
                if abs(idx - selected_idx) < M:
                    overlap = True
                    break
            
            if not overlap:
                top_k_indices.append(idx)
                
        results_per_query.append((top_k_indices, distances_q[top_k_indices]))
        
    return results_per_query
