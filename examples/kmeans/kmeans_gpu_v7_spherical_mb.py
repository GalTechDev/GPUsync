import time
import asyncio
import numpy as np
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.utils.bg_loop import auto_sync

from kmeans_engine import get_kmeans_spherical_multi_kernel
from kmeans_gpu_v6_spherical import kmeans_spherical_v6



async def process_chunk(kernel_func, x_sub, gpu_X, gpu_C, gpu_sums, gpu_counts, gpu_inertia, N_chunk, chunk_stride, D, K, P, scheduler, pool):
    # Overwrite le buffer VRAM existant avec les nouvelles données
    await gpu_X.overwrite(x_sub)
    
    # Coarsening factor
    COARSE_FACTOR = 4
    local_sz_val = 256
    num_threads_needed = math.ceil(N_chunk / COARSE_FACTOR)
    global_aligned = math.ceil(num_threads_needed / local_sz_val) * local_sz_val
    
    await kernel_func(gpu_X, gpu_C, gpu_sums, gpu_counts, gpu_inertia, N_chunk, chunk_stride, D, K, P, global_size=(global_aligned,), local_size=(local_sz_val,), scheduler=scheduler)
    
    # On rend le buffer au pool
    pool.put_nowait(gpu_X)


@auto_sync
async def kmeans_spherical_minibatch_v7(X_data, K: int, initial_C=None, batch_size: int = 10000, 
                                  max_iters: int = 100, tol: float = 1e-4, P: int = 16, backend="opencl",
                                  mutation_interval: int = 10, mutation_rate: float = 0.01):
    """
    Exécute le K-Means en mode MiniBatch Sphérique Asynchrone avec Algorithme Génétique.
    X_data DOIT être pré-normalisé (norme L2 = 1).
    initial_C doit être de forme (P, K, D) et normalisé.
    Met à jour les centroïdes de façon stochastique sur des petits échantillons (batch_size).
    """
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    N, D = X_data.shape
    kernel = get_kmeans_spherical_multi_kernel(P, K, D)
    
    if initial_C is not None:
        C = initial_C.copy()
    else:
        C = np.zeros((P, K, D), dtype=np.float32)
        for p in range(P):
            indices = np.random.choice(N, K, replace=False)
            C[p] = X_data[indices]
            
    # Historique global des "counts" (v_old) pour chaque population et cluster
    global_counts = np.zeros((P, K), dtype=np.float32)
            
    # Pools
    pool = asyncio.Queue()
    dummy_x = np.zeros((batch_size, D), dtype=np.float32)
    for _ in range(2):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_x.flatten()))
        
    gpu_C = AsyncBuffer.empty_like(scheduler, C.flatten())
    gpu_sums = AsyncBuffer.empty_like(scheduler, np.zeros(P * K * D, dtype=np.float32))
    gpu_counts = AsyncBuffer.empty_like(scheduler, np.zeros(P * K, dtype=np.float32))
    gpu_inertia = AsyncBuffer.empty_like(scheduler, np.zeros(P, dtype=np.float32))
    
    await gpu_C.overwrite(C.flatten())
    
    total_time = 0.0
    
    # Pré-allocations pour éviter le Garbage Collector dans la boucle
    zeros_sums = np.zeros(P * K * D, dtype=np.float32)
    zeros_counts = np.zeros(P * K, dtype=np.float32)
    zeros_inertia = np.zeros(P, dtype=np.float32)
    
    # Mélange et préparation (Transposition SoA) de tous les batches à l'avance
    # Dans un cas réel, les données sont généralement préparées par un DataLoader.
    permuted_indices = np.random.permutation(N)
    X_permuted = X_data[permuted_indices]
    
    num_batches = math.ceil(N / batch_size)
    precomputed_batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)
        chunk = X_permuted[start_idx:end_idx]
        
        if chunk.shape[0] < batch_size:
            padded = np.zeros((batch_size, D), dtype=np.float32)
            padded[:chunk.shape[0], :] = chunk
            chunk = padded
            
        precomputed_batches.append(np.ascontiguousarray(chunk.T).ravel())
        
    print(f"[MiniBatch GPUsync] Démarrage - N={N}, batch_size={batch_size}, P={P}")
    
    for it in range(max_iters):
        start_iter = time.perf_counter()
        
        # Récupération immédiate du batch pré-calculé (O(1))
        batch_idx = it % num_batches
        arr = precomputed_batches[batch_idx]
        
        await gpu_sums.overwrite(zeros_sums)
        await gpu_counts.overwrite(zeros_counts)
        await gpu_inertia.overwrite(zeros_inertia)
        
        gpu_X = await pool.get()
        await gpu_X.overwrite(arr)
        
        COARSE_FACTOR = 4
        local_sz_val = 256
        num_threads_needed = math.ceil(batch_size / COARSE_FACTOR)
        global_aligned = math.ceil(num_threads_needed / local_sz_val) * local_sz_val
        
        # Lancement du noyau
        await kernel(gpu_X, gpu_C, gpu_sums, gpu_counts, gpu_inertia, batch_size, batch_size, D, K, P, global_size=(global_aligned,), local_size=(local_sz_val,), scheduler=scheduler)
        
        host_sums = await gpu_sums.to_host()
        host_counts = await gpu_counts.to_host()
        host_inertia = await gpu_inertia.to_host()
        pool.put_nowait(gpu_X)
        
        host_sums = host_sums.reshape((P, K, D))
        host_counts = host_counts.reshape((P, K))
        
        # Mise à jour Stochastique MiniBatch (Vectorisée avec Numpy)
        v_new = global_counts + host_counts
        
        mask = v_new > 0
        learning_rate = np.zeros_like(v_new, dtype=np.float32)
        learning_rate[mask] = 1.0 / v_new[mask]
        
        C[mask] = C[mask] + learning_rate[mask, None] * (host_sums[mask] - host_counts[mask, None] * C[mask])
        
        # Re-normalisation sphérique
        norms = np.linalg.norm(C, axis=2, keepdims=True)
        C = C / np.where(norms > 1e-12, norms, 1.0)
        
        global_counts = v_new
        
        await gpu_C.overwrite(C.flatten())
        
        best_pop = np.argmin(host_inertia)
        best_inertia = host_inertia[best_pop]
        
        iter_time = time.perf_counter() - start_iter
        total_time += iter_time
        
        # --- LOGIQUE GÉNÉTIQUE (CULLING, MUTATION, PSO & CHASSE AUX MORTS) ---
        if mutation_interval > 0 and it > 0 and it % mutation_interval == 0:
            sorted_indices = np.argsort(host_inertia)
            half = P // 2
            best_half = sorted_indices[:half]
            worst_half = sorted_indices[half:]
            
            # 1. Recuit Simulé sur la mutation (Decaying Mutation)
            current_rate = max(0.001, mutation_rate * (1.0 - (it / max_iters)))
            std_C = np.std(C) * current_rate
            
            # 2. Global Best Attraction (PSO)
            global_best_C = C[best_half[0]].copy()
            
            for i in range(half):
                good_p = best_half[i]
                bad_p = worst_half[i]
                
                # Clonage et Mutation (sur la sphère)
                C[bad_p] = C[good_p].copy()
                C[bad_p] += np.random.normal(0, std_C, C[bad_p].shape).astype(np.float32)
                # Re-normalisation post-mutation
                norms = np.linalg.norm(C[bad_p], axis=1, keepdims=True)
                C[bad_p] = C[bad_p] / np.where(norms > 1e-12, norms, 1.0)
                
                # 3. Clones Souples (Hyperparam Evolution)
                global_counts[bad_p] = global_counts[good_p].copy() / 2.0
                
            # PSO: On attire toutes les populations (sauf la meilleure) vers le Global Best
            for p in range(1, P):
                p_idx = sorted_indices[p]
                C[p_idx] = 0.9 * C[p_idx] + 0.1 * global_best_C
                # Re-normalisation post-PSO
                norms = np.linalg.norm(C[p_idx], axis=1, keepdims=True)
                C[p_idx] = C[p_idx] / np.where(norms > 1e-12, norms, 1.0)
                
            # 3. Anti-dégénérescence (Chasse aux centroïdes morts)
            for p in range(P):
                dead_mask = v_new[p] == 0
                if np.any(dead_mask):
                    num_dead = np.sum(dead_mask)
                    # Les points X_data sont déjà normalisés, on peut les copier directement
                    random_indices = np.random.choice(N, num_dead, replace=False)
                    C[p][dead_mask] = X_data[random_indices]
                    # On réinitialise leur inertie pour qu'ils s'adaptent vite
                    global_counts[p][dead_mask] = 1.0
                
            # Mise à jour du GPU avec la nouvelle génération
            await gpu_C.overwrite(C.flatten())
            
    print(f"[Genetic MiniBatch] Temps total de calcul : {total_time:.2f}s")
    
    # Évaluation finale de l'inertie sur TOUT le dataset (pour comparaison loyale)
    # On fait une passe du vrai K-Means pour obtenir l'inertie finale
    best_C_exact, inertias_exact = await kmeans_spherical_v6(X_data, K, initial_C=C, max_iters=1, P=P, backend=backend)
    
    return best_C_exact, inertias_exact, total_time
