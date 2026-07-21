import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gpusync.compiler import gpusync_kernel
from gpusync.api.opencl import get_local_id, local_memory, barrier, CLK_LOCAL_MEM_FENCE
import math
import time
import asyncio
import numpy as np
from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.utils.bg_loop import auto_sync
import math

# Dummy functions for AST parsing
def get_global_id(dim): return 0
def get_local_id(dim): return 0
def atomic_add(target, val): pass
def atomic_add_float(target, val): pass
def atomic_add_float_global(target, val): pass
def local_memory(shape, dtype): return []
def barrier(flags): pass
CLK_LOCAL_MEM_FENCE = 0
constant_list = list

def get_kmeans_multi_kernel(P: int, K: int, D: int):
    """
    Génère le kernel K-Means Multi-Population.
    P populations sont évaluées simultanément par thread depuis la mémoire constante (cache).
    """
    PK = P * K
    PKD = P * K * D
    LOCAL_SZ = 256
    COARSE_FACTOR = 4
    ITERS_PKD = math.ceil(PKD / LOCAL_SZ)
    ITERS_PK = math.ceil(PK / LOCAL_SZ)
    
    @gpusync_kernel(constants={"PKD_CONST": PKD, "PK_CONST": PK, "K_CONST": K, "D_CONST": D, "P_CONST": P, "LOCAL_SZ": LOCAL_SZ, "COARSE_FACTOR": COARSE_FACTOR, "ITERS_PKD": ITERS_PKD, "ITERS_PK": ITERS_PK})
    def kmeans_multi_kernel(ctx: int, X: list, C: constant_list, sums: list, counts: list, inertia: list, N_arg: int, stride: int, D_arg: int, K_arg: int, P_arg: int):
        local_idx: int = get_local_id(0)
        global_idx: int = get_global_id(0)
        
        # Shared memory pour accumuler les résultats
        local_sums = local_memory((PKD_CONST,), "float")
        local_counts = local_memory((PK_CONST,), "float")
        local_inertia = local_memory((P_CONST,), "float")
        
        # Initialisation de la mémoire partagée
        for i in range(0, ITERS_PKD):
            offset: int = i * LOCAL_SZ
            if local_idx + offset < PKD_CONST:
                local_sums[local_idx + offset] = 0.0
                
        for i_k in range(0, ITERS_PK):
            offset_k: int = i_k * LOCAL_SZ
            if local_idx + offset_k < PK_CONST:
                local_counts[local_idx + offset_k] = 0.0
                
        if local_idx < P_CONST:
            local_inertia[local_idx] = 0.0
                
        barrier(CLK_LOCAL_MEM_FENCE)
        
        ctx_base: int = global_idx * COARSE_FACTOR
        
        if ctx_base < N_arg:
            for c in range(0, COARSE_FACTOR):
                ctx_point: int = ctx_base + c
                if ctx_point < N_arg:
                    
                    # Boucle sur les P populations
                    for p in range(0, P_CONST):
                        min_dist: float = 1e30
                        best_k: int = 0
                        
                        # Calcul de distance classique pour la population p
                        for k in range(0, K_CONST):
                            dist: float = 0.0
                            for d in range(0, D_CONST):
                                diff: float = X[d * stride + ctx_point] - C[p * K_CONST * D_CONST + k * D_CONST + d]
                                dist += diff * diff
                                
                            if dist < min_dist:
                                min_dist = dist
                                best_k = k
                                
                        # Accumulation atomique matérielle en Shared Memory
                        atomic_add_float(local_inertia[p], min_dist)
                        atomic_add_float(local_counts[p * K_CONST + best_k], 1.0)
                        for d in range(0, D_CONST):
                            val: float = X[d * stride + ctx_point]
                            atomic_add_float(local_sums[p * K_CONST * D_CONST + best_k * D_CONST + d], val)
                
        barrier(CLK_LOCAL_MEM_FENCE)
        
        # Rapport vers la VRAM Globale
        for j in range(0, ITERS_PKD):
            offset2: int = j * LOCAL_SZ
            if local_idx + offset2 < PKD_CONST:
                atomic_add_float_global(sums[local_idx + offset2], local_sums[local_idx + offset2])
                
        for j_k in range(0, ITERS_PK):
            offset_k2: int = j_k * LOCAL_SZ
            if local_idx + offset_k2 < PK_CONST:
                atomic_add_float_global(counts[local_idx + offset_k2], local_counts[local_idx + offset_k2])
                
        if local_idx < P_CONST:
            atomic_add_float_global(inertia[local_idx], local_inertia[local_idx])

    return kmeans_multi_kernel


def get_kmeans_spherical_multi_kernel(P: int, K: int, D: int):
    """
    Génère le kernel K-Means Spherical Multi-Population.
    Utilise la distance Cosinus: 1.0 - dot(X, C).
    """
    PK = P * K
    PKD = P * K * D
    LOCAL_SZ = 256
    COARSE_FACTOR = 4
    ITERS_PKD = math.ceil(PKD / LOCAL_SZ)
    ITERS_PK = math.ceil(PK / LOCAL_SZ)
    
    @gpusync_kernel(constants={"PKD_CONST": PKD, "PK_CONST": PK, "K_CONST": K, "D_CONST": D, "P_CONST": P, "LOCAL_SZ": LOCAL_SZ, "COARSE_FACTOR": COARSE_FACTOR, "ITERS_PKD": ITERS_PKD, "ITERS_PK": ITERS_PK})
    def kmeans_spherical_multi_kernel(ctx: int, X: list, C: constant_list, sums: list, counts: list, inertia: list, N_arg: int, stride: int, D_arg: int, K_arg: int, P_arg: int):
        local_idx: int = get_local_id(0)
        global_idx: int = get_global_id(0)
        
        # Shared memory pour accumuler les résultats
        local_sums = local_memory((PKD_CONST,), "float")
        local_counts = local_memory((PK_CONST,), "float")
        local_inertia = local_memory((P_CONST,), "float")
        
        # Initialisation de la mémoire partagée
        for i in range(0, ITERS_PKD):
            offset: int = i * LOCAL_SZ
            if local_idx + offset < PKD_CONST:
                local_sums[local_idx + offset] = 0.0
                
        for i_k in range(0, ITERS_PK):
            offset_k: int = i_k * LOCAL_SZ
            if local_idx + offset_k < PK_CONST:
                local_counts[local_idx + offset_k] = 0.0
                
        if local_idx < P_CONST:
            local_inertia[local_idx] = 0.0
                
        barrier(CLK_LOCAL_MEM_FENCE)
        
        ctx_base: int = global_idx * COARSE_FACTOR
        
        if ctx_base < N_arg:
            for c in range(0, COARSE_FACTOR):
                ctx_point: int = ctx_base + c
                if ctx_point < N_arg:
                    
                    # Boucle sur les P populations
                    for p in range(0, P_CONST):
                        min_dist: float = 1e30
                        best_k: int = 0
                        
                        # Calcul de distance COSINUS pour la population p
                        for k in range(0, K_CONST):
                            dot: float = 0.0
                            for d in range(0, D_CONST):
                                val_x: float = X[d * stride + ctx_point]
                                val_c: float = C[p * K_CONST * D_CONST + k * D_CONST + d]
                                dot += val_x * val_c
                                
                            dist: float = 1.0 - dot
                            if dist < min_dist:
                                min_dist = dist
                                best_k = k
                                
                        # Accumulation atomique matérielle en Shared Memory
                        atomic_add_float(local_inertia[p], min_dist)
                        atomic_add_float(local_counts[p * K_CONST + best_k], 1.0)
                        for d in range(0, D_CONST):
                            val_to_add: float = X[d * stride + ctx_point]
                            atomic_add_float(local_sums[p * K_CONST * D_CONST + best_k * D_CONST + d], val_to_add)
                
        barrier(CLK_LOCAL_MEM_FENCE)
        
        # Rapport vers la VRAM Globale
        for j in range(0, ITERS_PKD):
            offset2: int = j * LOCAL_SZ
            if local_idx + offset2 < PKD_CONST:
                atomic_add_float_global(sums[local_idx + offset2], local_sums[local_idx + offset2])
                
        for j_k in range(0, ITERS_PK):
            offset_k2: int = j_k * LOCAL_SZ
            if local_idx + offset_k2 < PK_CONST:
                atomic_add_float_global(counts[local_idx + offset_k2], local_counts[local_idx + offset_k2])
                
        if local_idx < P_CONST:
            atomic_add_float_global(inertia[local_idx], local_inertia[local_idx])

    return kmeans_spherical_multi_kernel




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
async def kmeans_classic(X_data, K: int, initial_C=None, max_iters: int = 10, tol: float = 1e-5, P: int = 8, backend="opencl"):
    """
    Exécute le K-Means en mode Batched Out-Of-Core Asynchrone.
    initial_C doit être de forme (P, K, D).
    """
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    N, D = X_data.shape
    kernel = get_kmeans_multi_kernel(P, K, D)
    
    # 1. Initialisation des centroïdes
    if initial_C is not None:
        C = initial_C.copy().astype(np.float32)
    else:
        np.random.seed(42)
        random_indices = np.random.choice(N, P * K, replace=False)
        C = X_data[random_indices].reshape((P, K, D)).astype(np.float32)
    
    gpu_C = await AsyncBuffer.to_device(scheduler, C.flatten())
    
    # 2. Préparation du Memory Pool pour l'Upload
    MAX_CHUNK = 1_000_000 
    IN_FLIGHT_LIMIT = 4
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    
    chunk_size = min(N, MAX_CHUNK)
    dummy_x = np.zeros((chunk_size, D), dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_x.flatten()))
        
    gpu_sums = AsyncBuffer.empty_like(scheduler, np.zeros((P, K, D), dtype=np.float32).flatten())
    gpu_counts = AsyncBuffer.empty_like(scheduler, np.zeros((P, K), dtype=np.float32).flatten())
    gpu_inertia = AsyncBuffer.empty_like(scheduler, np.zeros(P, dtype=np.float32))
    
    num_chunks = math.ceil(N / MAX_CHUNK)
    
    # 3. Préparation des chunks (Transposition en SoA)
    # Identique à V2...Transposition SoA des Chunks en RAM...")
    prepared_chunks = []
    for i in range(num_chunks):
        start_N = i * MAX_CHUNK
        end_N = min((i + 1) * MAX_CHUNK, N)
        current_N = end_N - start_N
        
        chunk = X_data[start_N:end_N]
        # Transposition mémoire (SoA)
        if chunk.shape[0] == chunk_size:
            arr = np.ascontiguousarray(chunk.T).ravel()
        else:
            padded = np.zeros((chunk_size, D), dtype=np.float32)
            padded[:chunk.shape[0], :] = chunk
            arr = np.ascontiguousarray(padded.T).ravel()
            
        prepared_chunks.append(arr)
        
    total_time = 0.0
    
    # Historique de la meilleure inertie pour le benchmark
    best_inertia_history = []
    
    for it in range(max_iters):
        start_iter = time.perf_counter()
        
        # Remise à zéro des accumulateurs
        await gpu_sums.overwrite(np.zeros((P, K, D), dtype=np.float32).flatten())
        await gpu_counts.overwrite(np.zeros((P, K), dtype=np.float32).flatten())
        await gpu_inertia.overwrite(np.zeros(P, dtype=np.float32))
        
        active_tasks = set()
        
        for i in range(num_chunks):
            start_N = i * MAX_CHUNK
            end_N = min((i + 1) * MAX_CHUNK, N)
            current_N = end_N - start_N
            
            res_extract = prepared_chunks[i]
            gpu_X = await pool.get()
            
            task = asyncio.create_task(process_chunk(
                kernel, res_extract, gpu_X, gpu_C, gpu_sums, gpu_counts, gpu_inertia, current_N, chunk_size, D, K, P, scheduler, pool
            ))
            active_tasks.add(task)
            
        if active_tasks:
            done, _ = await asyncio.wait(active_tasks)
            for coro in done:
                coro.result()
            
        # Téléchargement des résultats de l'itération
        host_sums = await gpu_sums.to_host()
        host_counts = await gpu_counts.to_host()
        host_inertia = await gpu_inertia.to_host()
        
        host_sums = host_sums.reshape((P, K, D))
        host_counts = host_counts.reshape((P, K))
        
        # On trouve la meilleure population (inertie min)
        best_pop = np.argmin(host_inertia)
        best_inertia = host_inertia[best_pop]
        
        # Mise à jour des centroïdes
        C_new = np.zeros_like(C)
        for p in range(P):
            for k in range(K):
                if host_counts[p, k] > 0:
                    C_new[p, k] = host_sums[p, k] / host_counts[p, k]
                else:
                    C_new[p, k] = C[p, k]
                
        # Le déplacement moyen n'a de sens que sur la meilleure population
        shift = np.linalg.norm(C_new[best_pop] - C[best_pop])
        C[:] = C_new[:]
        await gpu_C.overwrite(C.flatten())
        
        iter_time = time.perf_counter() - start_iter
        total_time += iter_time
        
        best_inertia_history.append(best_inertia)
        
        print(f"  Iter {it+1}/{max_iters} | Temps: {iter_time:.2f}s | Inertie Min: {best_inertia:.2e} | Pop Gagnante: {best_pop} | Déplacement: {shift:.4f}")
        
        if tol > 0.0 and shift < tol:
            print(f"[K-Means Batched OOC] Convergence de la meilleure population atteinte en {it+1} itérations !")
            # On prolonge l'historique jusqu'à max_iters pour le plot
            while len(best_inertia_history) < max_iters:
                best_inertia_history.append(best_inertia)
            break
            
    print(f"[K-Means Batched OOC] Temps total de calcul : {total_time:.2f}s")
    # On renvoie la meilleure population et les inerties de toutes les populations
    return C[best_pop], host_inertia


@auto_sync
async def kmeans_mini(X_data, K: int, initial_C=None, max_iters: int = 100, batch_size: int = 10000, P: int = 16, backend="opencl", mutation_interval=10, mutation_rate=0.01):
    """
    Exécute le MiniBatch K-Means en utilisant le noyau GPUsync.
    Met à jour les centroïdes de façon stochastique sur des petits échantillons (batch_size).
    """
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    N_total, D = X_data.shape
    kernel = get_kmeans_multi_kernel(P, K, D)
    
    if initial_C is not None:
        C = initial_C.copy()
    else:
        C = np.zeros((P, K, D), dtype=np.float32)
        for p in range(P):
            indices = np.random.choice(N_total, K, replace=False)
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
    permuted_indices = np.random.permutation(N_total)
    X_permuted = X_data[permuted_indices]
    
    num_batches = math.ceil(N_total / batch_size)
    precomputed_batches = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N_total)
        chunk = X_permuted[start_idx:end_idx]
        
        if chunk.shape[0] < batch_size:
            padded = np.zeros((batch_size, D), dtype=np.float32)
            padded[:chunk.shape[0], :] = chunk
            chunk = padded
            
        precomputed_batches.append(np.ascontiguousarray(chunk.T).ravel())
        
    print(f"[MiniBatch GPUsync] Démarrage - N={N_total}, batch_size={batch_size}, P={P}")
    
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
        mask = host_counts > 0
        
        v_old = global_counts[mask]
        v_new = v_old + host_counts[mask]
        global_counts[mask] = v_new
        
        v_old_expanded = v_old[:, None]
        v_new_expanded = v_new[:, None]
        
        C[mask] = (C[mask] * v_old_expanded + host_sums[mask]) / v_new_expanded
        
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
                
                # Clonage et Mutation
                C[bad_p] = C[good_p].copy()
                C[bad_p] += np.random.normal(0, std_C, C[bad_p].shape).astype(np.float32)
                
                # 3. Clones Souples (Hyperparam Evolution)
                global_counts[bad_p] = global_counts[good_p].copy() / 2.0
                
            # PSO: On attire toutes les populations (sauf la meilleure) vers le Global Best
            for p in range(1, P):
                p_idx = sorted_indices[p]
                C[p_idx] = 0.9 * C[p_idx] + 0.1 * global_best_C
                
            # 4. Anti-dégénérescence (Chasse aux centroïdes morts)
            for p in range(P):
                dead_mask = host_counts[p] == 0
                if np.any(dead_mask):
                    num_dead = np.sum(dead_mask)
                    # On ressuscite les morts en les plaçant sur des points aléatoires du dataset
                    random_indices = np.random.choice(N_total, num_dead, replace=False)
                    C[p][dead_mask] = X_data[random_indices]
                    # On réinitialise leur inertie pour qu'ils s'adaptent vite
                    global_counts[p][dead_mask] = 1.0
                
            # Mise à jour du GPU avec la nouvelle génération
            await gpu_C.overwrite(C.flatten())
            
    print(f"[Genetic MiniBatch] Temps total de calcul : {total_time:.2f}s")
    
    # Évaluation finale de l'inertie sur TOUT le dataset (pour comparaison loyale)
    # On fait une passe du vrai K-Means pour obtenir l'inertie finale
    best_C_exact, inertias_exact = await kmeans_classic(X_data, K, initial_C=C, max_iters=1, P=P, backend=backend)
    
    return best_C_exact, inertias_exact, total_time


@auto_sync
async def kmeans_genetic(X_data, K: int, initial_C=None, max_iters: int = 20, tol: float = 1e-5, P: int = 16, backend="opencl", mutation_interval=3, mutation_rate=0.01):
    """
    Exécute le K-Means en mode Batched Out-Of-Core Asynchrone avec Algorithme Génétique.
    initial_C doit être de forme (P, K, D).
    """
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    N, D = X_data.shape
    kernel = get_kmeans_multi_kernel(P, K, D)
    
    # 1. Initialisation des centroïdes
    if initial_C is not None:
        C = initial_C.copy().astype(np.float32)
    else:
        np.random.seed(42)
        random_indices = np.random.choice(N, P * K, replace=False)
        C = X_data[random_indices].reshape((P, K, D)).astype(np.float32)
    
    gpu_C = await AsyncBuffer.to_device(scheduler, C.flatten())
    
    # 2. Préparation du Memory Pool pour l'Upload
    MAX_CHUNK = 1_000_000 
    IN_FLIGHT_LIMIT = 4
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    
    chunk_size = min(N, MAX_CHUNK)
    dummy_x = np.zeros((chunk_size, D), dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_x.flatten()))
        
    gpu_sums = AsyncBuffer.empty_like(scheduler, np.zeros((P, K, D), dtype=np.float32).flatten())
    gpu_counts = AsyncBuffer.empty_like(scheduler, np.zeros((P, K), dtype=np.float32).flatten())
    gpu_inertia = AsyncBuffer.empty_like(scheduler, np.zeros(P, dtype=np.float32))
    
    num_chunks = math.ceil(N / MAX_CHUNK)
    
    # 3. Préparation des chunks (Transposition en SoA)
    # Identique à V2...Transposition SoA des Chunks en RAM...")
    prepared_chunks = []
    for i in range(num_chunks):
        start_N = i * MAX_CHUNK
        end_N = min((i + 1) * MAX_CHUNK, N)
        current_N = end_N - start_N
        
        chunk = X_data[start_N:end_N]
        # Transposition mémoire (SoA)
        if chunk.shape[0] == chunk_size:
            arr = np.ascontiguousarray(chunk.T).ravel()
        else:
            padded = np.zeros((chunk_size, D), dtype=np.float32)
            padded[:chunk.shape[0], :] = chunk
            arr = np.ascontiguousarray(padded.T).ravel()
            
        prepared_chunks.append(arr)
        
    total_time = 0.0
    
    # Historique de la meilleure inertie pour le benchmark
    best_inertia_history = []
    
    for it in range(max_iters):
        start_iter = time.perf_counter()
        
        # Remise à zéro des accumulateurs
        await gpu_sums.overwrite(np.zeros((P, K, D), dtype=np.float32).flatten())
        await gpu_counts.overwrite(np.zeros((P, K), dtype=np.float32).flatten())
        await gpu_inertia.overwrite(np.zeros(P, dtype=np.float32))
        
        active_tasks = set()
        
        for i in range(num_chunks):
            start_N = i * MAX_CHUNK
            end_N = min((i + 1) * MAX_CHUNK, N)
            current_N = end_N - start_N
            
            res_extract = prepared_chunks[i]
            gpu_X = await pool.get()
            
            task = asyncio.create_task(process_chunk(
                kernel, res_extract, gpu_X, gpu_C, gpu_sums, gpu_counts, gpu_inertia, current_N, chunk_size, D, K, P, scheduler, pool
            ))
            active_tasks.add(task)
            
        if active_tasks:
            done, _ = await asyncio.wait(active_tasks)
            for coro in done:
                coro.result()
            
        # Téléchargement des résultats de l'itération
        host_sums = await gpu_sums.to_host()
        host_counts = await gpu_counts.to_host()
        host_inertia = await gpu_inertia.to_host()
        
        host_sums = host_sums.reshape((P, K, D))
        host_counts = host_counts.reshape((P, K))
        
        # On trouve la meilleure population (inertie min)
        best_pop = np.argmin(host_inertia)
        best_inertia = host_inertia[best_pop]
        
        # Mise à jour des centroïdes
        C_new = np.zeros_like(C)
        for p in range(P):
            for k in range(K):
                if host_counts[p, k] > 0:
                    C_new[p, k] = host_sums[p, k] / host_counts[p, k]
                else:
                    C_new[p, k] = C[p, k]
                
        # Le déplacement moyen n'a de sens que sur la meilleure population
        shift = np.linalg.norm(C_new[best_pop] - C[best_pop])
        C[:] = C_new[:]
        
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
                
                # Clonage et Mutation
                C[bad_p] = C[good_p].copy()
                C[bad_p] += np.random.normal(0, std_C, C[bad_p].shape).astype(np.float32)
                
            # PSO: On attire toutes les populations (sauf la meilleure) vers le Global Best
            for p in range(1, P):
                p_idx = sorted_indices[p]
                C[p_idx] = 0.9 * C[p_idx] + 0.1 * global_best_C
                
            # 3. Anti-dégénérescence (Chasse aux centroïdes morts)
            N_total = X_data.shape[0]
            for p in range(P):
                dead_mask = host_counts[p] == 0
                if np.any(dead_mask):
                    num_dead = np.sum(dead_mask)
                    # On ressuscite les morts en les plaçant sur des points aléatoires du dataset
                    random_indices = np.random.choice(N_total, num_dead, replace=False)
                    C[p][dead_mask] = X_data[random_indices]
                    
        await gpu_C.overwrite(C.flatten())
        
        iter_time = time.perf_counter() - start_iter
        total_time += iter_time
        
        best_inertia_history.append(best_inertia)
        
        print(f"  Iter {it+1}/{max_iters} | Temps: {iter_time:.2f}s | Inertie Min: {best_inertia:.2e} | Pop Gagnante: {best_pop} | Déplacement: {shift:.4f}")
        
        if tol > 0.0 and shift < tol:
            print(f"[K-Means Batched OOC] Convergence de la meilleure population atteinte en {it+1} itérations !")
            # On prolonge l'historique jusqu'à max_iters pour le plot
            while len(best_inertia_history) < max_iters:
                best_inertia_history.append(best_inertia)
            break
            
    print(f"[K-Means Batched OOC] Temps total de calcul : {total_time:.2f}s")
    # On renvoie la meilleure population et les inerties de toutes les populations
    return C[best_pop], host_inertia


@auto_sync
async def kmeans_spherical_classic(X_data, K: int, initial_C=None, max_iters: int = 20, tol: float = 1e-5, P: int = 16, backend="opencl", mutation_interval=3, mutation_rate=0.01):
    """
    Exécute le K-Means Sphérique (Distance Cosinus) en mode Batched Out-Of-Core Asynchrone avec Algorithme Génétique.
    X_data DOIT être pré-normalisé (norme L2 = 1).
    initial_C doit être de forme (P, K, D).
    """
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    N, D = X_data.shape
    # On génère le kernel sphérique via l'AST parser
    kernel = get_kmeans_spherical_multi_kernel(P, K, D)
    
    # 1. Initialisation des centroïdes
    if initial_C is not None:
        C = initial_C.copy().astype(np.float32)
    else:
        np.random.seed(42)
        random_indices = np.random.choice(N, P * K, replace=False)
        C = X_data[random_indices].reshape((P, K, D)).astype(np.float32)
    
    gpu_C = await AsyncBuffer.to_device(scheduler, C.flatten())
    
    # 2. Préparation du Memory Pool pour l'Upload
    MAX_CHUNK = 1_000_000 
    IN_FLIGHT_LIMIT = 4
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    
    chunk_size = min(N, MAX_CHUNK)
    dummy_x = np.zeros((chunk_size, D), dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_x.flatten()))
        
    gpu_sums = AsyncBuffer.empty_like(scheduler, np.zeros((P, K, D), dtype=np.float32).flatten())
    gpu_counts = AsyncBuffer.empty_like(scheduler, np.zeros((P, K), dtype=np.float32).flatten())
    gpu_inertia = AsyncBuffer.empty_like(scheduler, np.zeros(P, dtype=np.float32))
    
    num_chunks = math.ceil(N / MAX_CHUNK)
    
    # 3. Préparation des chunks (Transposition en SoA)
    # Identique à V2...Transposition SoA des Chunks en RAM...")
    prepared_chunks = []
    for i in range(num_chunks):
        start_N = i * MAX_CHUNK
        end_N = min((i + 1) * MAX_CHUNK, N)
        current_N = end_N - start_N
        
        chunk = X_data[start_N:end_N]
        # Transposition mémoire (SoA)
        if chunk.shape[0] == chunk_size:
            arr = np.ascontiguousarray(chunk.T).ravel()
        else:
            padded = np.zeros((chunk_size, D), dtype=np.float32)
            padded[:chunk.shape[0], :] = chunk
            arr = np.ascontiguousarray(padded.T).ravel()
            
        prepared_chunks.append(arr)
        
    total_time = 0.0
    
    # Historique de la meilleure inertie pour le benchmark
    best_inertia_history = []
    
    for it in range(max_iters):
        start_iter = time.perf_counter()
        
        # Remise à zéro des accumulateurs
        await gpu_sums.overwrite(np.zeros((P, K, D), dtype=np.float32).flatten())
        await gpu_counts.overwrite(np.zeros((P, K), dtype=np.float32).flatten())
        await gpu_inertia.overwrite(np.zeros(P, dtype=np.float32))
        
        active_tasks = set()
        
        for i in range(num_chunks):
            start_N = i * MAX_CHUNK
            end_N = min((i + 1) * MAX_CHUNK, N)
            current_N = end_N - start_N
            
            res_extract = prepared_chunks[i]
            gpu_X = await pool.get()
            
            task = asyncio.create_task(process_chunk(
                kernel, res_extract, gpu_X, gpu_C, gpu_sums, gpu_counts, gpu_inertia, current_N, chunk_size, D, K, P, scheduler, pool
            ))
            active_tasks.add(task)
            
        if active_tasks:
            done, _ = await asyncio.wait(active_tasks)
            for coro in done:
                coro.result()
            
        # Téléchargement des résultats de l'itération
        host_sums = await gpu_sums.to_host()
        host_counts = await gpu_counts.to_host()
        host_inertia = await gpu_inertia.to_host()
        
        host_sums = host_sums.reshape((P, K, D))
        host_counts = host_counts.reshape((P, K))
        
        # On trouve la meilleure population (inertie min)
        best_pop = np.argmin(host_inertia)
        best_inertia = host_inertia[best_pop]
        
        # Mise à jour des centroïdes (et normalisation sphérique)
        C_new = np.zeros_like(C)
        for p in range(P):
            for k in range(K):
                if host_counts[p, k] > 0:
                    c_mean = host_sums[p, k] / host_counts[p, k]
                    # NORMALISATION SPHÉRIQUE
                    norm = np.linalg.norm(c_mean)
                    if norm > 1e-12:
                        C_new[p, k] = c_mean / norm
                    else:
                        C_new[p, k] = C[p, k]
                else:
                    C_new[p, k] = C[p, k]
                
        # Le déplacement moyen n'a de sens que sur la meilleure population
        shift = np.linalg.norm(C_new[best_pop] - C[best_pop])
        C[:] = C_new[:]
        
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
                
            # PSO: On attire toutes les populations vers le Global Best
            for p in range(1, P):
                p_idx = sorted_indices[p]
                C[p_idx] = 0.9 * C[p_idx] + 0.1 * global_best_C
                # Re-normalisation post-PSO
                norms = np.linalg.norm(C[p_idx], axis=1, keepdims=True)
                C[p_idx] = C[p_idx] / np.where(norms > 1e-12, norms, 1.0)
                
            # 3. Anti-dégénérescence (Chasse aux centroïdes morts)
            N_total = X_data.shape[0]
            for p in range(P):
                dead_mask = host_counts[p] == 0
                if np.any(dead_mask):
                    num_dead = np.sum(dead_mask)
                    random_indices = np.random.choice(N_total, num_dead, replace=False)
                    # Les points X_data sont déjà normalisés, on peut les copier directement
                    C[p][dead_mask] = X_data[random_indices]
                    
        await gpu_C.overwrite(C.flatten())
        
        iter_time = time.perf_counter() - start_iter
        total_time += iter_time
        
        best_inertia_history.append(best_inertia)
        
        print(f"  Iter {it+1}/{max_iters} | Temps: {iter_time:.2f}s | Inertie Min: {best_inertia:.2e} | Pop Gagnante: {best_pop} | Déplacement: {shift:.4f}")
        
        if tol > 0.0 and shift < tol:
            print(f"[K-Means Batched OOC] Convergence de la meilleure population atteinte en {it+1} itérations !")
            # On prolonge l'historique jusqu'à max_iters pour le plot
            while len(best_inertia_history) < max_iters:
                best_inertia_history.append(best_inertia)
            break
            
    print(f"[K-Means Batched OOC] Temps total de calcul : {total_time:.2f}s")
    # On renvoie la meilleure population et les inerties de toutes les populations
    return C[best_pop], host_inertia


@auto_sync
async def kmeans_spherical_mini(X_data, K: int, initial_C=None, batch_size: int = 10000, 
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
    best_C_exact, inertias_exact = await kmeans_spherical_classic(X_data, K, initial_C=C, max_iters=1, P=P, backend=backend)
    
    return best_C_exact, inertias_exact, total_time
