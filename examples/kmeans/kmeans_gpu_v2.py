import time
import asyncio
import numpy as np
import sys
import os
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.compiler import gpusync_kernel
from gpusync.api.opencl import get_local_id, local_memory, barrier, CLK_LOCAL_MEM_FENCE
from gpusync.utils.bg_loop import auto_sync

# Dummy functions for Python AST parsing (pour l'éditeur)
def get_global_id(dim): return 0
def get_local_id(dim): return 0
def atomic_add(target, val): pass
def atomic_add_float(target, val): pass
def atomic_add_float_global(target, val): pass
def local_memory(shape, dtype): return []
def sub_group_reduce_add(val): return val
def get_sub_group_local_id(): return 0
def get_global_id(dim): return 0
def barrier(flags): pass
CLK_LOCAL_MEM_FENCE = 0


async def process_chunk(kernel_func, x_sub, gpu_X, gpu_C, gpu_sums, gpu_counts, N_chunk, chunk_stride, D, K, scheduler, pool):
    # Overwrite le buffer VRAM existant avec les nouvelles données
    await gpu_X.overwrite(x_sub)
    
    # Coarsening factor
    COARSE_FACTOR = 4
    local_sz_val = 256
    num_threads_needed = math.ceil(N_chunk / COARSE_FACTOR)
    global_aligned = math.ceil(num_threads_needed / local_sz_val) * local_sz_val
    
    await kernel_func(gpu_X, gpu_C, gpu_sums, gpu_counts, N_chunk, chunk_stride, D, K, global_size=(global_aligned,), local_size=(local_sz_val,), scheduler=scheduler)
    
    # On rend le buffer au pool
    pool.put_nowait(gpu_X)


@auto_sync
async def kmeans_out_of_core_v2(X_data, K: int, initial_C=None, max_iters: int = 20, tol: float = 1e-5, backend="cuda"):
    """
    Exécute le K-Means en mode Out-Of-Core Asynchrone.
    """
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    N, D = X_data.shape
    
    # --- DÉFINITION DU KERNEL AVEC INJECTION JIT (CONSTANTES AST) ---
    KD = K * D
    LOCAL_SZ = 256  # On repasse à 256, la taille de sous-groupe classique
    COARSE_FACTOR = 4
    ITERS_KD = math.ceil(KD / LOCAL_SZ)
    ITERS_K = math.ceil(K / LOCAL_SZ)
    
    @gpusync_kernel(constants={"KD_CONST": KD, "K_CONST": K, "D_CONST": D, "LOCAL_SZ": LOCAL_SZ, "COARSE_FACTOR": COARSE_FACTOR, "ITERS_KD": ITERS_KD, "ITERS_K": ITERS_K})
    def kmeans_kernel(ctx: int, X: list, C: list, sums: list, counts: list, N: int, stride: int, D: int, K: int):
        local_idx: int = get_local_id(0)
        global_idx: int = get_global_id(0)
        
        # Shared memory avec allocation parfaite
        C_shared = local_memory((KD_CONST,), "float")
        local_sums = local_memory((KD_CONST,), "float")
        local_counts = local_memory((K_CONST,), "float")
        
        # Boucles FOR (déroulables par le compilateur) au lieu du WHILE dynamique lent
        for i in range(0, ITERS_KD):
            offset: int = i * LOCAL_SZ
            if local_idx + offset < KD_CONST:
                C_shared[local_idx + offset] = C[local_idx + offset]
                local_sums[local_idx + offset] = 0.0
                
        for i_k in range(0, ITERS_K):
            offset_k: int = i_k * LOCAL_SZ
            if local_idx + offset_k < K_CONST:
                local_counts[local_idx + offset_k] = 0.0
                
        barrier(CLK_LOCAL_MEM_FENCE)
        
        ctx_base: int = global_idx * COARSE_FACTOR
        
        if ctx_base < N:
            for p in range(0, COARSE_FACTOR):
                ctx_point: int = ctx_base + p
                if ctx_point < N:
                    min_dist: float = 1e30
                    best_k: int = 0
                    
                    # On garde K et D en variables pures (non-constantes) pour empêcher le compilateur
                    # de faire un "Loop Unrolling" géant qui ferait exploser les registres (Spilling).
                    for k in range(0, K):
                        dist: float = 0.0
                        for d in range(0, D):
                            diff: float = X[d * stride + ctx_point] - C_shared[k * D + d]
                            dist += diff * diff
                            
                        if dist < min_dist:
                            min_dist = dist
                            best_k = k
                            
                    # SUB-GROUP WARP SHUFFLING : Zéro Contention
                    for k in range(0, K):
                        match: float = 0.0
                        if best_k == k:
                            match = 1.0
                            
                        total_matches: float = sub_group_reduce_add(match)
                        
                        if total_matches > 0.0:
                            if get_sub_group_local_id() == 0:
                                atomic_add_float(local_counts[k], total_matches)
                                
                            for d in range(0, D):
                                val: float = 0.0
                                if best_k == k:
                                    val = X[d * stride + ctx_point]
                                total_val: float = sub_group_reduce_add(val)
                                
                                if get_sub_group_local_id() == 0:
                                    atomic_add_float(local_sums[k * D + d], total_val)
                
        barrier(CLK_LOCAL_MEM_FENCE)
        
        # Rapport vers VRAM avec boucles FOR (déroulables)
        for j in range(0, ITERS_KD):
            offset2: int = j * LOCAL_SZ
            if local_idx + offset2 < KD_CONST:
                atomic_add_float_global(sums[local_idx + offset2], local_sums[local_idx + offset2])
                
        for j_k in range(0, ITERS_K):
            offset_k2: int = j_k * LOCAL_SZ
            if local_idx + offset_k2 < K_CONST:
                atomic_add_float_global(counts[local_idx + offset_k2], local_counts[local_idx + offset_k2])
            
    # --- FIN DE LA DÉFINITION DU KERNEL ---
    
    # 1. Initialisation des centroïdes
    if initial_C is not None:
        C = initial_C.copy().astype(np.float32)
    else:
        np.random.seed(42)
        random_indices = np.random.choice(N, K, replace=False)
        C = X_data[random_indices].copy().astype(np.float32)
    
    gpu_C = await AsyncBuffer.to_device(scheduler, C.flatten())
    
    # 2. Préparation du Memory Pool pour l'Upload
    MAX_CHUNK = 1_000_000 # 5 Millions de points par passe (comme en V1)
    IN_FLIGHT_LIMIT = 4
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    
    chunk_size = min(N, MAX_CHUNK)
    dummy_x = np.zeros((chunk_size, D), dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_x.flatten()))
        
    gpu_sums = AsyncBuffer.empty_like(scheduler, np.zeros((K, D), dtype=np.float32).flatten())
    gpu_counts = AsyncBuffer.empty_like(scheduler, np.zeros(K, dtype=np.float32))
    
    num_chunks = math.ceil(N / MAX_CHUNK)
    
    print(f"[K-Means OOC] Démarrage... N={N}, D={D}, K={K}")
    
    print("[K-Means OOC] Préparation et Transposition SoA des Chunks en RAM...")
    prepared_chunks = []
    for i in range(num_chunks):
        start_N = i * MAX_CHUNK
        end_N = min((i + 1) * MAX_CHUNK, N)
        sub = X_data[start_N:end_N]
        if sub.shape[0] == chunk_size:
            arr = np.ascontiguousarray(sub.T).ravel()
        else:
            padded = np.zeros((chunk_size, D), dtype=np.float32)
            padded[:sub.shape[0], :] = sub
            arr = np.ascontiguousarray(padded.T).ravel()
        prepared_chunks.append(arr)
        
    total_time = 0
    
    for it in range(max_iters):
        start_iter = time.perf_counter()
        
        # Remise à zéro des accumulateurs
        await gpu_sums.overwrite(np.zeros((K, D), dtype=np.float32).flatten())
        await gpu_counts.overwrite(np.zeros(K, dtype=np.float32))
        
        active_tasks = set()
        
        for i in range(num_chunks):
            start_N = i * MAX_CHUNK
            end_N = min((i + 1) * MAX_CHUNK, N)
            current_N = end_N - start_N
            
            res_extract = prepared_chunks[i]
            gpu_X = await pool.get()
            
            task = asyncio.create_task(process_chunk(
                kmeans_kernel, res_extract, gpu_X, gpu_C, gpu_sums, gpu_counts, current_N, chunk_size, D, K, scheduler, pool
            ))
            active_tasks.add(task)
            
            while len(active_tasks) >= IN_FLIGHT_LIMIT:
                done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                for coro in done:
                    coro.result()
        
        while len(active_tasks) > 0:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for coro in done:
                coro.result()
            
        # Téléchargement des résultats de l'itération
        host_sums = await gpu_sums.to_host()
        host_counts = await gpu_counts.to_host()
        
        host_sums = host_sums.reshape((K, D))
        
        # Mise à jour des centroïdes
        C_new = np.zeros_like(C)
        for k in range(K):
            if host_counts[k] > 0:
                C_new[k] = host_sums[k] / host_counts[k]
            else:
                C_new[k] = C[k]
                
        shift = np.linalg.norm(C_new - C)
        C[:] = C_new[:]
        await gpu_C.overwrite(C.flatten())
        
        iter_time = time.perf_counter() - start_iter
        total_time += iter_time
        
        print(f"  Iter {it+1}/{max_iters} | Temps: {iter_time:.2f}s | Déplacement moyen: {shift:.4f}")
        
        if tol > 0.0 and shift < tol:
            print(f"[K-Means OOC] Convergence atteinte en {it+1} itérations !")
            break
            
    print(f"[K-Means OOC] Temps total de calcul : {total_time:.2f}s")
    return C, total_time
