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

# Dummy functions for Python AST parsing
def get_global_id(dim): return 0
def get_local_id(dim): return 0
def atomic_add(target, val): pass
def local_memory(shape, dtype): return []
def barrier(flags): pass
CLK_LOCAL_MEM_FENCE = 0

@gpusync_kernel
def kmeans_kernel(ctx: int, X: list, C: list, sums: list, counts: list, N: int, stride: int, D: int, K: int):
    local_idx: int = get_local_id(0)
    
    # Shared memory pour C, sums, et counts
    C_shared = local_memory((1024,), "float")
    local_sums = local_memory((1024,), "float")
    local_counts = local_memory((16,), "float")
    
    # Chargement de C et Initialisation de sums/counts à 0
    for i in range(0, 4):
        offset: int = i * 256
        if local_idx + offset < 1024:
            C_shared[local_idx + offset] = C[local_idx + offset]
            local_sums[local_idx + offset] = 0.0
            
    if local_idx < 16:
        local_counts[local_idx] = 0.0
            
    barrier(CLK_LOCAL_MEM_FENCE)
    
    if ctx < N:
        min_dist: float = 1e30
        best_k: int = 0
        
        for k in range(0, K):
            dist: float = 0.0
            for d in range(0, D):
                diff: float = X[d * stride + ctx] - C_shared[k * D + d]
                dist += diff * diff
                
            if dist < min_dist:
                min_dist = dist
                best_k = k
                
        # Accumulation atomique dans la mémoire partagée (Ultra rapide)
        atomic_add(local_counts[best_k], 1)
        for d in range(0, D):
            atomic_add(local_sums[best_k * D + d], X[d * stride + ctx])
            
    barrier(CLK_LOCAL_MEM_FENCE)
    
    # Rapport des accumulateurs locaux vers la VRAM globale
    for i in range(0, 4):
        offset2: int = i * 256
        if local_idx + offset2 < 1024:
            atomic_add(sums[local_idx + offset2], local_sums[local_idx + offset2])
            
    if local_idx < 16:
        atomic_add(counts[local_idx], local_counts[local_idx])

async def process_chunk(x_sub, gpu_X, gpu_C, gpu_sums, gpu_counts, N_chunk, chunk_stride, D, K, scheduler, pool):
    # Overwrite le buffer VRAM existant avec les nouvelles données
    await gpu_X.overwrite(x_sub)
    
    # Lancement du Kernel
    await kmeans_kernel(gpu_X, gpu_C, gpu_sums, gpu_counts, N_chunk, chunk_stride, D, K, global_size=(N_chunk,), scheduler=scheduler)
    
    # On rend le buffer au pool
    pool.put_nowait(gpu_X)

from gpusync.utils.bg_loop import auto_sync

@auto_sync
async def kmeans_out_of_core(X_data, K: int, initial_C=None, max_iters: int = 10, backend="cuda"):
    """
    Exécute le K-Means en mode Out-Of-Core Asynchrone.
    X_data peut être un numpy array ou un numpy memmap !
    """
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    N, D = X_data.shape
    
    # 1. Initialisation des centroïdes
    if initial_C is not None:
        C = initial_C.copy().astype(np.float32)
    else:
        np.random.seed(42)
        random_indices = np.random.choice(N, K, replace=False)
        C = X_data[random_indices].copy().astype(np.float32)
    
    gpu_C = await AsyncBuffer.to_device(scheduler, C.flatten())
    
    # 2. Préparation du Memory Pool pour l'Upload
    MAX_CHUNK = 5_000_000 # 5 Millions de points par passe (env. 1.2 Go, passe largement dans VRAM)
    IN_FLIGHT_LIMIT = 2
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    
    # Si le dataset est plus petit que MAX_CHUNK, on ajuste
    chunk_size = min(N, MAX_CHUNK)
    dummy_x = np.zeros((chunk_size, D), dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_x.flatten()))
        
    gpu_sums = AsyncBuffer.empty_like(scheduler, np.zeros((K, D), dtype=np.float32).flatten())
    gpu_counts = AsyncBuffer.empty_like(scheduler, np.zeros(K, dtype=np.float32))
    
    num_chunks = math.ceil(N / MAX_CHUNK)
    
    print(f"[K-Means OOC] Démarrage... N={N}, D={D}, K={K}")
    print(f"[K-Means OOC] MAX_CHUNK={MAX_CHUNK}, IN_FLIGHT={IN_FLIGHT_LIMIT}")
    
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
                res_extract, gpu_X, gpu_C, gpu_sums, gpu_counts, current_N, chunk_size, D, K, scheduler, pool
            ))
            active_tasks.add(task)
            
            while len(active_tasks) >= IN_FLIGHT_LIMIT:
                done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                for coro in done:
                    coro.result() # Lève l'exception si le kernel a planté
        
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
                C_new[k] = C[k] # Si un cluster est vide (très rare), on le garde
                
        # Vérification convergence
        shift = np.linalg.norm(C_new - C)
        C[:] = C_new[:]
        await gpu_C.overwrite(C.flatten())
        
        iter_time = time.perf_counter() - start_iter
        total_time += iter_time
        
        print(f"  Iter {it+1}/{max_iters} | Temps: {iter_time:.2f}s | Déplacement moyen: {shift:.4f}")
        
        if shift < 1e-5:
            print(f"[K-Means OOC] Convergence atteinte en {it+1} itérations !")
            break
            
    print(f"[K-Means OOC] Temps total de calcul : {total_time:.2f}s")
    return C, total_time
