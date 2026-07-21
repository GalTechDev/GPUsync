import time
import asyncio
import numpy as np
import sys
import os
import math
import cupy as cp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.executor import await_event

async def process_block(i, j, gpu_A, b_sub, K, CHUNK_M, CHUNK_N, start_M, start_N, scheduler, pool):
    gpu_B = await AsyncBuffer.to_device(scheduler, b_sub)
    gpu_C = await pool.get()
    
    device_manager = scheduler.get_best_device()
    device = device_manager.device
    queue = device_manager.get_queue()
    
    with device:
        # gpu_A and gpu_B are flattened 1D arrays on the GPU.
        # We need to reshape them to their matrix dimensions.
        A_mat = gpu_A.cl_buffer.reshape((CHUNK_M, K))
        B_mat = gpu_B.cl_buffer.reshape((K, CHUNK_N))
        
        # Le cast en VRAM causait le crash de 17 Go !
        # Maintenant, A_mat et B_mat ARRIVENT déjà en float16. On calcule directement.
        C_result = cp.matmul(A_mat, B_mat)
        
        # On calcule dynamiquement la taille du chunk pré-alloué
        MAX_CHUNK = int(math.sqrt(gpu_C.cl_buffer.size))
        C_full = gpu_C.cl_buffer.reshape((MAX_CHUNK, MAX_CHUNK))
        C_full[:CHUNK_M, :CHUNK_N] = C_result
        
    # On délègue l'attente et le téléchargement à un thread système.
    # Ainsi, cudaStreamSynchronize bloquera le thread système sans bloquer l'Event Loop Python !
    def download_result():
        with device:
            if queue:
                queue.synchronize()
            else:
                cp.cuda.Stream.null.synchronize()
            return gpu_C.cl_buffer.get()
            
    result_1d = await asyncio.to_thread(download_result)
    result = result_1d.reshape((MAX_CHUNK, MAX_CHUNK))
    
    pool.put_nowait(gpu_C)
    return start_M, start_N, CHUNK_M, CHUNK_N, result

async def matmul_cuda_cublas(A: np.ndarray, B: np.ndarray, scheduler: Scheduler) -> np.ndarray:
    assert A.shape[1] == B.shape[0], "Les dimensions internes (K) doivent correspondre."
    
    M, K = A.shape
    _, N = B.shape
    
    MAX_CHUNK = 8192*4
    
    IN_FLIGHT_LIMIT = 1
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    # Allocation en float16 pour économiser la VRAM !
    dummy_out = np.zeros(MAX_CHUNK * MAX_CHUNK, dtype=np.float16)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_out))
        
    C_padded = np.zeros((M, N), dtype=np.float16)
    
    num_chunks_M = math.ceil(M / MAX_CHUNK)
    num_chunks_N = math.ceil(N / MAX_CHUNK)
    total_blocks = num_chunks_M * num_chunks_N
    completed_blocks = 0
    active_tasks = set()
    
    for i in range(num_chunks_M):
        start_M = i * MAX_CHUNK
        end_M = min((i + 1) * MAX_CHUNK, M)
        current_CHUNK_M = end_M - start_M
        
        a_sub = await asyncio.to_thread(lambda s=start_M, e=end_M: A[s:e, :].flatten())
        gpu_A = await AsyncBuffer.to_device(scheduler, a_sub)
        
        for j in range(num_chunks_N):
            start_N = j * MAX_CHUNK
            end_N = min((j + 1) * MAX_CHUNK, N)
            current_CHUNK_N = end_N - start_N
            
            b_sub = await asyncio.to_thread(lambda s=start_N, e=end_N: B[:, s:e].copy().flatten())
            
            task = asyncio.create_task(process_block(
                i, j, gpu_A, b_sub, K, current_CHUNK_M, current_CHUNK_N, start_M, start_N, scheduler, pool
            ))
            active_tasks.add(task)
            
            while len(active_tasks) >= IN_FLIGHT_LIMIT:
                done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                for coro in done:
                    sm, sn, cm, cn, res = coro.result()
                    # Reshape the flat result array to the 2D chunk size
                    res_mat = res.reshape((MAX_CHUNK, MAX_CHUNK))[:cm, :cn]
                    C_padded[sm:sm+cm, sn:sn+cn] = res_mat
                    completed_blocks += 1
                    sys.stdout.write(f"\rProgression : {completed_blocks}/{total_blocks} blocs calculés")
                    sys.stdout.flush()

    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            sm, sn, cm, cn, res = coro.result()
            res_mat = res.reshape((MAX_CHUNK, MAX_CHUNK))[:cm, :cn]
            C_padded[sm:sm+cm, sn:sn+cn] = res_mat
            completed_blocks += 1
            sys.stdout.write(f"\rProgression : {completed_blocks}/{total_blocks} blocs calculés")
            sys.stdout.flush()
            
    return C_padded

async def main():
    scheduler = Scheduler(backend="cublas")
    await scheduler.initialize()

    # Matrice Titanesque (32768x32768 = 12.8 Go Total)
    M, K, N = 16384*2, 16384*2, 16384*2
    
    print("==================================================")
    print(f"CHEAT CODE: cuBLAS Out-Of-Core (Tensor Cores TF32)")
    print(f"Calcul de C ({M}x{N}) = A ({M}x{K}) x B ({K}x{N})")
    print("==================================================")

    np.random.seed(42)
    print("[1] Génération des Matrices en RAM...")
    # On génère directement en float16 pour éviter d'exploser la RAM et la VRAM !
    A = np.random.random((M, K)).astype(np.float16)
    B = np.random.random((K, N)).astype(np.float16)
    
    print(f"[2] Démarrage du Streaming cuBLAS Out-Of-Core...")
    start_time = time.perf_counter()
    
    C = await matmul_cuda_cublas(A, B, scheduler)
            
    total_time = time.perf_counter() - start_time
    giga_flops = ((2.0 * M * N * K) / total_time) / 1e12
    
    print(f"\n\n[SUCCESS] Matrix Multiplication cuBLAS Terminée !")
    print(f"-> Taille finale du Résultat : {C.shape}")
    print(f"-> Temps Total               : {total_time:.2f} secondes")
    print(f"-> Débit de Calcul Effectif  : {giga_flops:.2f} TeraFLOPS")
    
    print("\n[Vérification de l'intégrité Mathématique...]")
    test_size = 100
    numpy_C_tile = np.dot(A[:test_size, :].astype(np.float32), B[:, :test_size].astype(np.float32))
    diff = np.abs(C[:test_size, :test_size].astype(np.float32) - numpy_C_tile).max()
    print(f"-> Erreur Maximale par rapport au CPU NumPy : {diff:.6e}")

if __name__ == "__main__":
    asyncio.run(main())
