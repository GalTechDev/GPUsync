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
from gpusync.cuda import wmma

@gpusync_kernel
def wmma_raw_kernel(A: list, B: list, C: list, M: int, K: int, N: int):
    """
    WMMA pur (Sans Shared Memory, reposant sur le Cache L2 de l'architecture).
    """
    warpId: int = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y // 32) + (threadIdx.y * blockDim.x + threadIdx.x) // 32
    
    tilesX: int = N // 16
    tilesY: int = M // 16
    tileX: int = warpId % tilesX
    tileY: int = warpId // tilesX
    
    if tileX < tilesX and tileY < tilesY:
        warpRow: int = tileY * 16
        warpCol: int = tileX * 16
        
        a_frag = wmma.fragment("matrix_a", 16, 16, 8, "tf32", "row_major")
        b_frag = wmma.fragment("matrix_b", 16, 16, 8, "tf32", "row_major")
        c_frag = wmma.fragment("accumulator", 16, 16, 8, "float")
        
        wmma.fill_fragment(c_frag, 0.0)
        
        for k in range(0, K, 8):
            wmma.load_matrix_sync(a_frag, A + warpRow * K + k, K)
            wmma.load_matrix_sync(b_frag, B + k * N + warpCol, N)
            wmma.mma_sync(c_frag, a_frag, b_frag, c_frag)
            
        wmma.store_matrix_sync(C + warpRow * N + warpCol, c_frag, N, "mem_row_major")

async def process_block(i, j, gpu_A, b_sub, K, CHUNK_M, CHUNK_N, start_M, start_N, scheduler, pool):
    gpu_B = await AsyncBuffer.to_device(scheduler, b_sub)
    gpu_C = await pool.get()
    
    await wmma_raw_kernel(gpu_A, gpu_B, gpu_C, CHUNK_M, K, CHUNK_N, global_size=(CHUNK_N, CHUNK_M), scheduler=scheduler)
    
    result = await gpu_C.to_host()
    pool.put_nowait(gpu_C)
    return start_M, start_N, CHUNK_M, CHUNK_N, result

async def matmul_cuda_wmma(A: np.ndarray, B: np.ndarray, scheduler: Scheduler) -> np.ndarray:
    assert A.shape[1] == B.shape[0], "Les dimensions internes (K) doivent correspondre."
    
    orig_M, orig_K = A.shape
    _, orig_N = B.shape
    
    # 1. Zero-Padding pour atteindre des multiples de 16
    pad_M = (16 - (orig_M % 16)) % 16
    pad_K = (16 - (orig_K % 16)) % 16
    pad_N = (16 - (orig_N % 16)) % 16
    
    if pad_M > 0 or pad_K > 0:
        A = np.pad(A, ((0, pad_M), (0, pad_K)), mode='constant', constant_values=0.0)
    if pad_K > 0 or pad_N > 0:
        B = np.pad(B, ((0, pad_K), (0, pad_N)), mode='constant', constant_values=0.0)
        
    M, K = A.shape
    _, N = B.shape
    
    # Taille de tuile maximale poussée à 8192 pour gorger le GPU et réduire l'overhead CPU
    MAX_CHUNK = 8192
    
    IN_FLIGHT_LIMIT = 4
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    dummy_out = np.zeros(MAX_CHUNK * MAX_CHUNK, dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_out))
        
    C_padded = np.zeros((M, N), dtype=np.float32)
    
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
                    res = res[:cm * cn]
                    C_padded[sm:sm+cm, sn:sn+cn] = res.reshape((cm, cn))
                    completed_blocks += 1
                    sys.stdout.write(f"\rProgression : {completed_blocks}/{total_blocks} blocs calculés")
                    sys.stdout.flush()

    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            sm, sn, cm, cn, res = coro.result()
            res = res[:cm * cn]
            C_padded[sm:sm+cm, sn:sn+cn] = res.reshape((cm, cn))
            completed_blocks += 1
            sys.stdout.write(f"\rProgression : {completed_blocks}/{total_blocks} blocs calculés")
            sys.stdout.flush()
            
    return C_padded[:orig_M, :orig_N]

async def main():
    scheduler = Scheduler(backend="cuda")
    await scheduler.initialize()

    # Matrice de 40960 x 40960 (Environ 20 Go de RAM, 137 TeraFLOPS de calcul)
    M = 40960
    K = 40960
    N = 40960
    
    print("==================================================")
    print(f"CUDA UNIVERSAL WMMA (Tensor Cores Dynamiques)")
    print(f"Calcul de C ({M}x{N}) = A ({M}x{K}) x B ({K}x{N})")
    print("==================================================")

    np.random.seed(42)
    print("[1] Génération des Matrices en RAM...")
    A = np.random.random((M, K)).astype(np.float32)
    B = np.random.random((K, N)).astype(np.float32)
    
    print(f"[2] Démarrage du Streaming Tensor Cores...")
    start_time = time.perf_counter()
    
    C = await matmul_cuda_wmma(A, B, scheduler)
            
    total_time = time.perf_counter() - start_time
    giga_flops = ((2.0 * M * N * K) / total_time) / 1e12
    
    print(f"\n\n[SUCCESS] Matrix Multiplication WMMA Terminée !")
    print(f"-> Taille finale du Résultat : {C.shape}")
    print(f"-> Temps Total               : {total_time:.2f} secondes")
    print(f"-> Débit de Calcul Effectif  : {giga_flops:.2f} TeraFLOPS")
    
    print("\n[Vérification de l'intégrité Mathématique...]")
    test_size = 100
    numpy_C_tile = np.dot(A[:test_size, :], B[:, :test_size])
    diff = np.abs(C[:test_size, :test_size] - numpy_C_tile).max()
    print(f"-> Erreur Maximale par rapport au CPU NumPy : {diff:.6e}")

if __name__ == "__main__":
    asyncio.run(main())
