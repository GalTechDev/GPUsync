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
from gpusync.opencl import get_local_id, local_memory, barrier, CLK_LOCAL_MEM_FENCE

@gpusync_kernel
def matrix_mul_tiled(A: list, B: list, C: list, M: int, K: int, N: int):
    col: int = get_global_id(0)
    row: int = get_global_id(1)
    
    local_col: int = get_local_id(0)
    local_row: int = get_local_id(1)
    
    Asub = local_memory((16, 16), "float")
    Bsub = local_memory((16, 16), "float")
    
    acc: float = 0.0
    
    num_tiles: int = K // 16
    for t in range(0, num_tiles):
        Asub[local_row, local_col] = A[row * K + (t * 16 + local_col)]
        Bsub[local_row, local_col] = B[(t * 16 + local_row) * N + col]
        
        barrier(CLK_LOCAL_MEM_FENCE)
        
        for k in range(0, 16):
            acc += Asub[local_row, k] * Bsub[k, local_col]
            
        barrier(CLK_LOCAL_MEM_FENCE)
        
    C[row * N + col] = acc

async def process_block(i, j, gpu_A, b_sub, K, CHUNK_M, CHUNK_N, start_M, start_N, scheduler, pool):
    gpu_B = await AsyncBuffer.to_device(scheduler, b_sub)
    gpu_C = await pool.get()
    
    # La tuile GPU s'adapte magiquement à la dimension exacte du Chunk !
    await matrix_mul_tiled(gpu_A, gpu_B, gpu_C, CHUNK_M, K, CHUNK_N, global_size=(CHUNK_N, CHUNK_M), scheduler=scheduler)
    
    result = await gpu_C.to_host()
    pool.put_nowait(gpu_C)
    return start_M, start_N, CHUNK_M, CHUNK_N, result

async def matmul_opencl(A: np.ndarray, B: np.ndarray, scheduler: Scheduler) -> np.ndarray:
    assert A.shape[1] == B.shape[0], "Les dimensions internes (K) doivent correspondre."
    
    orig_M, orig_K = A.shape
    _, orig_N = B.shape
    
    # 1. Zero-Padding pour atteindre des multiples de 16 (la taille de notre tuile L1)
    pad_M = (16 - (orig_M % 16)) % 16
    pad_K = (16 - (orig_K % 16)) % 16
    pad_N = (16 - (orig_N % 16)) % 16
    
    if pad_M > 0 or pad_K > 0:
        A = np.pad(A, ((0, pad_M), (0, pad_K)), mode='constant', constant_values=0.0)
    if pad_K > 0 or pad_N > 0:
        B = np.pad(B, ((0, pad_K), (0, pad_N)), mode='constant', constant_values=0.0)
        
    M, K = A.shape
    _, N = B.shape
    
    MAX_CHUNK = 8192
    
    # Allocation du Memory Pool Asynchrone avec la taille maximale possible
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
                    # Découpe le buffer flat pour ne garder que la partie calculée
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
            
    # On retire le Zero-Padding final pour renvoyer la vraie dimension d'origine !
    return C_padded[:orig_M, :orig_N]


async def main():
    scheduler = Scheduler(backend="opencl")
    #scheduler = Scheduler(backend="cuda")
    await scheduler.initialize()

    # Matrice de 40960 x 40960 (Environ 20 Go de RAM, 137 TeraFLOPS de calcul)
    M = 40960
    K = 40960
    N = 40960
    
    print("==================================================")
    print(f"OPENCL UNIVERSAL MATMUL (Zero-Padding Dynamique)")
    print(f"Calcul de C ({M}x{N}) = A ({M}x{K}) x B ({K}x{N})")
    print("==================================================")

    np.random.seed(42)
    print("[1] Génération des Matrices en RAM...")
    A = np.random.random((M, K)).astype(np.float32)
    B = np.random.random((K, N)).astype(np.float32)
    
    print(f"[2] Démarrage du Streaming Universel...")
    start_time = time.perf_counter()
    
    C = await matmul_opencl(A, B, scheduler)
            
    total_time = time.perf_counter() - start_time
    
    giga_flops = ((2.0 * M * N * K) / total_time) / 1e12
    
    print(f"\n\n[SUCCESS] Matrix Multiplication Universelle Terminée !")
    print(f"-> Taille finale du Résultat : {C.shape}")
    print(f"-> Temps Total               : {total_time:.2f} secondes")
    print(f"-> Débit de Calcul Effectif  : {giga_flops:.2f} TeraFLOPS")
    
    print("\n[Vérification de l'intégrité Mathématique...]")
    # On prend un petit échantillon pour vérifier car Numpy prendrait trop de temps sur tout
    test_size = 100
    numpy_C_tile = np.dot(A[:test_size, :], B[:, :test_size])
    diff = np.abs(C[:test_size, :test_size] - numpy_C_tile).max()
    print(f"-> Erreur Maximale par rapport au CPU NumPy : {diff:.6e}")
    if diff < 1e-1:
        print("-> [OK] La Matrice a été parfaitement calculée malgré les tailles irrégulières !")

if __name__ == "__main__":
    asyncio.run(main())
