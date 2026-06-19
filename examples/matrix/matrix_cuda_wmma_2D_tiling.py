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
from gpusync.opencl import get_local_id
def get_group_id(dim): pass

@gpusync_kernel
def wmma_2d_tiling_kernel(A: list, B: list, C: list, M: int, K: int, N: int):
    # Un bloc contient local_size=(32, 4) threads, soit 4 Warps de 32 threads.
    # get_local_id(1) donne l'ID du Warp (0, 1, 2, 3)
    warpId: int = get_local_id(1)
    warpY: int = warpId // 2  # 0 ou 1
    warpX: int = warpId % 2   # 0 ou 1
    
    # Le Block entier traite une tuile C de 64x64
    # Chaque Warp traite une tuile de 32x32 au sein de cette tuile 64x64
    global_warp_row: int = get_group_id(1) * 64 + warpY * 32
    global_warp_col: int = get_group_id(0) * 64 + warpX * 32
    
    if global_warp_col < N and global_warp_row < M:
        # Chaque Warp gère 4 fragments C (2x2 tuiles de 16x16 = 32x32)
        c00 = wmma.fragment("accumulator", 16, 16, 8, "float")
        c01 = wmma.fragment("accumulator", 16, 16, 8, "float")
        c10 = wmma.fragment("accumulator", 16, 16, 8, "float")
        c11 = wmma.fragment("accumulator", 16, 16, 8, "float")
        
        wmma.fill_fragment(c00, 0.0)
        wmma.fill_fragment(c01, 0.0)
        wmma.fill_fragment(c10, 0.0)
        wmma.fill_fragment(c11, 0.0)
        
        # Les fragments A et B (2 de chaque par itération pour nourrir les 4 MACs)
        a0 = wmma.fragment("matrix_a", 16, 16, 8, "tf32", "row_major")
        a1 = wmma.fragment("matrix_a", 16, 16, 8, "tf32", "row_major")
        b0 = wmma.fragment("matrix_b", 16, 16, 8, "tf32", "row_major")
        b1 = wmma.fragment("matrix_b", 16, 16, 8, "tf32", "row_major")
        
        for k in range(0, K, 8):
            # Charger 2 blocs de A (Lignes HAUTES et BASSES du Warp)
            wmma.load_matrix_sync(a0, A + (global_warp_row + 0) * K + k, K)
            wmma.load_matrix_sync(a1, A + (global_warp_row + 16) * K + k, K)
            
            # Charger 2 blocs de B (Colonnes GAUCHES et DROITES du Warp)
            wmma.load_matrix_sync(b0, B + k * N + (global_warp_col + 0), N)
            wmma.load_matrix_sync(b1, B + k * N + (global_warp_col + 16), N)
            
            # 4 multiplications Tensor Core 16x16 (AI doublée !)
            wmma.mma_sync(c00, a0, b0, c00)
            wmma.mma_sync(c01, a0, b1, c01)
            wmma.mma_sync(c10, a1, b0, c10)
            wmma.mma_sync(c11, a1, b1, c11)
            
        # Sauvegarde Globale
        wmma.store_matrix_sync(C + (global_warp_row + 0) * N + (global_warp_col + 0), c00, N, "mem_row_major")
        wmma.store_matrix_sync(C + (global_warp_row + 0) * N + (global_warp_col + 16), c01, N, "mem_row_major")
        wmma.store_matrix_sync(C + (global_warp_row + 16) * N + (global_warp_col + 0), c10, N, "mem_row_major")
        wmma.store_matrix_sync(C + (global_warp_row + 16) * N + (global_warp_col + 16), c11, N, "mem_row_major")

async def process_block(i, j, gpu_A, b_sub, K, CHUNK_M, CHUNK_N, start_M, start_N, scheduler, pool):
    gpu_B = await AsyncBuffer.to_device(scheduler, b_sub)
    gpu_C = await pool.get()
    
    # local_size = (32, 4) -> 128 threads
    # grid = (CHUNK_N // 64, CHUNK_M // 64)
    # global_size = (grid_x * local_x, grid_y * local_y)
    grid_x = CHUNK_N // 64
    grid_y = CHUNK_M // 64
    global_size = (grid_x * 32, grid_y * 4)
    local_size = (32, 4)
    
    await wmma_2d_tiling_kernel(
        gpu_A, gpu_B, gpu_C, CHUNK_M, K, CHUNK_N, 
        global_size=global_size, local_size=local_size, scheduler=scheduler
    )
    
    result = await gpu_C.to_host()
    pool.put_nowait(gpu_C)
    return start_M, start_N, CHUNK_M, CHUNK_N, result

async def matmul_cuda_wmma_2d(A: np.ndarray, B: np.ndarray, scheduler: Scheduler) -> np.ndarray:
    assert A.shape[1] == B.shape[0], "Les dimensions internes (K) doivent correspondre."
    
    orig_M, orig_K = A.shape
    _, orig_N = B.shape
    
    # Padding pour que tout soit divisible par 64 (la tuile de Bloc)
    pad_M = (64 - (orig_M % 64)) % 64
    pad_K = (16 - (orig_K % 16)) % 16
    pad_N = (64 - (orig_N % 64)) % 64
    
    if pad_M > 0 or pad_K > 0:
        A = np.pad(A, ((0, pad_M), (0, pad_K)), mode='constant', constant_values=0.0)
    if pad_K > 0 or pad_N > 0:
        B = np.pad(B, ((0, pad_K), (0, pad_N)), mode='constant', constant_values=0.0)
        
    M, K = A.shape
    _, N = B.shape
    
    MAX_CHUNK = 8192*4
    
    IN_FLIGHT_LIMIT = 1
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

    # Matrice Titanesque (40960x40960 = 20 Go)
    M, K, N = 16384*2, 16384*2, 16384*2
    
    print("==================================================")
    print(f"ULTIMATE KERNEL: WMMA 2D REGISTER TILING (Tensor Cores)")
    print(f"Calcul de C ({M}x{N}) = A ({M}x{K}) x B ({K}x{N})")
    print("==================================================")

    np.random.seed(42)
    print("[1] Génération des Matrices en RAM...")
    A = np.random.random((M, K)).astype(np.float32)
    B = np.random.random((K, N)).astype(np.float32)
    
    print(f"[2] Démarrage du Streaming Tensor Cores 2D...")
    start_time = time.perf_counter()
    
    C = await matmul_cuda_wmma_2d(A, B, scheduler)
            
    total_time = time.perf_counter() - start_time
    giga_flops = ((2.0 * M * N * K) / total_time) / 1e12
    
    print(f"\n\n[SUCCESS] Matrix Multiplication WMMA 2D Terminée !")
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
