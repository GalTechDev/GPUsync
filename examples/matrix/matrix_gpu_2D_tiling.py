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

@gpusync_kernel
def matrix_mul_2d_tiling(A: list, B: list, C: list, M: int, K: int, N: int):
    # Un Thread Block (Work-Group) contient 16x16 = 256 threads
    local_col: int = get_local_id(0)
    local_row: int = get_local_id(1)
    
    # Le Bloc va calculer une tuile C de 64x64 !
    # Chaque thread calcule 4x4 = 16 pixels.
    global_col_start: int = get_group_id(0) * 64
    global_row_start: int = get_group_id(1) * 64
    
    # Allocation du Cache L1 (Local Memory)
    Asub = local_memory((64, 16), "float")
    Bsub = local_memory((16, 64), "float")
    
    # Les 16 REGISTRES D'ACCUMULATION (2D Register Tiling)
    c00: float = 0.0; c01: float = 0.0; c02: float = 0.0; c03: float = 0.0
    c10: float = 0.0; c11: float = 0.0; c12: float = 0.0; c13: float = 0.0
    c20: float = 0.0; c21: float = 0.0; c22: float = 0.0; c23: float = 0.0
    c30: float = 0.0; c31: float = 0.0; c32: float = 0.0; c33: float = 0.0
    
    tid: int = local_row * 16 + local_col
    
    num_tiles: int = K // 16
    for t in range(num_tiles):
        # 1. Chargement Collaboratif: Les 256 threads chargent 1024 floats de Asub (4 par thread)
        for i in range(4):
            idx_a: int = tid + i * 256
            r_a: int = idx_a // 16
            c_a: int = idx_a % 16
            Asub[r_a, c_a] = A[(global_row_start + r_a) * K + (t * 16 + c_a)]
            
        # B: 1024 floats.
        for i in range(4):
            idx_b: int = tid + i * 256
            r_b: int = idx_b // 64
            c_b: int = idx_b % 64
            Bsub[r_b, c_b] = B[(t * 16 + r_b) * N + (global_col_start + c_b)]
            
        barrier(CLK_LOCAL_MEM_FENCE)
        
        # 2. Calcul Intensif dans les Registres Physiques
        for k in range(16):
            # Chaque thread lit 4 valeurs de Asub et 4 valeurs de Bsub
            # Pour calculer ses 16 registres sans faire d'appels mémoires !
            a0: float = Asub[local_row * 4 + 0, k]
            a1: float = Asub[local_row * 4 + 1, k]
            a2: float = Asub[local_row * 4 + 2, k]
            a3: float = Asub[local_row * 4 + 3, k]
            
            b0: float = Bsub[k, local_col * 4 + 0]
            b1: float = Bsub[k, local_col * 4 + 1]
            b2: float = Bsub[k, local_col * 4 + 2]
            b3: float = Bsub[k, local_col * 4 + 3]
            
            # Les 16 FMA (Fused Multiply-Add)
            c00 += a0 * b0; c01 += a0 * b1; c02 += a0 * b2; c03 += a0 * b3
            c10 += a1 * b0; c11 += a1 * b1; c12 += a1 * b2; c13 += a1 * b3
            c20 += a2 * b0; c21 += a2 * b1; c22 += a2 * b2; c23 += a2 * b3
            c30 += a3 * b0; c31 += a3 * b1; c32 += a3 * b2; c33 += a3 * b3
            
        barrier(CLK_LOCAL_MEM_FENCE)
        
    # 3. Sauvegarde dans la Matrice C (VRAM globale)
    # Dépliage manuel pour éviter les boucles For en AST
    base_row: int = global_row_start + local_row * 4
    base_col: int = global_col_start + local_col * 4
    
    C[(base_row + 0) * N + (base_col + 0)] = c00
    C[(base_row + 0) * N + (base_col + 1)] = c01
    C[(base_row + 0) * N + (base_col + 2)] = c02
    C[(base_row + 0) * N + (base_col + 3)] = c03
    
    C[(base_row + 1) * N + (base_col + 0)] = c10
    C[(base_row + 1) * N + (base_col + 1)] = c11
    C[(base_row + 1) * N + (base_col + 2)] = c12
    C[(base_row + 1) * N + (base_col + 3)] = c13
    
    C[(base_row + 2) * N + (base_col + 0)] = c20
    C[(base_row + 2) * N + (base_col + 1)] = c21
    C[(base_row + 2) * N + (base_col + 2)] = c22
    C[(base_row + 2) * N + (base_col + 3)] = c23
    
    C[(base_row + 3) * N + (base_col + 0)] = c30
    C[(base_row + 3) * N + (base_col + 1)] = c31
    C[(base_row + 3) * N + (base_col + 2)] = c32
    C[(base_row + 3) * N + (base_col + 3)] = c33


async def process_block(i, j, gpu_A, b_sub, K, CHUNK_M, CHUNK_N, start_M, start_N, scheduler, pool):
    gpu_B = await AsyncBuffer.to_device(scheduler, b_sub)
    gpu_C = await pool.get()
    
    # Important: global_size doit refléter le nombre de blocs, pas le nombre de threads totaux !
    # En OpenCL, on passera un global_size divisé par 4 (car chaque thread fait 4 pixels).
    # Wait, pyopencl global_size est le nombre de THREADS !
    # Si le chunk fait 8192 x 8192.
    # On a besoin de (8192 / 4) threads = 2048 threads en X et Y.
    # Donc global_size = (CHUNK_N // 4, CHUNK_M // 4)
    await matrix_mul_2d_tiling(gpu_A, gpu_B, gpu_C, CHUNK_M, K, CHUNK_N, global_size=(CHUNK_N // 4, CHUNK_M // 4), scheduler=scheduler)
    
    result = await gpu_C.to_host()
    pool.put_nowait(gpu_C)
    return start_M, start_N, CHUNK_M, CHUNK_N, result

async def matmul_opencl_2d(A: np.ndarray, B: np.ndarray, scheduler: Scheduler) -> np.ndarray:
    assert A.shape[1] == B.shape[0]
    
    orig_M, orig_K = A.shape
    _, orig_N = B.shape
    
    # Zero-Padding pour multiples de 64 (La tuile fait 64x64)
    pad_M = (64 - (orig_M % 64)) % 64
    pad_K = (16 - (orig_K % 16)) % 16
    pad_N = (64 - (orig_N % 64)) % 64
    
    if pad_M > 0 or pad_K > 0:
        A = np.pad(A, ((0, pad_M), (0, pad_K)), mode='constant')
    if pad_K > 0 or pad_N > 0:
        B = np.pad(B, ((0, pad_K), (0, pad_N)), mode='constant')
        
    M, K = A.shape
    _, N = B.shape
    
    MAX_CHUNK = 8192 * 2
    
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
    scheduler = Scheduler(backend="opencl")
    #scheduler = Scheduler(backend="cuda")
    await scheduler.initialize()

    # Mêmes dimensions gigantesques de 16384*2 x 16384*2
    M = 16384*2
    K = 16384*2
    N = 16384*2
    
    print("==================================================")
    print(f"OPENCL 2D REGISTER TILING (VRAM Wall Breaker)")
    print(f"Calcul de C ({M}x{N}) = A ({M}x{K}) x B ({K}x{N})")
    print("==================================================")

    np.random.seed(42)
    print("[1] Génération des Matrices en RAM...")
    A = np.random.random((M, K)).astype(np.float32)
    B = np.random.random((K, N)).astype(np.float32)
    
    print(f"[2] Démarrage du Streaming Universel...")
    start_time = time.perf_counter()
    
    C = await matmul_opencl_2d(A, B, scheduler)
            
    total_time = time.perf_counter() - start_time
    giga_flops = ((2.0 * M * N * K) / total_time) / 1e12
    
    print(f"\n\n[SUCCESS] Matrix Multiplication 2D Tiling Terminée !")
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
