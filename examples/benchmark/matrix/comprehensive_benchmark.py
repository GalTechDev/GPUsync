import os
import sys
import time
import math
import asyncio
import numpy as np
import csv
import psutil
import gc

try:
    import torch
except ImportError:
    torch = None

import cupy as cp

# Ajouter le chemin racine au path Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer

# Imports de tous les algorithmes existants
import examples.matrix.matrix_cuda_double_buffering as mod_cublas
import examples.matrix.matrix_cuda_wmma_2D_tiling as mod_wmma_2d
import examples.matrix.matrix_cuda_wmma as mod_wmma
import examples.matrix.matrix_gpu_2D_tiling as mod_gpu_2d
import examples.matrix.matrix_gpu as mod_gpu

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

def run_pytorch(N):
    if not torch or not torch.cuda.is_available():
        return None, None, None
    torch.cuda.empty_cache()
    try:
        # On génère directement en float32 pour éviter le pic de RAM float64
        rng = np.random.default_rng(42) # On fixe la graine pour avoir exactement les mêmes matrices
        A_cpu = rng.random((N, N), dtype=np.float32).astype(np.float16)
        B_cpu = rng.random((N, N), dtype=np.float32).astype(np.float16)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Le chronomètre inclut le transfert PCIe (Aller)
        A = torch.tensor(A_cpu, device="cuda")
        B = torch.tensor(B_cpu, device="cuda")
        
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            C = torch.matmul(A, B)
            
        # Et le transfert PCIe (Retour)
        C_cpu = C.cpu().numpy()
        torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        
        tflops = ((2.0 * N * N * N) / total_time) / 1e12
        vram = torch.cuda.max_memory_allocated() / 1e9
        
        # Calcul de l'erreur sur un petit échantillon (100x100)
        test_size = 100
        A_tile = A_cpu[:test_size, :].astype(np.float32)
        B_tile = B_cpu[:, :test_size].astype(np.float32)
        C_tile = C_cpu[:test_size, :test_size].astype(np.float32)
        C_exact = np.dot(A_tile, B_tile)
        max_err = float(np.abs(C_tile - C_exact).max())
        
        del A, B, C, A_cpu, B_cpu, C_cpu
        torch.cuda.empty_cache()
        return total_time, tflops, vram, max_err
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None, None
    except Exception as e:
        print(f"    [ERREUR PyTorch] {e}")
        torch.cuda.empty_cache()
        return None, None, None, None

def run_incore_cublas(N):
    cp.get_default_memory_pool().free_all_blocks()
    try:
        # On génère directement en float32
        rng = np.random.default_rng(42)
        A_cpu = rng.random((N, N), dtype=np.float32).astype(np.float16)
        B_cpu = rng.random((N, N), dtype=np.float32).astype(np.float16)
        
        cp.cuda.Stream.null.synchronize()
        start_time = time.perf_counter()
        
        # Transfert PCIe (Aller)
        A = cp.asarray(A_cpu)
        B = cp.asarray(B_cpu)
        
        # Calcul GPU
        C = cp.matmul(A, B)
        
        # Transfert PCIe (Retour)
        C_cpu = C.get()
        cp.cuda.Stream.null.synchronize()
        total_time = time.perf_counter() - start_time
        
        tflops = ((2.0 * N * N * N) / total_time) / 1e12
        vram = cp.get_default_memory_pool().used_bytes() / 1e9
        
        test_size = 100
        A_tile = A_cpu[:test_size, :].astype(np.float32)
        B_tile = B_cpu[:, :test_size].astype(np.float32)
        C_tile = C_cpu[:test_size, :test_size].astype(np.float32)
        C_exact = np.dot(A_tile, B_tile)
        max_err = float(np.abs(C_tile - C_exact).max())
        
        del A, B, C, A_cpu, B_cpu, C_cpu
        cp.get_default_memory_pool().free_all_blocks()
        return total_time, tflops, vram, max_err
    except cp.cuda.memory.OutOfMemoryError:
        cp.get_default_memory_pool().free_all_blocks()
        return None, None, None, None
    except Exception as e:
        print(f"    [ERREUR CuPy] {e}")
        cp.get_default_memory_pool().free_all_blocks()
        return None, None, None, None

async def run_gpusync(N, algo_id):
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    if algo_id == "cuda_double_buffering":
        backend, mod, dtype_in, dtype_out = "cublas", mod_cublas, np.float16, np.float16
        MAX_CHUNK = 8192 # On peut passer à 8192 sans exploser les entiers 32-bits car on est en float16 (2 fois moins lourd)
    elif algo_id == "cuda_wmma_2D_tiling":
        backend, mod, dtype_in, dtype_out = "cuda", mod_wmma_2d, np.float32, np.float32
        MAX_CHUNK = 8192
    elif algo_id == "cuda_wmma":
        backend, mod, dtype_in, dtype_out = "cuda", mod_wmma, np.float32, np.float32
        MAX_CHUNK = 4096
    elif algo_id == "gpu_2D_tiling (cuda)":
        backend, mod, dtype_in, dtype_out = "cuda", mod_gpu_2d, np.float32, np.float32
        MAX_CHUNK = 4096
    elif algo_id == "gpu_2D_tiling (opencl)":
        backend, mod, dtype_in, dtype_out = "opencl", mod_gpu_2d, np.float32, np.float32
    elif algo_id == "gpu (cuda)":
        backend, mod, dtype_in, dtype_out = "cuda", mod_gpu, np.float32, np.float32
    elif algo_id == "gpu (opencl)":
        backend, mod, dtype_in, dtype_out = "opencl", mod_gpu, np.float32, np.float32
    else:
        raise ValueError(f"Algorithme inconnu: {algo_id}")
        
    bytes_per_element = 2 if dtype_in == np.float16 else 4
    total_memory_required_bytes = 3 * N * N * bytes_per_element
    VRAM_LIMIT_BYTES = 13.5 * 1024**3 # ~13.5 GB threshold pour la RTX 4080 (laissant 2.5 Go à Windows)
    
    is_incore = total_memory_required_bytes < VRAM_LIMIT_BYTES
    
    if is_incore:
        MAX_CHUNK = N
        IN_FLIGHT_LIMIT = 1
    else:
        IN_FLIGHT_LIMIT = 2
        if algo_id == "cuda_wmma_2D_tiling":
            MAX_CHUNK = 8192
        elif "2D_tiling" in algo_id or algo_id == "cuda_wmma" or algo_id == "cuda_double_buffering":
            MAX_CHUNK = 4096
        else:
            MAX_CHUNK = 2048 # Prévention TDR
        
    scheduler = Scheduler(backend=backend)
    await scheduler.initialize()
    
    # On génère directement en float32 pour diviser le pic de RAM par 2
    rng = np.random.default_rng(42)
    A = rng.random((N, N), dtype=np.float32).astype(dtype_in)
    B = rng.random((N, N), dtype=np.float32).astype(dtype_in)
    
    start_time = time.perf_counter()

    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    dummy_out = np.zeros(MAX_CHUNK * MAX_CHUNK, dtype=dtype_out)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_out))
        
    C_padded = np.zeros((N, N), dtype=dtype_out)
    num_chunks = math.ceil(N / MAX_CHUNK)
    active_tasks = set()
    
    for i in range(num_chunks):
        start_M = i * MAX_CHUNK
        end_M = min((i + 1) * MAX_CHUNK, N)
        current_CHUNK_M = end_M - start_M
        
        a_sub = await asyncio.to_thread(lambda s=start_M, e=end_M: A[s:e, :].flatten())
        gpu_A = await AsyncBuffer.to_device(scheduler, a_sub)
        
        for j in range(num_chunks):
            start_N = j * MAX_CHUNK
            end_N = min((j + 1) * MAX_CHUNK, N)
            current_CHUNK_N = end_N - start_N
            
            b_sub = await asyncio.to_thread(lambda s=start_N, e=end_N: B[:, s:e].copy().flatten())
            
            task = asyncio.create_task(mod.process_block(
                i, j, gpu_A, b_sub, N, current_CHUNK_M, current_CHUNK_N, start_M, start_N, scheduler, pool
            ))
            active_tasks.add(task)
            
            while len(active_tasks) >= IN_FLIGHT_LIMIT:
                done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                for coro in done:
                    sm, sn, cm, cn, res = coro.result()
                    C_padded[sm:sm+cm, sn:sn+cn] = res.reshape((MAX_CHUNK, MAX_CHUNK))[:cm, :cn]

    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            sm, sn, cm, cn, res = coro.result()
            C_padded[sm:sm+cm, sn:sn+cn] = res.reshape((MAX_CHUNK, MAX_CHUNK))[:cm, :cn]
            
    total_time = time.perf_counter() - start_time
    tflops = ((2.0 * N * N * N) / total_time) / 1e12
    vram = cp.get_default_memory_pool().used_bytes() / 1e9
    
    test_size = 100
    A_tile = A[:test_size, :].astype(np.float32)
    B_tile = B[:, :test_size].astype(np.float32)
    C_tile = C_padded[:test_size, :test_size].astype(np.float32)
    C_exact = np.dot(A_tile, B_tile)
    max_err = float(np.abs(C_tile - C_exact).max())
    
    del A, B, C_padded, pool
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    
    return total_time, tflops, vram, max_err

async def main():
    sizes = [4096, 8192, 16384, 32768, 49152, 65536, 65536*2]
    
    
    """algorithms = [
        "pytorch sur gpu",
        "cuda_incore_cublas",
        "cuda_double_buffering",
        "cuda_wmma_2D_tiling",
        "cuda_wmma",
        "gpu_2D_tiling (cuda)",
        "gpu_2D_tiling (opencl)",
        "gpu (cuda)",
        "gpu (opencl)"
    ]"""
    algorithms = [
        "cuda_double_buffering",
        "cuda_incore_cublas",
        "cuda_wmma_2D_tiling",
        "gpu_2D_tiling (opencl)",
        "pytorch sur gpu"
    ]
    
    # Création du répertoire de résultats
    output_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'results.csv')
    
    # Initialisation du fichier CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Matrix_Size", "Time_s", "TFLOPS", "VRAM_GB", "RAM_GB", "Max_Error", "Status"])
        
    print(f"==================================================")
    print(f"--- DEMARRAGE DU BENCHMARK UNIVERSEL (9 ALGOS)")
    print(f"--- Fichier de sortie : {csv_file}")
    print(f"==================================================\n")
    
    for N in sizes:
        print(f"\n--- [Taille de Matrice : {N}x{N}] ---")
        for algo in algorithms:
            print(f"  > Exécution de '{algo}'...")
            
            ram_before = get_ram_usage()
            
            try:
                if algo == "pytorch sur gpu":
                    t, tflops, vram, err = run_pytorch(N)
                elif algo == "cuda_incore_cublas":
                    t, tflops, vram, err = run_incore_cublas(N)
                else:
                    t, tflops, vram, err = await run_gpusync(N, algo)
                    
                ram_after = get_ram_usage()
                ram = max(ram_before, ram_after)
                
                if t is None:
                    status = "OOM"
                    t_str, tflops_str, vram_str, ram_str, err_str = "", "", "", "", ""
                    print(f"    [ÉCHEC] Out Of Memory (OOM)")
                else:
                    status = "SUCCESS"
                    t_str, tflops_str, vram_str, ram_str, err_str = f"{t:.2f}", f"{tflops:.2f}", f"{vram:.2f}", f"{ram:.2f}", f"{err:.6e}"
                    print(f"    [SUCCÈS] Temps: {t_str}s | Débit: {tflops_str} TFLOPS | VRAM: {vram_str} Go | RAM: {ram_str} Go | Erreur: {err_str}")
                    
            except Exception as e:
                print(f"    [ERREUR FATALE] {e}")
                status = "ERROR"
                t_str, tflops_str, vram_str, ram_str, err_str = "", "", "", "", ""
                
            # Écriture immédiate dans le CSV pour sauvegarder les données en temps réel
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([algo, N, t_str, tflops_str, vram_str, ram_str, err_str, status])

if __name__ == "__main__":
    asyncio.run(main())
