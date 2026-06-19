import os
import sys
import time
import math
import asyncio
import numpy as np
import csv
import cupy as cp
import gc
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
import examples.matrix.matrix_cuda_double_buffering as mod_cublas

def get_ram_usage():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

async def run_experiment(N, max_chunk, in_flight_limit):
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    scheduler = Scheduler(backend="cublas")
    await scheduler.initialize()
    
    # Génération en float32 directe pour économiser la RAM
    rng = np.random.default_rng()
    A = rng.random((N, N), dtype=np.float32)
    B = rng.random((N, N), dtype=np.float32)
    
    pool = asyncio.Queue(maxsize=in_flight_limit)
    dummy_out = np.zeros(max_chunk * max_chunk, dtype=np.float32)
    for _ in range(in_flight_limit):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_out))
        
    C_padded = np.zeros((N, N), dtype=np.float32)
        
    num_chunks = math.ceil(N / max_chunk)
    active_tasks = set()
    
    cp.cuda.Stream.null.synchronize()
    start_time = time.perf_counter()
    
    for i in range(num_chunks):
        start_M = i * max_chunk
        end_M = min((i + 1) * max_chunk, N)
        current_CHUNK_M = end_M - start_M
        
        a_sub = await asyncio.to_thread(lambda s=start_M, e=end_M: A[s:e, :].flatten())
        gpu_A = await AsyncBuffer.to_device(scheduler, a_sub)
        
        for j in range(num_chunks):
            start_N = j * max_chunk
            end_N = min((j + 1) * max_chunk, N)
            current_CHUNK_N = end_N - start_N
            
            b_sub = await asyncio.to_thread(lambda s=start_N, e=end_N: B[:, s:e].copy().flatten())
            
            task = asyncio.create_task(mod_cublas.process_block(
                i, j, gpu_A, b_sub, N, current_CHUNK_M, current_CHUNK_N, start_M, start_N, scheduler, pool
            ))
            active_tasks.add(task)
            
            while len(active_tasks) >= in_flight_limit:
                done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                for coro in done:
                    sm, sn, cm, cn, res = coro.result()
                    C_padded[sm:sm+cm, sn:sn+cn] = res.reshape((max_chunk, max_chunk))[:cm, :cn]

    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            sm, sn, cm, cn, res = coro.result()
            C_padded[sm:sm+cm, sn:sn+cn] = res.reshape((max_chunk, max_chunk))[:cm, :cn]

    cp.cuda.Stream.null.synchronize()
    total_time = time.perf_counter() - start_time
    
    tflops = ((2.0 * N * N * N) / total_time) / 1e12
    vram = cp.get_default_memory_pool().used_bytes() / 1e9
    
    # Calcul de l'erreur
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
    N = 32768 # Taille massive pour forcer un pur Out-Of-Core
    configurations = [
        # (MAX_CHUNK, IN_FLIGHT_LIMIT)
        (2048, 8),  # Petits Chunks, Grande Parallélisation
        (2048, 4),
        (2048, 2),
        (4096, 8),
        (4096, 4),
        (4096, 2),
        (8192, 4),  # Grands Chunks, Moyenne Parallélisation
        (8192, 2),  # Le Sweet Spot théorique
        (8192, 1)   # Pipeline bloquant (sans double buffering)
    ]
    
    output_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'pipeline_tuning.csv')
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["MAX_CHUNK", "IN_FLIGHT_LIMIT", "Time_s", "TFLOPS", "VRAM_GB", "RAM_GB", "Max_Error"])
        
    print(f"==================================================")
    print(f"BENCHMARK PIPELINE: IN_FLIGHT_LIMIT vs MAX_CHUNK")
    print(f"==================================================\n")
    print(f"Matrice de test : {N}x{N} (Out-Of-Core strict)")
    
    for max_chunk, in_flight in configurations:
        print(f"  > Test [CHUNK: {max_chunk:4d} | IN_FLIGHT: {in_flight}] ...")
        
        ram_before = get_ram_usage()
        t, tflops, vram, err = await run_experiment(N, max_chunk, in_flight)
        ram = get_ram_usage() - ram_before
        
        print(f"    [SUCCÈS] Temps: {t:.2f}s | Débit: {tflops:.2f} TFLOPS | VRAM: {vram:.2f} Go | RAM: {ram:.2f} Go | Erreur: {err:e}")
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([max_chunk, in_flight, f"{t:.2f}", f"{tflops:.2f}", f"{vram:.2f}", f"{ram:.2f}", f"{err:e}"])

if __name__ == "__main__":
    asyncio.run(main())
