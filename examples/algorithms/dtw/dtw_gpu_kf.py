import time
import asyncio
# pyrefly: ignore [missing-import]
import numpy as np
import sys
import os

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer

from examples.algorithms.dtw.dtw_gpu_engine import get_batch_dtw_kernel

# Création du Kernel compilé (M=32, num_refs=100) pour préserver le setup exact
batch_dtw_fused_constant = get_batch_dtw_kernel(32, 100)

async def process_chunk_kf(chunk_id, offset, chunk_size, gpu_refs_all, gpu_targets_chunk, scheduler, pool, t, out_matrix):
    """
    Exécute le noyau fusionné pour tout le chunk de cibles et rapatrie le bloc d'outputs.
    """
    gpu_outputs_batch = await pool.get()
    
    await batch_dtw_fused_constant(
        gpu_refs_all, gpu_targets_chunk, 0, 32, gpu_outputs_batch,
        t, chunk_size, global_size=(chunk_size,), scheduler=scheduler
    )
        
    res_flat = await gpu_outputs_batch.to_host()
    res_matrix = res_flat[:t * chunk_size].reshape(t, chunk_size)
    
    # Copie dans la matrice de sortie globale
    np.copyto(out_matrix[:, offset : offset + chunk_size], res_matrix)
    
    pool.put_nowait(gpu_outputs_batch)
    return chunk_size

async def main():
    os.environ["PYOPENCL_NO_CACHE"] = "True"
    scheduler = Scheduler(backend="opencl")
    await scheduler.initialize()

    n = 30
    t = 100
    s = 1024 * 1024 * 2 # 2 Millions

    # CHUNK_SIZE réduit à 100 000 (comme dans Phnx V2 / dtw_fast) pour diviser le pool d'outputs par 10 (40 Mo par buffer)
    CHUNK_SIZE = 100_000
    NUM_CHUNKS = s // CHUNK_SIZE
    if s % CHUNK_SIZE != 0:
        NUM_CHUNKS += 1

    np.random.seed(1)
    print("[1] Génération des données...")
    zz0 = np.random.random((t, n)).astype(np.float32)
    zz1 = np.random.random((s, n)).astype(np.float32)

    zz0_pad = np.zeros((t, 32), dtype=np.float32)
    zz0_pad[:, :n] = zz0
    
    zz1_pad = np.zeros((s, 32), dtype=np.float32)
    zz1_pad[:, :n] = zz1

    start_time = time.perf_counter()
    print("[2] Upload des clusters en Constant VRAM...")
    gpu_refs_all = await AsyncBuffer.to_device(scheduler, zz0_pad.flatten())
        
    print("[3] Initialisation du pool d'outputs...")
    # Pool de taille 20 buffers pour paralléliser les transferts (20 * 40 Mo = 800 Mo total)
    IN_FLIGHT_LIMIT = 20
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    dummy_out = np.zeros(t * CHUNK_SIZE, dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_out))
        
    print(f"\n[4] Démarrage de GalTech V3 Kernel Fusion (Gestion type dtw_fast)...")
    matrice_distances = np.zeros((t, s), dtype=np.float32)
    active_tasks = set()
    completed_targets = 0
    
    zz1_flat = zz1_pad.reshape(-1)
    
    for chunk_id in range(NUM_CHUNKS):
        offset = chunk_id * CHUNK_SIZE
        current_chunk_size = min(CHUNK_SIZE, s - offset)
        
        chunk_data = zz1_flat[offset * 32 : (offset + current_chunk_size) * 32]
        gpu_targets_chunk = await AsyncBuffer.to_device(scheduler, chunk_data)
        
        task = asyncio.create_task(process_chunk_kf(
            chunk_id, offset, current_chunk_size,
            gpu_refs_all, gpu_targets_chunk,
            scheduler, pool, t, matrice_distances
        ))
        active_tasks.add(task)
        
        while len(active_tasks) >= IN_FLIGHT_LIMIT:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for coro in done:
                res_len = coro.result()
                completed_targets += res_len
                elapsed = time.perf_counter() - start_time
                speed = (completed_targets * t) / elapsed
                sys.stdout.write(f"\rProgress: {completed_targets:,} / {s:,} | Débit: {speed:,.0f} / sec")
                sys.stdout.flush()

    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            res_len = coro.result()
            completed_targets += res_len
            elapsed = time.perf_counter() - start_time
            speed = (completed_targets * t) / elapsed
            sys.stdout.write(f"\rProgress: {completed_targets:,} / {s:,} | Débit: {speed:,.0f} / sec")
            sys.stdout.flush()
            
    print(f"\n\n[SUCCESS] GalTech V3 KF terminé !")
    print(f"-> Temps Total : {time.perf_counter() - start_time:.2f} secondes.")
    
    matrice_distances = np.sqrt(matrice_distances)
    print(f"Dimension de la matrice : {matrice_distances.shape}")
    print(matrice_distances)

if __name__ == "__main__":
    asyncio.run(main())
