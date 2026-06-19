import time
import asyncio
import numpy as np
import sys
import os
import pyopencl as cl

# Permet d'importer le dossier gpusync parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gpusync.scheduler import Scheduler
from gpusync.memory import AsyncBuffer
from gpusync.compiler import gpusync_kernel

@gpusync_kernel
def batch_dtw_streaming(ctx: int, reference: list, targets: list, targets_offset: int, seq_length: int, outputs: list):
    """
    Kernel DTW ultra-optimisé avec préchargement des registres L1.
    """
    target_idx: int = targets_offset + ctx
    target_start: int = target_idx * seq_length
    
    # Préchargement total en Registres L1 pour couper l'accès à la VRAM
    ref_local = [0.0] * 30
    tgt_local = [0.0] * 30
    
    for j in range(0, seq_length):
        ref_local[j] = reference[j]
        tgt_local[j] = targets[target_start + j]
    
    # Déclaration locale via le compilateur (utilisation des registres GPU)
    prev_row = [0.0] * 30
    curr_row = [0.0] * 30
    
    # Première ligne
    d = ref_local[0] - tgt_local[0]
    curr_row[0] = d * d
    
    for j in range(1, seq_length):
        d = ref_local[0] - tgt_local[j]
        curr_row[j] = curr_row[j - 1] + d * d
        
    for j in range(0, seq_length):
        prev_row[j] = curr_row[j]
        
    # Lignes suivantes
    for i in range(1, seq_length):
        d = ref_local[i] - tgt_local[0]
        curr_row[0] = prev_row[0] + d * d
        
        for j in range(1, seq_length):
            d = ref_local[i] - tgt_local[j]
            cost = d * d
            
            m1 = prev_row[j]
            m2 = curr_row[j - 1]
            m3 = prev_row[j - 1]
            
            min_val = m1
            if m2 < min_val:
                min_val = m2
            if m3 < min_val:
                min_val = m3
                
            curr_row[j] = cost + min_val
            
        for j in range(0, seq_length):
            prev_row[j] = curr_row[j]
            
    outputs[ctx] = curr_row[seq_length - 1]

async def process_chunk(ref_id, chunk_id, offset, chunk_size, gpu_reference, gpu_targets_chunk, scheduler, pool, SEQ_LENGTH):
    """
    Exécute un bloc sur le GPU, récupère les résultats, et libère le buffer de sortie.
    Note : gpu_targets_chunk pointe ici vers un bloc qui commence à 0. Donc targets_offset = 0 pour ce sous-bloc.
    """
    gpu_outputs = await pool.get()
    
    # Attention: targets_offset=0 car on a uploadé seulement ce chunk (il commence à l'index 0 en VRAM)
    await batch_dtw_streaming(gpu_reference, gpu_targets_chunk, 0, SEQ_LENGTH, gpu_outputs, global_size=(chunk_size,), scheduler=scheduler)
    
    results = await gpu_outputs.to_host()
    
    pool.put_nowait(gpu_outputs)
    return ref_id, offset, results[:chunk_size].copy()

async def main():
    scheduler = Scheduler(backend="cuda")
    await scheduler.initialize()

    n = 30
    t = 100
    s = 1024 * 1024 * 2 # 64 Millions (Test Massif)

    # CHUNK_SIZE = 1 Million. 
    # Pour 64M, cela fera 64 itérations de la boucle principale.
    CHUNK_SIZE = 1_000_000
    NUM_CHUNKS = s // CHUNK_SIZE
    if s % CHUNK_SIZE != 0:
        NUM_CHUNKS += 1

    np.random.seed(1)
    print("[1] Génération des données en RAM (Peut prendre quelques secondes)...")
    zz0 = np.random.random((t, n)).astype(np.float32)
    zz1 = np.random.random((s, n)).astype(np.float32)


    start_time = time.perf_counter()
    print("[2] Pré-upload des Clusters (Références) en VRAM...")
    gpu_refs = []
    for ref_id in range(t):
        gpu_refs.append(await AsyncBuffer.to_device(scheduler, zz0[ref_id].flatten()))
        
    print("[3] Pré-allocation du Memory Pool (Outputs)...")
    IN_FLIGHT_LIMIT = 40
    pool = asyncio.Queue(maxsize=IN_FLIGHT_LIMIT)
    dummy_out = np.zeros(CHUNK_SIZE, dtype=np.float32)
    for _ in range(IN_FLIGHT_LIMIT):
        pool.put_nowait(AsyncBuffer.empty_like(scheduler, dummy_out))
        
    print(f"\n[4] Démarrage du TRUE STREAMING : VRAM bridée à ~120 Mo !")
    
    matrice_distances = np.zeros((t, s), dtype=np.float32)
    active_tasks = set()
    
    completed_targets = 0
    total_calculations = t * s
    
    # INVERSION DES BOUCLES : On parcours le grand dataset, puis les clusters.
    for chunk_id in range(NUM_CHUNKS):
        offset = chunk_id * CHUNK_SIZE
        current_chunk_size = min(CHUNK_SIZE, s - offset)
        
        # On extrait seulement le petit bloc de RAM système
        chunk_data = zz1[offset:offset + current_chunk_size].flatten()
        # On l'envoie sur le GPU (120 Mo)
        gpu_targets_chunk = await AsyncBuffer.to_device(scheduler, chunk_data)
        
        for ref_id in range(t):
            task = asyncio.create_task(process_chunk(
                ref_id, chunk_id, offset, current_chunk_size, 
                gpu_refs[ref_id], gpu_targets_chunk, 
                scheduler, pool, n
            ))
            active_tasks.add(task)
            
        # Fenêtre glissante : on s'assure de ne jamais avoir plus de 2 Chunks en VRAM en même temps
        # pour éviter d'exploser la VRAM avec les tâches en attente.
        while len(active_tasks) >= t * 2:
            done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
            for coro in done:
                r_id, off, res = coro.result()
                matrice_distances[r_id, off:off+len(res)] = res
                
                completed_targets += len(res)
                elapsed = time.perf_counter() - start_time
                speed = completed_targets / elapsed
                sys.stdout.write(f"\rProgress: {completed_targets:,} / {total_calculations:,} | Débit: {speed:,.0f} / sec | VRAM: Sécurisée")
                sys.stdout.flush()

    # Vider les dernières tâches restantes
    while len(active_tasks) > 0:
        done, active_tasks = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        for coro in done:
            r_id, off, res = coro.result()
            matrice_distances[r_id, off:off+len(res)] = res
            
            completed_targets += len(res)
            elapsed = time.perf_counter() - start_time
            speed = completed_targets / elapsed
            sys.stdout.write(f"\rProgress: {completed_targets:,} / {total_calculations:,} | Débit: {speed:,.0f} / sec | VRAM: Sécurisée")
            sys.stdout.flush()
            
    print(f"\n\n[SUCCESS] Benchmark True Streaming terminé !")
    print(f"-> Temps Total GPU : {time.perf_counter() - start_time:.2f} secondes.")
    
    print("\n[5] Formatage de la matrice...")
    matrice_distances = np.sqrt(matrice_distances)
    
    print("\n--- MATRICE DES DISTANCES (CLUSTERS x INDIVIDUS) ---")
    print(f"Dimension de la matrice obtenue : {matrice_distances.shape}")
    print("\nAperçu de la matrice:")
    print(matrice_distances)

if __name__ == "__main__":
    asyncio.run(main())
