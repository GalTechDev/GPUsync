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
    Kernel DTW ultra-optimisé avec réduction de l'empreinte registre (Register Spilling prevention).
    """
    target_idx: int = targets_offset + ctx
    target_start: int = target_idx * seq_length
    
    # 1. Copie des targets en registres (30 floats)
    tgt_local = [0.0] * 30
    for j in range(0, seq_length):
        tgt_local[j] = targets[target_start + j]
        
    # La 'reference' est lue directement depuis la VRAM car elle est identique
    # pour tous les threads du bloc. Le cache L1 du GPU va automatiquement 
    # la diffuser (Broadcast) à coût zéro ! Pas besoin de la copier en registre.
    
    # 2. Une seule ligne pour le Dynamic Programming au lieu de deux (30 floats au lieu de 60)
    row = [0.0] * 30
    
    # Première itération (i = 0)
    ref_val = reference[0]
    d = ref_val - tgt_local[0]
    row[0] = d * d
    
    for j in range(1, seq_length):
        d = ref_val - tgt_local[j]
        row[j] = row[j - 1] + d * d
        
    # Itérations suivantes (i > 0)
    for i in range(1, seq_length):
        ref_val = reference[i]
        
        d = ref_val - tgt_local[0]
        m3 = row[0] # Ancien prev_row[0]
        row[0] = row[0] + d * d # Devient curr_row[0]
        m2 = row[0] # curr_row[j-1] pour la prochaine itération
        
        for j in range(1, seq_length):
            d = ref_val - tgt_local[j]
            cost = d * d
            
            m1 = row[j] # prev_row[j]
            
            # min(m1, m2, m3)
            min_val = m1
            if m2 < min_val:
                min_val = m2
            if m3 < min_val:
                min_val = m3
                
            m3 = row[j] # L'actuel prev_row[j] deviendra le prev_row[j-1] au prochain tour
            row[j] = cost + min_val
            m2 = row[j] # L'actuel curr_row[j] deviendra le curr_row[j-1] au prochain tour
            
    outputs[ctx] = row[seq_length - 1]

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
    s = 1024 * 1024 * 64 # 2 Millions d'individus (Le commentaire '64M' était erroné)

    # CHUNK_SIZE = 10 Millions. 
    # Pour 64M, cela fera 7 itérations de la boucle principale. Chaque chunk pèsera ~1.2 Go.
    CHUNK_SIZE = 10_000_000
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
        
    print(f"\n[4] Démarrage du TRUE STREAMING : VRAM augmentée à ~1.2 Go par Chunk (Exploitation des 8 Go) !")
    
    matrice_distances = np.zeros((t, s), dtype=np.float32)
    active_tasks = set()
    
    completed_targets = 0
    total_calculations = t * s
    
    # Aplatissement instantané O(1) pour éviter la copie CPU dans la boucle
    zz1_flat = zz1.reshape(-1)
    
    # INVERSION DES BOUCLES : On parcours le grand dataset, puis les clusters.
    for chunk_id in range(NUM_CHUNKS):
        offset = chunk_id * CHUNK_SIZE
        current_chunk_size = min(CHUNK_SIZE, s - offset)
        
        # Extraction O(1) de la vue (sans copie)
        chunk_data = zz1_flat[offset * n : (offset + current_chunk_size) * n]
        
        # On l'envoie sur le GPU (1.2 Go)
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
